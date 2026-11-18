"""
解耦多模态生存预测模型（增强版 V2）
在基线模型基础上增加解耦网络，分离公共特征和私有特征

增强说明：
1. UncertaintyEstimator 输入改为 512维原始编码特征（path_emb, omics_emb）
2. 对比损失使用不确定性 Tempering（sigma_sq * InfoNCE per sample）
3. 去除了 sigma L2 正则化（改为由 contrastive tempering 自然学习）
4. 模型深度增加，dropout 结构调整
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any

from .aggregator import PathologyAggregator
from .omic_encoder import OmicsEncoder
from .fusion_head import FusionHead, MultiTaskLoss
from .disentangler import Disentangler, orthogonal_loss
from .contrastive import CrossModalContrastiveLoss
from .uncertainty import UncertaintyEstimator
from .adaptive_fusion import AdaptiveFusion


class DisentangledModel(nn.Module):
    """
    解耦多模态生存预测模型（增强版 V2）

    架构流程：
    1. 病理聚合: (N, 1024) -> path_emb (512)
    2. 组学编码: (17472, 3) -> omics_emb (512)
    3. 不确定性估计（从原始编码特征）: path_emb -> sigma_p, omics_emb -> sigma_o
    4. 解耦: path_emb -> z_p_c(128), z_p_s(128); omics_emb -> z_o_c(128), z_o_s(128)
    5. 自适应融合: z_fused = α_p * z_p_c + α_o * z_o_c (用sigma计算权重)
    6. 预测: z_fused -> risk_score, risk_class
    7. 损失: L_task + λ1*L_dis + λ2*L_cross(sigma_tempered) + (无L_unc)

    设计：
    - σ估计基于原始特征，不经过解耦层，梯度更直接
    - 对比损失中高σ样本的损失被抑制，模型自动学习区分可靠/不可靠对齐
    - 移除σ的L2正则化，避免压制不确定性表达的多样性
    - 解耦输出维度降至128（原256），减少参数量
    """

    def __init__(
        self,
        # 病理聚合器配置
        path_aggregator_mode: str = "mean",
        path_hidden_dim: int = 512,
        path_dropout: float = 0.3,
        # 组学编码器配置
        omics_encoder_type: str = "cnn",
        omics_hidden_dim: int = 512,
        omics_num_layers: int = 3,
        omics_dropout: float = 0.3,
        # 解耦器配置（维度降低至128，缓解过拟合）
        disentangler_input_dim: int = 512,
        disentangler_hidden_dim: int = 256,
        disentangler_output_dim: int = 128,
        # 融合头配置
        fusion_hidden_dim: int = 128,
        fusion_dim: int = 64,
        fusion_dropout: float = 0.3,
        num_classes: int = 3,
        # 解耦损失权重
        disentangle_lambda: float = 0.05,
        # 对比学习配置
        cross_modal_lambda: float = 0.2,
        contrastive_temperature: float = 0.07,
        # 不确定性估计配置
        uncertainty_hidden_dim: int = 256,
        # 自适应融合配置
        fusion_gamma_init: float = 2.0,
        # 是否使用不确定性感知融合
        use_uncertainty_fusion: bool = True,
        # 不确定性正则化（V2中不使用，保留参数兼容旧checkpoint）
        uncertainty_reg_lambda: float = 0.0,
        omics_input_genes: int = 17472,
        use_gene_gating: bool = False,
        # shebei
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化解耦多模态生存预测模型（增强版 V2）

        Args:
            path_aggregator_mode: 病理聚合模式 ("mean" 或 "attention")
            path_hidden_dim: 病理聚合器隐藏维度
            path_dropout: 病理聚合器dropout概率
            omics_encoder_type: 组学编码器类型 ("cnn" 或 "transformer")
            omics_hidden_dim: 组学编码器隐藏维度
            omics_num_layers: 组学编码器层数
            omics_dropout: 组学编码器dropout概率
            disentangler_input_dim: 解耦器输入维度（默认512，接收编码特征）
            disentangler_hidden_dim: 解耦器隐藏层维度（默认256）
            disentangler_output_dim: 解耦器输出维度（默认128，公共/私有各128）
            fusion_hidden_dim: 融合头隐藏维度
            fusion_dim: 融合特征维度
            fusion_dropout: 融合头dropout概率
            num_classes: 风险等级数量
            disentangle_lambda: 解耦损失权重 λ1
            cross_modal_lambda: 跨模态对比损失权重 λ2
            contrastive_temperature: 对比损失温度 τ（默认0.07，CLIP标准值）
            uncertainty_hidden_dim: 不确定性估计器隐藏层维度（默认256）
            fusion_gamma_init: 自适应融合 γ 初始值
            use_uncertainty_fusion: 是否使用不确定性感知融合
            uncertainty_reg_lambda: 不确定性正则化权重（V2中设置0.0关闭）
            device: 设备 ("cuda" 或 "cpu")
        """
        super().__init__()

        self.device = device
        self.disentangle_lambda = disentangle_lambda
        self.cross_modal_lambda = cross_modal_lambda
        self.use_uncertainty_fusion = use_uncertainty_fusion
        self.uncertainty_reg_lambda = uncertainty_reg_lambda

        # ============ 特征编码层 ============

        # 病理特征聚合器 (N, 1024) -> (512)
        self.path_aggregator = PathologyAggregator(
            input_dim=1024,
            hidden_dim=path_hidden_dim,
            output_dim=512,
            aggregation_mode=path_aggregator_mode,
            dropout=path_dropout
        )

        # 组学特征编码器 (17472, 3) -> (512)
        self.omics_encoder = OmicsEncoder(
            input_genes=omics_input_genes,
            input_channels=3,
            hidden_dim=omics_hidden_dim,
            output_dim=512,
            encoder_type=omics_encoder_type,
            use_gene_gating=use_gene_gating,
            num_layers=omics_num_layers,
            dropout=omics_dropout
        )

        # ============ 不确定性估计（从原始编码特征 512→256→1） ============
        # 重要：σ估计基于原始编码特征，而非解耦后特征
        # 这样不确定性可以更直接地反映原始输入的质量
        self.uncertainty_path = UncertaintyEstimator(
            input_dim=512,  # path_emb 的维度
            hidden_dim_1=uncertainty_hidden_dim,
            hidden_dim_2=128,
            output_dim=1,
            dropout=0.2
        )
        self.uncertainty_omics = UncertaintyEstimator(
            input_dim=512,  # omics_emb 的维度
            hidden_dim_1=uncertainty_hidden_dim,
            hidden_dim_2=128,
            output_dim=1,
            dropout=0.2
        )

        # ============ 特征解耦层 512→256→(128, 128) ============
        self.disentangler_path = Disentangler(
            input_dim=disentangler_input_dim,
            hidden_dim=disentangler_hidden_dim,
            output_dim=disentangler_output_dim,
            dropout=0.2
        )

        self.disentangler_omics = Disentangler(
            input_dim=disentangler_input_dim,
            hidden_dim=disentangler_hidden_dim,
            output_dim=disentangler_output_dim,
            dropout=0.2
        )

        # ============ 自适应融合 ============
        self.adaptive_fusion = AdaptiveFusion(
            feature_dim=disentangler_output_dim,
            gamma_init=fusion_gamma_init
        )

        # ============ 融合与预测头（single模式：输入解耦后的公共特征） ============
        self.fusion_head = FusionHead(
            input_dim=disentangler_output_dim,  # 解耦后公共特征维度
            hidden_dim=fusion_hidden_dim,
            fusion_dim=fusion_dim,
            dropout=fusion_dropout,
            num_classes=num_classes,
            fusion_mode="single" if use_uncertainty_fusion else "concat"
        )

        # ============ 损失函数 ============
        self.loss_fn = MultiTaskLoss(cox_weight=1.0, ce_weight=0.5)

        # 跨模态对比损失（默认温度0.07，支持不确定性tempering）
        self.contrastive_loss = CrossModalContrastiveLoss(
            temperature=contrastive_temperature
        )

        # 移动到设备
        self.to(self.device)

    def forward(
        self,
        path_features_list: List[torch.Tensor],
        omics_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            path_features_list: 病理特征列表，每个元素形状 (N_i, 1024)
            omics_features: 组学特征，形状 (batch_size, 17472, 3)

        Returns:
            risk_score: 风险评分，形状 (batch_size, 1)
            risk_class: 风险等级logits，形状 (batch_size, num_classes)
            z_p_c: 病理公共特征，形状 (batch_size, 128)
            z_p_s: 病理私有特征，形状 (batch_size, 128)
            z_o_c: 组学公共特征，形状 (batch_size, 128)
            z_o_s: 组学私有特征，形状 (batch_size, 128)
            sigma_p: 病理不确定性，形状 (batch_size, 1)（基于原始 path_emb）
            sigma_o: 组学不确定性，形状 (batch_size, 1)（基于原始 omics_emb）
            alpha_p: 病理融合权重，形状 (batch_size, 1)
            alpha_o: 组学融合权重，形状 (batch_size, 1)
        """
        # 1. 特征编码
        path_emb = self.path_aggregator(path_features_list)  # (batch, 512)
        omics_emb = self.omics_encoder(omics_features)  # (batch, 512)

        # 2. 不确定性估计（基于原始编码特征）
        sigma_p = self.uncertainty_path(path_emb)    # (batch, 1)
        sigma_o = self.uncertainty_omics(omics_emb)  # (batch, 1)

        # 3. 特征解耦
        z_p_c, z_p_s = self.disentangler_path(path_emb)      # 各 (batch, 128)
        z_o_c, z_o_s = self.disentangler_omics(omics_emb)    # 各 (batch, 128)

        if self.use_uncertainty_fusion:
            # 4. 自适应融合（用σ计算权重）
            z_fused, alpha_p, alpha_o = self.adaptive_fusion(z_p_c, z_o_c, sigma_p, sigma_o)
            # 5. 预测（single模式）
            risk_score, risk_class = self.fusion_head(z_fused)
        else:
            # 回退到原始拼接融合
            alpha_p = torch.full((z_p_c.size(0), 1), 0.5, device=self.device)
            alpha_o = torch.full((z_o_c.size(0), 1), 0.5, device=self.device)
            risk_score, risk_class = self.fusion_head(z_p_c, z_o_c)

        return (risk_score, risk_class, 
                z_p_c, z_p_s, z_o_c, z_o_s,
                sigma_p, sigma_o, alpha_p, alpha_o)

    def compute_loss(
        self,
        risk_score: torch.Tensor,
        risk_class: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
        z_p_c: torch.Tensor,
        z_p_s: torch.Tensor,
        z_o_c: torch.Tensor,
        z_o_s: torch.Tensor,
        sigma_p: Optional[torch.Tensor] = None,
        sigma_o: Optional[torch.Tensor] = None,
        risk_labels: Optional[torch.Tensor] = None,
        **kwargs  # 接受额外参数，保持向后兼容
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算总损失

        L = L_task + λ1 * L_dis + λ2 * L_cross(sigma_tempered) + λ3 * L_unc (λ3 = 0 in V2)

        关键变化：
        - 对比损失使用不确定性 Tempering：高σ样本损失被抑制
        - 不确定性正则化 λ3=0：不再用L2压制σ，改为通过对比损失自然学习
        - σ的信息含量来自对比损失的选择性抑制

        Args:
            risk_score: 风险评分，形状 (batch_size, 1)
            risk_class: 风险等级logits，形状 (batch_size, num_classes)
            time: 生存时间，形状 (batch_size,)
            event: 事件指示器，形状 (batch_size,)
            z_p_c: 病理公共特征 (batch, 128)
            z_p_s: 病理私有特征 (batch, 128)
            z_o_c: 组学公共特征 (batch, 128)
            z_o_s: 组学私有特征 (batch, 128)
            sigma_p: 病理不确定性 (batch, 1)，用于对比损失tempering
            sigma_o: 组学不确定性 (batch, 1)，用于对比损失tempering
            risk_labels: 风险等级标签 (可选)

        Returns:
            total_loss: 总损失
            cox_loss: Cox损失
            ce_loss: 交叉熵损失
            dis_loss: 解耦损失
            cross_loss: 对比损失（含不确定性tempering）
            pos_sim: 平均正样本相似度
            unc_loss: 不确定性正则化损失（V2中为0）
        """
        # 确保数据在正确的设备上
        time = time.to(self.device)
        event = event.to(self.device)
        if risk_labels is not None:
            risk_labels = risk_labels.to(self.device)

        # 1. 任务损失（Cox + 分类）
        total_loss, cox_loss, ce_loss = self.loss_fn(
            risk_score, risk_class, time, event, risk_labels
        )

        # 2. 解耦损失：公共/私有特征正交
        dis_loss_path = orthogonal_loss(z_p_c, z_p_s)
        dis_loss_omics = orthogonal_loss(z_o_c, z_o_s)
        dis_loss = (dis_loss_path + dis_loss_omics) / 2.0

        # 3. 跨模态对比损失（含不确定性Tempering）
        # 当sigma_p, sigma_o传入时，contrastive_loss会自动进行tempering
        # 机制：高σ样本的对比损失被σ²抑制，模型学会用σ表示对齐可靠性
        cross_loss, pos_sim = self.contrastive_loss(
            z_p_c, z_o_c, 
            sigma_p=sigma_p, sigma_o=sigma_o
        )

        # 4. 不确定性正则化（V2中关闭：uncertainty_reg_lambda=0）
        unc_loss = torch.tensor(0.0, device=self.device)

        # 总损失
        total_loss = (total_loss
                      + self.disentangle_lambda * dis_loss
                      + self.cross_modal_lambda * cross_loss
                      + self.uncertainty_reg_lambda * unc_loss)

        return total_loss, cox_loss, ce_loss, dis_loss, cross_loss, pos_sim, unc_loss

    def predict(
        self,
        path_features_list: List[torch.Tensor],
        omics_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        预测接口

        Args:
            path_features_list: 病理特征列表
            omics_features: 组学特征

        Returns:
            字典，包含:
            - risk_score: (batch, 1)
            - risk_class_logits: (batch, num_classes)
            - risk_level: (batch,)
            - z_p_c: 病理公共特征 (batch, 128)
            - z_o_c: 组学公共特征 (batch, 128)
            - sigma_p: 病理不确定性 (batch, 1)
            - sigma_o: 组学不确定性 (batch, 1)
            - alpha_p: 病理权重 (batch, 1)
            - alpha_o: 组学权重 (batch, 1)
        """
        self.eval()
        with torch.no_grad():
            risk_score, risk_class, z_p_c, z_p_s, z_o_c, z_o_s, sigma_p, sigma_o, alpha_p, alpha_o = \
                self.forward(path_features_list, omics_features)
            risk_level = self.fusion_head.predict_risk_level(risk_class)

        return {
            'risk_score': risk_score,
            'risk_class_logits': risk_class,
            'risk_level': risk_level,
            'z_p_c': z_p_c,
            'z_o_c': z_o_c,
            'sigma_p': sigma_p,
            'sigma_o': sigma_o,
            'alpha_p': alpha_p,
            'alpha_o': alpha_o
        }

    def get_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            'path_aggregator_mode': self.path_aggregator.aggregation_mode,
            'path_hidden_dim': self.path_aggregator.hidden_dim,
            'path_dropout': self.path_aggregator.dropout,
            'omics_encoder_type': self.omics_encoder.encoder_type,
            'omics_hidden_dim': self.omics_encoder.hidden_dim,
            'omics_num_layers': self.omics_encoder.num_layers,
            'omics_dropout': 0.1,
            'disentangler_input_dim': self.disentangler_path.input_dim,
            'disentangler_hidden_dim': self.disentangler_path.hidden_dim,
            'disentangler_output_dim': self.disentangler_path.output_dim,
            'fusion_hidden_dim': self.fusion_head.hidden_dim,
            'fusion_dim': self.fusion_head.fusion_dim,
            'fusion_dropout': 0.1,
            'num_classes': self.fusion_head.num_classes,
            'disentangle_lambda': self.disentangle_lambda,
            'cross_modal_lambda': self.cross_modal_lambda,
            'contrastive_temperature': self.contrastive_loss.temperature,
            'uncertainty_hidden_dim': 256,
            'fusion_gamma_init': 2.0,
            'uncertainty_reg_lambda': self.uncertainty_reg_lambda,
            'use_uncertainty_fusion': self.use_uncertainty_fusion,
            'omics_input_genes': self.omics_encoder.input_genes,
            'use_gene_gating': getattr(self.omics_encoder, 'use_gene_gating', False),
            'device': self.device
        }

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.get_config()
        }, path)
        print(f"模型已保存到: {path}")

    @classmethod
    def load(cls, path: str, device: str = None):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        if device is not None:
            config['device'] = device
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(config['device'])
        print(f"模型已从 {path} 加载")
        return model


def test_disentangled_model():
    """测试解耦多模态生存预测模型（增强版 V2）"""
    import numpy as np
    import tempfile
    import os

    torch.manual_seed(42)

    batch_size = 4
    path_features_list = []
    for i in range(batch_size):
        N = np.random.randint(10, 100)
        features = torch.randn(N, 1024)
        path_features_list.append(features)

    omics_features = torch.randn(batch_size, 17472, 3)
    time = torch.rand(batch_size) * 100
    event = torch.randint(0, 2, (batch_size,))
    risk_labels = torch.randint(0, 3, (batch_size,))

    print(f"病理特征列表长度: {len(path_features_list)}")
    print(f"组学特征形状: {omics_features.shape}")
    print(f"生存时间形状: {time.shape}")
    print(f"事件指示器形状: {event.shape}")

    # 测试模型创建和前向
    print("\n测试模型创建和前向传播...")
    model = DisentangledModel(
        path_aggregator_mode="mean",
        omics_encoder_type="cnn",
        disentangle_lambda=0.05,
        cross_modal_lambda=0.2,
        contrastive_temperature=0.07,
        use_uncertainty_fusion=True,
        uncertainty_reg_lambda=0.0
    )

    # 前向传播
    (risk_score, risk_class, z_p_c, z_p_s, z_o_c, z_o_s,
     sigma_p, sigma_o, alpha_p, alpha_o) = model(path_features_list, omics_features)

    print(f"风险评分形状: {risk_score.shape}")
    print(f"风险等级logits形状: {risk_class.shape}")
    print(f"病理公共特征形状: {z_p_c.shape}")
    print(f"病理私有特征形状: {z_p_s.shape}")
    print(f"组学公共特征形状: {z_o_c.shape}")
    print(f"组学私有特征形状: {z_o_s.shape}")
    print(f"σ_p 范围: [{sigma_p.min().item():.4f}, {sigma_p.max().item():.4f}] 均值: {sigma_p.mean().item():.4f}")
    print(f"σ_o 范围: [{sigma_o.min().item():.4f}, {sigma_o.max().item():.4f}] 均值: {sigma_o.mean().item():.4f}")
    print(f"α_p 范围: [{alpha_p.min().item():.4f}, {alpha_p.max().item():.4f}]")
    print(f"α_o 范围: [{alpha_o.min().item():.4f}, {alpha_o.max().item():.4f}]")

    # 验证输出维度（新：128维解耦特征）
    assert risk_score.shape == (batch_size, 1)
    assert risk_class.shape == (batch_size, 3)
    assert z_p_c.shape == (batch_size, 128), f"z_p_c shape: {z_p_c.shape}"
    assert z_p_s.shape == (batch_size, 128)
    assert z_o_c.shape == (batch_size, 128)
    assert z_o_s.shape == (batch_size, 128)
    assert sigma_p.shape == (batch_size, 1)
    assert sigma_o.shape == (batch_size, 1)
    assert alpha_p.shape == (batch_size, 1)
    assert alpha_o.shape == (batch_size, 1)
    assert torch.allclose(alpha_p + alpha_o, torch.ones_like(alpha_p), atol=1e-5)

    # 计算损失
    print("\n计算损失...")
    total_loss, cox_loss, ce_loss, dis_loss, cross_loss, pos_sim, unc_loss = \
        model.compute_loss(risk_score, risk_class, time, event,
                          z_p_c, z_p_s, z_o_c, z_o_s,
                          sigma_p, sigma_o, risk_labels)

    print(f"总损失: {total_loss.item():.4f}")
    print(f"Cox损失: {cox_loss.item():.4f}")
    print(f"交叉熵损失: {ce_loss.item():.4f}")
    print(f"解耦损失: {dis_loss.item():.4f}")
    print(f"对比损失(含tempering): {cross_loss.item():.4f}")
    print(f"正样本相似度: {pos_sim.item():.4f}")
    print(f"不确定性正则化: {unc_loss.item():.4f}")

    # 测试无Tempering对比
    print("\n测试无Tempering对比损失...")
    cross_loss_no_temp, pos_sim_no_temp = model.contrastive_loss(z_p_c, z_o_c)
    print(f"  无Tempering对比损失: {cross_loss_no_temp.item():.4f}")

    # 测试预测接口
    print("\n测试预测接口...")
    predictions = model.predict(path_features_list, omics_features)
    print(f"风险等级预测: {predictions['risk_level']}")
    print(f"预测中 σ_p 形状: {predictions['sigma_p'].shape}")

    # 测试保存和加载
    print("\n测试模型保存和加载...")
    model.eval()
    with torch.no_grad():
        (r1, c1, *_) = model(path_features_list, omics_features)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model.pth")
        model.save(save_path)
        loaded_model = DisentangledModel.load(save_path)
        loaded_model.eval()

        with torch.no_grad():
            (r2, c2, *_) = loaded_model(path_features_list, omics_features)
        assert torch.allclose(r1, r2, rtol=1e-4)
        assert torch.allclose(c1, c2, rtol=1e-4)
        print("模型保存和加载测试通过!")

    print("\n所有测试通过!")


if __name__ == "__main__":
    print("=" * 60)
    print("测试 DisentangledModel (V2)")
    print("=" * 60)
    test_disentangled_model()