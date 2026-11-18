"""
自适应融合门
根据特征质量和不确定性动态调整融合权重

V2更新：
- feature_dim默认值改为128（匹配解耦输出维度降低）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AdaptiveFusion(nn.Module):
    """
    自适应融合门

    根据两个模态的特征质量和不确定性，动态计算融合权重。

    输入：
    - z_p_c: 病理公共特征，形状 (batch, 128)
    - z_o_c: 组学公共特征，形状 (batch, 128)
    - σ_p: 病理不确定性，形状 (batch, 1) 或 (batch, D)
    - σ_o: 组学不确定性，形状 (batch, 1) 或 (batch, D)

    权重计算：
    α = softmax([W_p·z_p_c - γ·log(σ_p), W_o·z_o_c - γ·log(σ_o)])

    其中：
    - W_p, W_o: 可学习的线性投影 (128 → 1)
    - γ: 可学习标量，控制不确定性对权重的影响程度
    - log(σ) 为负值（因为 σ ∈ (0,1)），所以 -γ·log(σ) 是正值
    - 不确定性越低（σ 越小），-γ·log(σ) 越大，该模态权重越高
    - 向量σ通过可学习聚合器降为标量

    融合输出：
    z_fused = α_p * z_p_c + α_o * z_o_c
    """

    def __init__(
        self,
        feature_dim: int = 128,  # V2: 默认改为128
        gamma_init: float = 2.0,  # V2: 默认改为2.0，放大不确定性影响
    ):
        """
        初始化自适应融合门

        Args:
            feature_dim: 特征维度（默认128，匹配解耦输出）
            gamma_init: γ 初始值（默认2.0，更强调不确定性）
        """
        super().__init__()

        self.feature_dim = feature_dim

        # 可学习的线性投影：将特征映射到标量logit
        self.W_p = nn.Linear(feature_dim, 1, bias=False)
        self.W_o = nn.Linear(feature_dim, 1, bias=False)

        # 可学习标量 γ：控制不确定性对权重的影响
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        z_p_c: torch.Tensor,
        z_o_c: torch.Tensor,
        sigma_p: torch.Tensor,
        sigma_o: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            z_p_c: 病理公共特征，形状 (batch_size, feature_dim)
            z_o_c: 组学公共特征，形状 (batch_size, feature_dim)
            sigma_p: 病理不确定性，形状 (batch_size, 1)，值域 (0,1)
            sigma_o: 组学不确定性，形状 (batch_size, 1)，值域 (0,1)

        Returns:
            z_fused: 融合特征，形状 (batch_size, feature_dim)
            alpha_p: 病理权重，形状 (batch_size, 1)
            alpha_o: 组学权重，形状 (batch_size, 1)
        """
        # 确保σ值域安全（避免log(0)和log爆炸）
        sigma_p = torch.clamp(sigma_p, min=1e-6, max=10.0)
        sigma_o = torch.clamp(sigma_o, min=1e-6, max=10.0)

        # 计算每个模态的logit
        # logit_p = W_p·z_p_c - γ·log(σ_p)
        # logit_o = W_o·z_o_c - γ·log(σ_o)
        logit_p = self.W_p(z_p_c) - self.gamma * torch.log(sigma_p)  # (batch, 1)
        logit_o = self.W_o(z_o_c) - self.gamma * torch.log(sigma_o)  # (batch, 1)

        # 拼接logits并计算softmax权重
        logits = torch.cat([logit_p, logit_o], dim=1)  # (batch, 2)
        alphas = F.softmax(logits, dim=1)  # (batch, 2)

        alpha_p = alphas[:, 0:1]  # (batch, 1)
        alpha_o = alphas[:, 1:2]  # (batch, 1)

        # 加权融合
        z_fused = alpha_p * z_p_c + alpha_o * z_o_c  # (batch, feature_dim)

        return z_fused, alpha_p, alpha_o

    def get_gamma(self) -> float:
        """获取当前 γ 值"""
        return self.gamma.item()

    def get_weights(
        self,
        z_p_c: torch.Tensor,
        z_o_c: torch.Tensor,
        sigma_p: torch.Tensor,
        sigma_o: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取详细的权重信息（用于分析）

        Returns:
            alpha_p: 病理权重
            alpha_o: 组学权重
            logit_p: 病理logit
            logit_o: 组学logit
        """
        logit_p = self.W_p(z_p_c) - self.gamma * torch.log(sigma_p)
        logit_o = self.W_o(z_o_c) - self.gamma * torch.log(sigma_o)
        logits = torch.cat([logit_p, logit_o], dim=1)
        alphas = F.softmax(logits, dim=1)
        return alphas[:, 0:1], alphas[:, 1:2], logit_p, logit_o

    def get_config(self) -> dict:
        """获取配置"""
        return {
            'feature_dim': self.feature_dim,
            'gamma_init': self.gamma.item(),
        }


def test_adaptive_fusion():
    """测试自适应融合门"""
    # 设置随机种子
    torch.manual_seed(42)

    batch_size = 8
    feature_dim = 128  # V2: 128维

    # 创建模拟数据
    z_p_c = torch.randn(batch_size, feature_dim)
    z_o_c = torch.randn(batch_size, feature_dim)

    print(f"z_p_c 形状: {z_p_c.shape}")
    print(f"z_o_c 形状: {z_o_c.shape}")

    # 测试自适应融合
    print("\n测试自适应融合门...")
    fusion = AdaptiveFusion(feature_dim=feature_dim, gamma_init=2.0)

    # 测试1：两个模态不确定性相同
    sigma_p = torch.full((batch_size, 1), 0.5)
    sigma_o = torch.full((batch_size, 1), 0.5)
    z_fused, alpha_p, alpha_o = fusion(z_p_c, z_o_c, sigma_p, sigma_o)

    print(f"相同不确定性 (σ=0.5):")
    print(f"  α_p 均值: {alpha_p.mean().item():.4f}, α_o 均值: {alpha_o.mean().item():.4f}")
    print(f"  z_fused 形状: {z_fused.shape}")
    print(f"  γ 值: {fusion.get_gamma():.4f}")

    # 验证权重和为1
    assert torch.allclose(alpha_p + alpha_o, torch.ones_like(alpha_p), atol=1e-6), \
        "权重和不为1"

    # 测试2：病理高确定性，组学低确定性
    sigma_p_high = torch.full((batch_size, 1), 0.1)  # 高确定性
    sigma_o_low = torch.full((batch_size, 1), 0.9)   # 低确定性
    z_fused2, alpha_p2, alpha_o2 = fusion(z_p_c, z_o_c, sigma_p_high, sigma_o_low)

    print(f"\n病理高确定性 (σ=0.1), 组学低确定性 (σ=0.9):")
    print(f"  α_p 均值: {alpha_p2.mean().item():.4f}, α_o 均值: {alpha_o2.mean().item():.4f}")

    # 病理权重应该更高
    assert alpha_p2.mean() > alpha_o2.mean(), \
        "高确定性模态的权重应该更高"

    # 测试3：组学高确定性，病理低确定性
    sigma_p_low = torch.full((batch_size, 1), 0.9)   # 低确定性
    sigma_o_high = torch.full((batch_size, 1), 0.1)  # 高确定性
    z_fused3, alpha_p3, alpha_o3 = fusion(z_p_c, z_o_c, sigma_p_low, sigma_o_high)

    print(f"\n病理低确定性 (σ=0.9), 组学高确定性 (σ=0.1):")
    print(f"  α_p 均值: {alpha_p3.mean().item():.4f}, α_o 均值: {alpha_o3.mean().item():.4f}")

    # 组学权重应该更高
    assert alpha_o3.mean() > alpha_p3.mean(), \
        "高确定性模态的权重应该更高"

    # 测试4：极端情况 - 两个模态都非常确定
    sigma_p_extreme = torch.full((batch_size, 1), 0.01)
    sigma_o_extreme = torch.full((batch_size, 1), 0.01)
    z_fused4, alpha_p4, alpha_o4 = fusion(z_p_c, z_o_c, sigma_p_extreme, sigma_o_extreme)

    print(f"\n两个模态都非常确定 (σ=0.01):")
    print(f"  α_p 均值: {alpha_p4.mean().item():.4f}, α_o 均值: {alpha_o4.mean().item():.4f}")

    # 测试5：验证输出维度
    assert z_fused.shape == (batch_size, feature_dim), \
        f"融合特征形状错误: {z_fused.shape}"
    assert alpha_p.shape == (batch_size, 1), \
        f"α_p 形状错误: {alpha_p.shape}"
    assert alpha_o.shape == (batch_size, 1), \
        f"α_o 形状错误: {alpha_o.shape}"

    # 测试6：测试 γ 的可训练性
    print(f"\nγ 是否可训练: {fusion.gamma.requires_grad}")

    # 测试7：测试 get_weights 方法
    alphas_p, alphas_o, logit_p, logit_o = fusion.get_weights(
        z_p_c, z_o_c, sigma_p, sigma_o
    )
    print(f"\nget_weights 输出形状:")
    print(f"  alpha_p: {alphas_p.shape}, alpha_o: {alphas_o.shape}")
    print(f"  logit_p: {logit_p.shape}, logit_o: {logit_o.shape}")

    print("\n所有测试通过!")


if __name__ == "__main__":
    test_adaptive_fusion()