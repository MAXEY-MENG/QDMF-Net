"""
跨模态对比学习模块（含不确定性Tempering）
实现 InfoNCE 损失，让病理和组学的公共特征在语义空间对齐

输入：
- z_p_c: 病理公共特征，形状 (batch, 64)
- z_o_c: 组学公共特征，形状 (batch, 64)

正样本对：同一患者的 (z_p_c[i], z_o_c[i])
负样本对：不同患者的 (z_p_c[i], z_o_c[j])，i != j

使用双向 InfoNCE 损失（类似 CLIP），温度 τ 可调。

不确定性Tempering:
- 当提供 sigma_p, sigma_o 时，高不确定性样本的对比损失会被抑制
- 机制: loss_sample = σ_mean² × loss_InfoNCE_per_sample
- 这样模型学会用σ来表示哪些样本的跨模态对齐不可靠

参考: CLIP (Radford et al., 2021) + Uncertainty-weighted loss (Kendall & Gal, 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CrossModalContrastiveLoss(nn.Module):
    """
    跨模态对比损失（InfoNCE）含不确定性Tempering

    以病理特征为锚点，组学特征为正/负样本，反之亦然。
    最终损失为两个方向的均值。

    L = 0.5 * (L_p2o + L_o2p)

    其中 L_p2o = -1/N * sum_i log( exp(sim(z_p[i], z_o[i])/τ) / sum_j exp(sim(z_p[i], z_o[j])/τ) )
    
    当启用不确定性tempering时:
    L = 0.5 * (L_p2o_unc + L_o2p_unc)
    其中 L_p2o_unc = 1/N * sum_i σ_mean_i² * l_i
    """

    def __init__(self, temperature: float = 0.07):
        """
        初始化对比损失

        Args:
            temperature: 温度参数 τ，控制相似度分布的平滑程度
                         默认 0.07（参考 CLIP 论文）
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_p_c: torch.Tensor,
        z_o_c: torch.Tensor,
        sigma_p: Optional[torch.Tensor] = None,
        sigma_o: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算跨模态对比损失（可选不确定性tempering）

        Args:
            z_p_c: 病理公共特征，形状 (batch_size, feature_dim)
            z_o_c: 组学公共特征，形状 (batch_size, feature_dim)
            sigma_p: 病理不确定性 (batch, 1)，可选，用于tempering
            sigma_o: 组学不确定性 (batch, 1)，可选，用于tempering

        Returns:
            loss: 对比损失值（标量）
            pos_sim: 平均正样本相似度（标量，用于日志记录）
        """
        batch_size = z_p_c.shape[0]

        # L2 归一化
        z_p_norm = F.normalize(z_p_c, p=2, dim=1)  # (batch, dim)
        z_o_norm = F.normalize(z_o_c, p=2, dim=1)  # (batch, dim)

        # 计算相似度矩阵: sim[i][j] = cosine_sim(z_p[i], z_o[j])
        sim_matrix = z_p_norm @ z_o_norm.T  # (batch, batch)

        # 除以温度
        sim_matrix = sim_matrix / self.temperature  # (batch, batch)

        # 正样本对: 对角线元素
        pos_sim = sim_matrix.diag()  # (batch,)

        # 计算平均正样本相似度（用于日志）
        avg_pos_sim = pos_sim.mean().detach()

        # 标签: 对于每行，正样本是对角线
        labels = torch.arange(batch_size, device=z_p_c.device)

        if sigma_p is not None and sigma_o is not None:
            # === 不确定性Tempering ===
            # 高不确定性样本 → 对比损失被抑制（tempering）
            # weight = 1 / (1 + σ²): σ小(confident)→weight≈1, σ大(uncertain)→weight→0
            sigma_mean = (sigma_p + sigma_o) / 2.0  # (batch, 1)

            # 如果sigma是向量，先降为标量（防御性检查）
            if sigma_mean.dim() > 1 and sigma_mean.shape[1] > 1:
                sigma_scalar = sigma_mean.mean(dim=1, keepdim=True)  # (batch, 1)
            else:
                sigma_scalar = sigma_mean

            # 温度权重: 2 / (1 + σ²)
            # σ=1 (KL先验) → weight=1.0 (全强度)
            # σ>1 (不确定) → weight→0 (抑制)
            # σ<1 (高置信) → weight→2 (适度增强)
            temper_weight = 2.0 / (1.0 + sigma_scalar ** 2)  # (batch, 1)
            temper_weight = temper_weight.squeeze(-1)  # (batch,)

            # --- 方向1: 以病理为锚点 ---
            log_probs_p2o = F.log_softmax(sim_matrix, dim=1)  # (batch, batch)
            nll_p2o = -log_probs_p2o[torch.arange(batch_size, device=z_p_c.device), labels]  # (batch,)
            loss_p2o = (temper_weight * nll_p2o).mean()

            # --- 方向2: 以组学为锚点 ---
            log_probs_o2p = F.log_softmax(sim_matrix.T, dim=1)  # (batch, batch)
            nll_o2p = -log_probs_o2p[torch.arange(batch_size, device=z_p_c.device), labels]  # (batch,)
            loss_o2p = (temper_weight * nll_o2p).mean()
        else:
            # 标准InfoNCE（无tempering）
            loss_p2o = F.cross_entropy(sim_matrix, labels)
            loss_o2p = F.cross_entropy(sim_matrix.T, labels)

        # 最终损失: 两个方向的均值
        loss = (loss_p2o + loss_o2p) / 2.0

        return loss, avg_pos_sim


def test_contrastive_loss():
    """测试跨模态对比损失"""
    torch.manual_seed(42)

    batch_size = 8
    feature_dim = 64  # 新的特征维度

    # 创建模拟数据
    z_p_c = torch.randn(batch_size, feature_dim)
    z_o_c = torch.randn(batch_size, feature_dim)

    print(f"z_p_c 形状: {z_p_c.shape}")
    print(f"z_o_c 形状: {z_o_c.shape}")

    # 测试默认温度
    print("\n测试默认温度 (τ=0.07)...")
    criterion = CrossModalContrastiveLoss(temperature=0.07)
    loss, pos_sim = criterion(z_p_c, z_o_c)
    print(f"对比损失: {loss.item():.4f}")
    print(f"平均正样本相似度: {pos_sim.item():.4f}")

    # 验证损失值范围
    assert loss.item() > 0, f"损失应为正数，得到 {loss.item()}"
    assert -1.0 <= pos_sim.item() <= 1.0, \
        f"正样本相似度应在 [-1, 1] 范围内，得到 {pos_sim.item()}"

    # 测试不同温度
    print("\n测试不同温度...")
    for tau in [0.01, 0.05, 0.1, 0.5, 1.0]:
        criterion_t = CrossModalContrastiveLoss(temperature=tau)
        loss_t, pos_sim_t = criterion_t(z_p_c, z_o_c)
        print(f"  τ={tau:.2f}: loss={loss_t.item():.4f}, pos_sim={pos_sim_t.item():.4f}")

    # 测试不确定性Tempering
    print("\n测试不确定性Tempering...")
    sigma_p = torch.rand(batch_size, 1) * 0.5 + 0.25  # 0.25-0.75
    sigma_o = torch.rand(batch_size, 1) * 0.5 + 0.25
    loss_unc, pos_sim_unc = criterion(z_p_c, z_o_c, sigma_p, sigma_o)
    print(f"  有Tempering: loss={loss_unc.item():.4f}")
    # 有tempering vs 无tempering
    loss_base, _ = criterion(z_p_c, z_o_c)
    print(f"  无Tempering: loss={loss_base.item():.4f}")

    # 测试完全对齐的情况（正样本相似度接近1）
    print("\n测试完全对齐的特征...")
    z_aligned = torch.randn(batch_size, feature_dim)
    loss_align, pos_sim_align = criterion(z_aligned, z_aligned)
    print(f"  对齐特征: loss={loss_align.item():.4f}, pos_sim={pos_sim_align.item():.4f}")
    # 完全对齐时，正样本相似度应接近1，损失应较小
    assert pos_sim_align.item() > 0.5, \
        f"对齐特征的正样本相似度应较高，得到 {pos_sim_align.item()}"

    # 验证对齐特征的损失小于随机特征的损失
    print("\n测试随机特征...")
    z_random1 = torch.randn(batch_size, feature_dim)
    z_random2 = torch.randn(batch_size, feature_dim)
    loss_rand, pos_sim_rand = criterion(z_random1, z_random2)
    print(f"  随机特征: loss={loss_rand.item():.4f}, pos_sim={pos_sim_rand.item():.4f}")
    assert loss_align.item() < loss_rand.item(), \
        "对齐特征的损失应小于随机特征的损失"

    # 测试梯度可传播
    print("\n测试梯度传播...")
    z_p = torch.randn(batch_size, feature_dim, requires_grad=True)
    z_o = torch.randn(batch_size, feature_dim, requires_grad=True)
    loss_grad, _ = criterion(z_p, z_o)
    loss_grad.backward()
    assert z_p.grad is not None, "z_p_c 的梯度不应为 None"
    assert z_o.grad is not None, "z_o_c 的梯度不应为 None"
    print(f"  z_p_c 梯度形状: {z_p.grad.shape}")
    print(f"  z_o_c 梯度形状: {z_o.grad.shape}")
    print("  梯度传播正常!")

    # 测试Tempering梯度
    print("\n测试Tempering梯度传播...")
    z_p_t = torch.randn(batch_size, feature_dim, requires_grad=True)
    z_o_t = torch.randn(batch_size, feature_dim, requires_grad=True)
    sigma_p_t = torch.rand(batch_size, 1, requires_grad=True)
    sigma_o_t = torch.rand(batch_size, 1, requires_grad=True)
    loss_t, _ = criterion(z_p_t, z_o_t, sigma_p_t, sigma_o_t)
    loss_t.backward()
    assert sigma_p_t.grad is not None, "sigma_p梯度不应为None"
    print(f"  sigma_p gradient: {sigma_p_t.grad.abs().mean().item():.6f}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Test CrossModalContrastiveLoss (V2)")
    print("=" * 60)
    test_contrastive_loss()