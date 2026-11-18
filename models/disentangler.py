"""
解耦器模块
用于将特征分解为公共特征（common）和私有特征（private）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def orthogonal_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    正交损失函数：计算 |cosine_sim(z1, z2)| 的均值

    鼓励两个特征向量在方向上正交，从而解耦公共和私有信息。

    Args:
        z1: 第一个特征张量，形状 (batch_size, dim)
        z2: 第二个特征张量，形状 (batch_size, dim)

    Returns:
        正交损失值（标量）
    """
    z1_norm = F.normalize(z1, p=2, dim=1)
    z2_norm = F.normalize(z2, p=2, dim=1)
    cos_sim = (z1_norm * z2_norm).sum(dim=1)  # (batch_size,)
    loss = torch.abs(cos_sim).mean()
    return loss


class Disentangler(nn.Module):
    """
    解耦器模块

    将输入特征分解为两个正交的子空间：
    - z_common: 公共特征（与其他模态共享的信息）
    - z_private: 私有特征（当前模态特有的信息）

    结构：
    1. 共享层：Linear(D, H) + LayerNorm + ReLU + Dropout
    2. 公共投影头：Linear(H, O)  (无Dropout，输出需稳定)
    3. 私有投影头：Linear(H, O)  (无Dropout)

    Args:
        input_dim: 输入特征维度（默认512）
        hidden_dim: 隐藏层维度（默认128）
        output_dim: 输出特征维度（默认128）
        dropout: Dropout概率（默认0.3）
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # 共享层（LayerNorm稳定训练）
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 公共特征投影头（无Dropout，输出特征需稳定用于对比学习）
        self.proj_common = nn.Linear(hidden_dim, output_dim)

        # 私有特征投影头（无Dropout）
        self.proj_private = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入特征，形状 (batch_size, input_dim)

        Returns:
            z_common: 公共特征，形状 (batch_size, output_dim)
            z_private: 私有特征，形状 (batch_size, output_dim)
        """
        h = self.shared_layer(x)        # (batch, hidden_dim)
        z_common = self.proj_common(h)  # (batch, output_dim)
        z_private = self.proj_private(h)  # (batch, output_dim)
        return z_common, z_private


def test_disentangler():
    """测试解耦器模块"""
    torch.manual_seed(42)

    batch_size = 8
    input_dim = 512
    output_dim = 256

    x = torch.randn(batch_size, input_dim)
    print(f"输入形状: {x.shape}")

    print("\n测试解耦器...")
    disentangler = Disentangler(input_dim=input_dim, output_dim=output_dim)
    z_common, z_private = disentangler(x)

    print(f"公共特征形状: {z_common.shape}")
    print(f"私有特征形状: {z_private.shape}")

    assert z_common.shape == (batch_size, output_dim)
    assert z_private.shape == (batch_size, output_dim)

    print("\n测试正交损失...")
    loss = orthogonal_loss(z_common, z_private)
    print(f"正交损失: {loss.item():.6f}")
    assert 0 <= loss.item() <= 1.0

    print("\n所有测试通过!")


if __name__ == "__main__":
    test_disentangler()
