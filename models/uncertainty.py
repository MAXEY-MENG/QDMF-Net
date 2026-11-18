"""
不确定性估计网络（增强版）
用于估计特征的不确定性/置信度，sigma值小表示高确定性

增强说明：
- 输入改为原始编码特征（512维），而非解耦后的抽象特征
- 3层MLP (512->256->128->1) 提供更强的建模能力
- BatchNorm稳定训练
- 不使用L2正则化压制sigma，而是通过不确定性tempering对比损失自然学习
"""

import torch
import torch.nn as nn


class UncertaintyEstimator(nn.Module):
    """
    不确定性估计网络（增强版）

    输入：原始编码特征（如 path_emb, omics_emb），形状 (batch, 512)
    输出：不确定性 sigma ∈ (0,1)，形状 (batch, 1)

    结构：
    - BatchNorm(512) → Linear(512, 256) + ReLU + Dropout(0.2)
    - Linear(256, 128) + ReLU
    - Linear(128, 1) + Sigmoid

    sigma 值小表示高确定性（低不确定性）。
    例如 sigma=0.1 表示模型对该特征很有信心，sigma=0.9 表示很不确定。
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim_1: int = 256,
        hidden_dim_2: int = 128,
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        """
        初始化不确定性估计网络

        Args:
            input_dim: 输入特征维度（默认512）
            hidden_dim_1: 第一隐藏层维度（默认256）
            hidden_dim_2: 第二隐藏层维度（默认128）
            output_dim: 输出维度（默认1）
            dropout: Dropout概率（默认0.2）
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.output_dim = output_dim

        # 三层 MLP + BatchNorm + Dropout + Sigmoid
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, output_dim),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用更小的初始化，防止sigma饱和
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    # 偏置初始化为使sigmoid输出接近0.5
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, z_c: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            z_c: 原始编码特征，形状 (batch_size, input_dim)
                 注意：输入应是原始编码特征而非解耦后的特征

        Returns:
            sigma: 不确定性估计，形状 (batch_size, 1)，值域 (0,1)
                   sigma 值小 → 高确定性，sigma 值大 → 低不确定性
        """
        sigma = self.net(z_c)  # (batch, 1)
        # 添加一个小epsilon防止log(0)
        sigma = torch.clamp(sigma, min=1e-6, max=1.0 - 1e-6)
        return sigma

    def get_config(self) -> dict:
        """获取配置"""
        return {
            'input_dim': self.input_dim,
            'hidden_dim_1': self.hidden_dim_1,
            'hidden_dim_2': self.hidden_dim_2,
            'output_dim': self.output_dim,
        }


def test_uncertainty_estimator():
    """测试不确定性估计网络"""
    # 设置随机种子
    torch.manual_seed(42)

    batch_size = 8
    input_dim = 512  # 原始编码特征维度

    # 创建模拟数据
    z_c = torch.randn(batch_size, input_dim)
    print(f"输入形状: {z_c.shape}")

    # 测试不确定性估计器
    print("\n测试不确定性估计器（增强版）...")
    estimator = UncertaintyEstimator(input_dim=input_dim)
    sigma = estimator(z_c)

    print(f"sigma 形状: {sigma.shape}")
    print(f"sigma 值范围: [{sigma.min().item():.6f}, {sigma.max().item():.6f}]")
    print(f"sigma 均值: {sigma.mean().item():.6f}")
    print(f"sigma 标准差: {sigma.std().item():.6f}")

    # 验证输出维度
    assert sigma.shape == (batch_size, 1), f"sigma 形状错误: {sigma.shape}"

    # 验证值域在 (0,1)
    assert torch.all(sigma > 0) and torch.all(sigma < 1), \
        f"sigma 值域错误: [{sigma.min().item()}, {sigma.max().item()}]"

    # 测试不同配置
    print("\n测试不同配置...")
    estimator2 = UncertaintyEstimator(input_dim=128, hidden_dim_1=64, hidden_dim_2=32)
    z_c2 = torch.randn(batch_size, 128)
    sigma2 = estimator2(z_c2)
    print(f"自定义配置 - sigma 形状: {sigma2.shape}")
    assert sigma2.shape == (batch_size, 1)

    # 测试确定性输入（全零特征 → 高不确定性？取决于学习）
    print("\n测试确定性输入...")
    z_c_zero = torch.zeros(batch_size, input_dim)
    sigma_zero = estimator(z_c_zero)
    print(f"零输入 sigma 均值: {sigma_zero.mean().item():.6f}")

    # 测试极端输入
    z_c_large = torch.randn(batch_size, input_dim) * 100
    sigma_large = estimator(z_c_large)
    print(f"大值输入 sigma 范围: [{sigma_large.min().item():.6f}, {sigma_large.max().item():.6f}]")

    # 测试相同输入 vs 不同输入（应该给出不同的不确定性）
    z_same = torch.ones(batch_size, input_dim) * 42.0  # 完全相同
    sigma_same = estimator(z_same)
    print(f"\n相同输入 sigma 标准差: {sigma_same.std().item():.6f} (应接近0)")

    # 验证梯度可传播
    print("\n测试梯度传播...")
    z_grad = torch.randn(batch_size, input_dim, requires_grad=True)
    sigma_grad = estimator(z_grad)
    loss = sigma_grad.mean()
    loss.backward()
    assert z_grad.grad is not None, "梯度不应为 None"
    print(f"  输入梯度形状: {z_grad.grad.shape}")
    print(f"  输入梯度均值: {z_grad.grad.mean().item():.6f}")
    print("  梯度传播正常!")

    # 测试输出差异性 - 不同输入应产生不同的sigma
    z_random1 = torch.randn(batch_size, input_dim)
    z_random2 = torch.randn(batch_size, input_dim) + 10.0  # 明显不同的分布
    sigma1 = estimator(z_random1)
    sigma2 = estimator(z_random2)
    diff = (sigma1 - sigma2).abs().mean().item()
    print(f"\n不同输入sigma差异: {diff:.6f}")
    # 期望: 不同分布输入产生不同sigma

    print("\n所有测试通过!")


if __name__ == "__main__":
    test_uncertainty_estimator()