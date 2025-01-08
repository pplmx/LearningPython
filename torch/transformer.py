import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

import torch
import torch.nn as nn

# 设置字体为 SimHei (黑体) 以支持中文显示
rcParams["font.sans-serif"] = ["SimHei"]  # 用黑体显示中文
rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def plot_attention_weights(attention_weights, title="Attention Weights Heatmap"):
    """
    可视化注意力权重

    Args:
        attention_weights (torch.Tensor): 注意力权重矩阵 [seq_len, seq_len]
        title (str): 图表标题
    """
    # 确保输入是numpy数组并且是2D的
    if torch.is_tensor(attention_weights):
        weights = attention_weights.detach().cpu().numpy()
    else:
        weights = np.array(attention_weights)

    # 确保是2D数组
    if weights.ndim != 2:
        raise ValueError(f"注意力权重必须是2D数组，但得到的shape是 {weights.shape}")

    plt.figure(figsize=(10, 8))
    plt.imshow(weights, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Key位置")
    plt.ylabel("Query位置")
    plt.tight_layout()
    plt.show()


def compare_feature_distributions(before, after, title="特征分布对比"):
    """
    对比层归一化前后的特征分布

    Args:
        before (torch.Tensor): 归一化前的特征
        after (torch.Tensor): 归一化后的特征
        title (str): 图表标题
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 转换为numpy数组
    before_np = before.flatten().detach().cpu().numpy()
    after_np = after.flatten().detach().cpu().numpy()

    # 绘制归一化前的分布
    ax1.hist(before_np, bins=50)
    ax1.set_title("归一化前")
    ax1.set_xlabel("特征值")
    ax1.set_ylabel("频次")
    ax1.grid(True, alpha=0.3)

    # 绘制归一化后的分布
    ax2.hist(after_np, bins=50)
    ax2.set_title("归一化后")
    ax2.set_xlabel("特征值")
    ax2.set_ylabel("频次")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


class SimpleTransformerLayer(nn.Module):
    """
    简单Transformer层实现

    核心组件说明:
    1. 多头自注意力机制 (Multi-head Self-attention)
       - 允许模型关注序列中的不同位置
       - 多个注意力头提供不同的关注视角

    2. LayerNorm归一化层
       - 在序列的每个位置上独立进行特征归一化
       - 帮助训练稳定性和梯度流动

    3. 前馈神经网络 (Feed-forward Network)
       - 对每个位置独立进行非线性变换
       - 增加模型的表达能力

    4. 残差连接 (Residual Connections)
       - 帮助深层网络的训练
       - 缓解梯度消失问题

    Args:
        hidden_size (int): 隐藏层维度，决定了特征表示的丰富程度
        num_heads (int): 注意力头数量，每个头关注不同的特征模式
        dropout (float): Dropout比率，用于防止过拟合
    """

    def __init__(self, hidden_size=8, num_heads=2, dropout=0.1):
        super().__init__()

        # 确保hidden_size能被num_heads整除
        assert (
            hidden_size % num_heads == 0
        ), f"hidden_size({hidden_size})必须能被num_heads({num_heads})整除"

        # 多头注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,  # 特征维度
            num_heads=num_heads,  # 注意力头数量
            dropout=dropout,  # 注意力dropout
            batch_first=False,  # 输入格式为[seq_len, batch, hidden_size]
        )

        # 第一个LayerNorm
        # 对注意力输出进行归一化，帮助训练稳定性
        self.norm1 = nn.LayerNorm(hidden_size)

        # 前馈神经网络
        # 由两个线性变换组成，中间有ReLU激活函数
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),  # 扩展特征维度
            nn.ReLU(),  # 非线性激活
            nn.Linear(hidden_size * 4, hidden_size),  # 恢复原始维度
        )

        # 第二个LayerNorm
        # 对前馈网络输出进行归一化
        self.norm2 = nn.LayerNorm(hidden_size)

        # Dropout层
        # 用于防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 保存注意力权重用于可视化
        self.last_attention_weights = None

    def forward(self, x):
        """
        前向传播过程

        步骤说明:
        1. 自注意力计算
        2. 残差连接和第一次归一化
        3. 前馈网络处理
        4. 残差连接和第二次归一化

        Args:
            x (torch.Tensor): 输入张量，形状为[seq_len, batch_size, hidden_size]

        Returns:
            torch.Tensor: 输出张量，形状与输入相同
        """
        # 1. 多头自注意力层
        # attn_output: 注意力层的输出
        # attention_weights: 注意力权重矩阵
        attn_output, attention_weights = self.self_attention(x, x, x)
        self.last_attention_weights = attention_weights  # 保存权重用于可视化

        # 2. 第一个残差连接和层归一化
        # 注意力输出 -> Dropout -> 残差连接 -> LayerNorm
        attention_output = x + self.dropout(attn_output)  # 残差连接
        normed_attention = self.norm1(attention_output)  # 层归一化

        # 保存中间状态用于可视化
        self.attention_before_norm = attention_output
        self.attention_after_norm = normed_attention

        # 3. 前馈网络
        ff_output = self.feed_forward(normed_attention)

        # 4. 第二个残差连接和层归一化
        # 前馈输出 -> Dropout -> 残差连接 -> LayerNorm
        ff_output = normed_attention + self.dropout(ff_output)  # 残差连接
        output = self.norm2(ff_output)  # 层归一化

        # 保存中间状态用于可视化
        self.ff_before_norm = ff_output
        self.ff_after_norm = output

        return output


def transformer_demo():
    """
    演示Transformer层的工作原理

    包含以下演示内容:
    1. 基本用法展示
    2. 注意力权重可视化
    3. 特征分布对比
    4. 参数和梯度信息
    """
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)

    print("=== 1. 基本用法展示 ===")
    # 定义基本参数
    sequence_length = 10  # 序列长度
    batch_size = 2  # 批次大小
    hidden_size = 8  # 隐藏层维度

    # 创建模型实例
    model = SimpleTransformerLayer(hidden_size=hidden_size)
    print("\n模型结构:")
    print(model)

    # 创建随机输入数据
    input_tensor = torch.randn(sequence_length, batch_size, hidden_size)
    print(f"\n输入形状: {input_tensor.shape}")
    print("输入数据统计:")
    print(f"- 均值: {input_tensor.mean():.4f}")
    print(f"- 标准差: {input_tensor.std():.4f}")

    # 前向传播
    output = model(input_tensor)
    print(f"\n输出形状: {output.shape}")
    print("输出数据统计:")
    print(f"- 均值: {output.mean():.4f}")
    print(f"- 标准差: {output.std():.4f}")

    print("\n=== 2. 注意力权重可视化 ===")
    # 获取注意力权重并处理成二维矩阵
    # 注意力权重形状为 [batch_size, num_heads, seq_len, seq_len]
    attention_weights = model.last_attention_weights
    print("\n原始注意力权重形状:", attention_weights.shape)

    # 取第一个批次，并对所有头的权重取平均
    if attention_weights.ndim == 4:  # 确保是 [batch, num_heads, seq_len, seq_len]
        avg_attention = attention_weights[0].mean(dim=0)  # [seq_len, seq_len]
    elif attention_weights.ndim == 3:  # 如果没有多头, 直接是 [batch, seq_len, seq_len]
        avg_attention = attention_weights[0]  # [seq_len, seq_len]
    else:
        raise ValueError(f"注意力权重的形状不正确: {attention_weights.shape}")

    print("处理后的注意力权重形状:", avg_attention.shape)

    # 可视化
    print("\n绘制注意力权重热力图...")
    plot_attention_weights(avg_attention, "平均注意力权重分布")

    print("\n=== 3. 特征分布可视化 ===")
    print("\n可视化自注意力层的归一化效果...")
    compare_feature_distributions(
        model.attention_before_norm,
        model.attention_after_norm,
        "自注意力层 - LayerNorm前后对比",
    )

    print("\n可视化前馈网络层的归一化效果...")
    compare_feature_distributions(
        model.ff_before_norm, model.ff_after_norm, "前馈网络层 - LayerNorm前后对比"
    )

    print("\n=== 4. 参数和梯度信息 ===")
    # 计算一个样例损失并反向传播
    target = torch.randn_like(output)
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)
    loss.backward()

    # 打印模型参数和梯度信息
    print("\n模型参数和梯度统计:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"\n{name}:")
            print(f"- 参数形状: {param.shape}")
            print(f"- 参数均值: {param.mean():.4f}")
            print(f"- 梯度形状: {param.grad.shape}")
            print(f"- 梯度均值: {param.grad.mean():.4f}")


if __name__ == "__main__":
    transformer_demo()
