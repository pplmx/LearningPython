import math
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import rcParams

# 设置matplotlib的中文显示
rcParams["font.sans-serif"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False


class PositionalEncoding(nn.Module):
    """
    位置编码模块

    在Transformer中，由于没有类似RNN的循环结构，需要额外的位置信息来表示序列中词的位置。
    位置编码使用正弦和余弦函数的组合，可以让模型学习到相对位置信息。

    Args:
        d_model (int): 模型的维度
        max_seq_length (int): 最大序列长度
        dropout (float): dropout比率
    """

    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [seq_len, batch_size, embedding_dim]
        Returns:
            添加位置编码后的张量 [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    手动实现的多头注意力机制

    相比直接使用nn.MultiheadAttention，这个实现更清晰地展示了内部计算过程

    Args:
        hidden_size (int): 隐藏层维度
        num_heads (int): 注意力头数量
        dropout (float): dropout比率
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Q、K、V的线性变换
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播过程

        Args:
            query: 查询张量 [seq_len, batch_size, hidden_size]
            key: 键张量 [seq_len, batch_size, hidden_size]
            value: 值张量 [seq_len, batch_size, hidden_size]
            mask: 掩码张量，用于mask掉某些位置的注意力分数

        Returns:
            output: 注意力的输出 [seq_len, batch_size, hidden_size]
            attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        """
        seq_len, batch_size, _ = query.shape

        # 线性变换并分头
        q = self.q_linear(query).view(seq_len, batch_size, self.num_heads, self.head_dim)
        k = self.k_linear(key).view(seq_len, batch_size, self.num_heads, self.head_dim)
        v = self.v_linear(value).view(seq_len, batch_size, self.num_heads, self.head_dim)

        # 调整维度顺序为 [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(0, 1).transpose(1, 2)
        k = k.transpose(0, 1).transpose(1, 2)
        v = v.transpose(0, 1).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用mask（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重并应用dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 计算输出
        out = torch.matmul(attention_weights, v)

        # 重新调整维度顺序并合并多头
        out = out.transpose(1, 2).transpose(0, 1).contiguous()
        out = out.view(seq_len, batch_size, self.hidden_size)

        # 最后的线性变换
        out = self.out_proj(out)

        return out, attention_weights


class EnhancedTransformerLayer(nn.Module):
    """
    增强版Transformer层实现

    特点：
    1. 支持自定义的多头注意力实现
    2. 支持位置编码
    3. 提供多种激活函数选择
    4. 支持注意力mask
    5. 增强的可视化功能

    Args:
        hidden_size (int): 隐藏层维度
        num_heads (int): 注意力头数量
        dropout (float): dropout比率
        activation (str): 激活函数类型，支持 'relu', 'gelu', 'swish'
        use_custom_attention (bool): 是否使用自定义的多头注意力实现
    """

    def __init__(self, hidden_size: int = 256, num_heads: int = 8, dropout: float = 0.1,
                 activation: str = 'relu', use_custom_attention: bool = False):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_size, dropout=dropout)

        # 多头注意力
        if use_custom_attention:
            self.self_attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        else:
            self.self_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=False
            )

        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # 前馈网络
        self.feed_forward = self._build_ffn(hidden_size, activation)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 可视化相关的属性
        self.vis_attention_weights = None
        self.vis_states = {}

    def _build_ffn(self, hidden_size: int, activation: str) -> nn.Sequential:
        """构建前馈网络"""
        activation_layer = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': lambda x: x * torch.sigmoid(x)  # Swish激活函数
        }.get(activation.lower(), nn.ReLU())

        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            activation_layer,
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [seq_len, batch_size, hidden_size]
            mask: 注意力mask

        Returns:
            输出张量 [seq_len, batch_size, hidden_size]
        """
        # 添加位置编码
        x = self.pos_encoding(x)
        self.vis_states['after_pos_encoding'] = x

        # 自注意力层
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        self.vis_attention_weights = attention_weights
        self.vis_states['after_attention'] = attn_output

        # 第一个残差连接和层归一化
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        self.vis_states['after_first_norm'] = x

        # 前馈网络
        ff_output = self.feed_forward(x)
        self.vis_states['after_ffn'] = ff_output

        # 第二个残差连接和层归一化
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        self.vis_states['final_output'] = x

        return x


def enhanced_visualization(model: EnhancedTransformerLayer,
                           input_data: torch.Tensor,
                           save_path: Optional[str] = None):
    """
    增强的可视化函数

    提供更丰富的可视化内容：
    1. 位置编码的模式
    2. 注意力权重的热力图
    3. 特征在各个阶段的分布变化
    4. 残差连接的效果

    Args:
        model: EnhancedTransformerLayer实例
        input_data: 输入数据
        save_path: 保存图像的路径（如果提供）
    """
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(20, 15))

    # 1. 位置编码可视化
    ax1 = plt.subplot(3, 2, 1)
    pos_encoding = model.pos_encoding.pe.squeeze(1)[:50, :20].detach()
    plt.imshow(pos_encoding, cmap='RdBu', aspect='auto')
    plt.colorbar()
    plt.title('位置编码模式 (前50个位置，前20个维度)')
    plt.xlabel('编码维度')
    plt.ylabel('序列位置')

    # 2. 注意力权重热力图
    ax2 = plt.subplot(3, 2, 2)
    if model.vis_attention_weights is not None:
        # 取第一个批次，所有头的平均
        avg_attention = model.vis_attention_weights[0].mean(0).detach()
        plt.imshow(avg_attention, cmap='viridis')
        plt.colorbar()
        plt.title('平均注意力权重分布')
        plt.xlabel('Key位置')
        plt.ylabel('Query位置')

    # 3. 特征分布变化
    stages = ['after_pos_encoding', 'after_attention', 'after_first_norm',
              'after_ffn', 'final_output']

    ax3 = plt.subplot(3, 2, (3, 4))
    for i, stage in enumerate(stages):
        if stage in model.vis_states:
            values = model.vis_states[stage].detach().flatten().numpy()
            plt.hist(values, bins=50, alpha=0.5, label=stage)
    plt.title('各阶段特征分布对比')
    plt.xlabel('特征值')
    plt.ylabel('频次')
    plt.legend()

    # 4. 残差连接效果
    ax4 = plt.subplot(3, 2, (5, 6))
    if 'after_attention' in model.vis_states and 'after_first_norm' in model.vis_states:
        residual = (model.vis_states['after_first_norm'] -
                    model.vis_states['after_attention']).detach().flatten().numpy()
        plt.hist(residual, bins=50, color='green', alpha=0.7)
        plt.title('残差连接的影响分布')
        plt.xlabel('残差值')
        plt.ylabel('频次')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def demo():
    """
    增强版的演示函数
    展示Transformer的各种功能和行为：
    1. 基本用法
    2. 不同配置的对比
    3. 位置编码的效果
    4. 注意力机制的行为
    5. 可视化展示
    """
    # 设置随机种子
    torch.manual_seed(42)

    print("=== Transformer增强演示 ===")

    # 1. 创建不同配置的模型
    configs = [
        {'hidden_size': 64, 'num_heads': 4, 'activation': 'relu'},
        {'hidden_size': 64, 'num_heads': 4, 'activation': 'gelu'},
        {'hidden_size': 64, 'num_heads': 4, 'activation': 'swish'}
    ]

    models = [EnhancedTransformerLayer(**config) for config in configs]

    # 2. 准备演示数据
    batch_size = 2
    seq_length = 10
    hidden_size = 64

    # 创建示例输入数据
    input_data = torch.randn(seq_length, batch_size, hidden_size)

    # 创建示例mask
    mask = torch.ones(batch_size, seq_length, seq_length)
    mask[:, :, seq_length // 2:] = 0  # mask掉后半部分，用于演示注意力mask的效果

    print("\n1. 基本用法演示")
    print("-" * 50)
    print(f"输入数据形状: {input_data.shape}")
    print(f"Mask形状: {mask.shape}")

    # 3. 对比不同配置的效果
    print("\n2. 不同激活函数的对比")
    print("-" * 50)
    outputs = []
    for i, (model, config) in enumerate(zip(models, configs)):
        output = model(input_data, mask)
        outputs.append(output)
        print(f"配置 {i + 1}: {config['activation']} - 输出均值: {output.mean().item():.4f}")

    # 4. 位置编码效果演示
    print("\n3. 位置编码效果展示")
    print("-" * 50)
    model = models[0]  # 使用第一个模型

    # 获取原始位置编码
    pos_encoding = model.pos_encoding.pe.squeeze(1)[:seq_length, :hidden_size]
    print(f"位置编码形状: {pos_encoding.shape}")
    print("位置编码示例(前5个位置，前5个维度):")
    print(pos_encoding[:5, :5].detach().numpy())

    # 5. 注意力机制行为分析
    print("\n4. 注意力机制分析")
    print("-" * 50)
    output = model(input_data, mask)
    if hasattr(model, 'vis_attention_weights') and model.vis_attention_weights is not None:
        attn_weights = model.vis_attention_weights[0]  # 取第一个batch
        print(f"注意力权重形状: {attn_weights.shape}")
        print("注意力模式示例(第一个头的前5x5):")
        print(attn_weights[0, :5, :5].detach().numpy())

    # 6. 可视化展示
    print("\n5. 生成可视化")
    print("-" * 50)

    # 创建三组不同的输入数据进行对比
    inputs = [
        torch.randn(seq_length, batch_size, hidden_size),  # 随机数据
        torch.ones(seq_length, batch_size, hidden_size),  # 全1数据
        torch.zeros(seq_length, batch_size, hidden_size)  # 全0数据
    ]

    for i, input_data in enumerate(inputs):
        print(f"处理输入数据集 {i + 1}...")
        model(input_data, mask)  # 运行模型以获取中间状态
        enhanced_visualization(model, input_data, f'transformer_vis_{i}.png')

    # 7. 模型行为分析
    print("\n6. 模型行为分析")
    print("-" * 50)

    # 测试不同长度的输入
    test_lengths = [5, 10, 20]
    for length in test_lengths:
        test_input = torch.randn(length, batch_size, hidden_size)
        output = model(test_input)
        print(f"序列长度 {length} - 输出形状: {output.shape}")

    # 8. 额外的教育性示例：展示梯度流动
    print("\n7. 梯度流动演示")
    print("-" * 50)

    # 启用梯度计算
    input_data.requires_grad_(True)
    output = model(input_data)
    loss = output.mean()
    loss.backward()

    print(f"输入梯度大小: {input_data.grad.abs().mean().item():.4f}")

    # 打印主要层的参数梯度
    print("\n各层梯度概况:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: {param.grad.abs().mean().item():.4f}")

    print("\n演示完成！")


if __name__ == "__main__":
    demo()
