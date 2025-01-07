import torch
import torch.nn as nn


def multi_head_attention_demo():
    """
    演示 PyTorch 中 MultiHeadAttention 的基本使用
    多头注意力的计算步骤：
    1. 将输入分成多个头
    2. 对每个头计算注意力分数
    3. 进行注意力加权
    4. 合并多个头的结果
    """
    # 设置随机种子
    torch.manual_seed(42)

    # 定义基本参数
    sequence_length = 10
    batch_size = 2
    hidden_size = 8
    num_heads = 2  # 注意力头数

    # 创建输入张量: shape = (sequence_length, batch_size, hidden_size)
    input_tensor = torch.randn(sequence_length, batch_size, hidden_size)
    print(f"Input shape: {input_tensor.shape}")

    # 创建多头注意力层
    # embed_dim: 输入的特征维度
    # num_heads: 注意力头的数量
    # 注意：embed_dim 必须能被 num_heads 整除
    multi_head_attn = nn.MultiheadAttention(
        embed_dim=hidden_size,
        num_heads=num_heads,
        batch_first=False,  # 如果为True，则输入形状应为 (batch_size, sequence_length, hidden_size)
    )

    # 前向传播
    # 这里使用自注意力机制，所以query、key、value都是同一个输入
    # output shape: (sequence_length, batch_size, hidden_size)
    # attention_weights shape: (batch_size, sequence_length, sequence_length)
    output, attention_weights = multi_head_attn(
        query=input_tensor, key=input_tensor, value=input_tensor
    )

    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # 创建模拟的梯度
    grad = torch.randn_like(output)

    # 反向传播
    output.backward(grad)

    # 打印一些参数的梯度形状
    for name, param in multi_head_attn.named_parameters():
        if param.grad is not None:
            print(f"\nGradient shape for {name}: {param.grad.shape}")

    # 显示注意力权重的示例（第一个batch的前3x3部分）
    print("\nAttention weights sample (first batch, 3x3 section):")
    print(attention_weights[0, :3, :3])  # batch_idx=0，显示3x3的部分


if __name__ == "__main__":
    multi_head_attention_demo()
