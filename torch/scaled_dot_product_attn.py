import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    实现缩放点积注意力机制

    参数:
        query: 查询张量, shape = (..., seq_len_q, d_k)
        key: 键张量, shape = (..., seq_len_k, d_k)
        value: 值张量, shape = (..., seq_len_v, d_v)
        mask: 可选的掩码张量

    返回:
        注意力输出和注意力权重
    """
    # 获取维度
    d_k = query.size(-1)

    # 计算注意力分数
    # matmul 后的形状: (..., seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1))

    # 缩放
    scaled_scores = scores / math.sqrt(d_k)

    # 应用掩码（如果提供）
    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(mask == 0, -1e9)

    # 应用 softmax
    attention_weights = F.softmax(scaled_scores, dim=-1)

    # 计算输出
    # shape = (..., seq_len_q, d_v)
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


def attention_demo():
    """
    演示缩放点积注意力的计算过程
    """
    # 设置随机种子
    torch.manual_seed(42)

    # 设置参数
    batch_size = 2
    num_heads = 1
    seq_length = 4
    d_k = 8  # 注意力空间的维度

    # 创建假设的 query, key, value 张量
    query = torch.randn(batch_size, num_heads, seq_length, d_k)
    key = torch.randn(batch_size, num_heads, seq_length, d_k)
    value = torch.randn(batch_size, num_heads, seq_length, d_k)

    print("输入形状:")
    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")

    # 创建一个简单的掩码（可选）
    # 这里我们创建一个下三角掩码，用于序列生成任务
    mask = torch.tril(torch.ones(seq_length, seq_length)).unsqueeze(0).unsqueeze(0)
    print(f"\nMask shape: {mask.shape}")
    print("Mask:")
    print(mask[0, 0])

    # 计算注意力
    output, attention_weights = scaled_dot_product_attention(query, key, value, mask)

    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # 打印注意力权重
    print("\nAttention weights for first head, first batch:")
    print(attention_weights[0, 0])

    # 打印输出的一部分
    print("\nOutput for first head, first batch:")
    print(output[0, 0])


if __name__ == "__main__":
    attention_demo()
