import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_output = self.attention(Q, K, V, mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(attn_output)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 自注意力机制，只允许关注当前和之前的词（通过tgt_mask）
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 交叉注意力机制，关注整个编码器的输出
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        return self.norm3(x + self.dropout(ff_output))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff,
                 max_seq_length, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        # Encoder
        for enc_layer in self.encoder_layers:
            src_embedded = enc_layer(src_embedded, src_mask)

        # Decoder
        for dec_layer in self.decoder_layers:
            tgt_embedded = dec_layer(tgt_embedded, src_embedded, src_mask, tgt_mask)

        output = self.fc(tgt_embedded)
        return output


def create_mask(src, tgt, pad_idx, model):
    src_seq_len = src.size(1)
    tgt_seq_len = tgt.size(1)

    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_seq_len)
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)  # (batch_size, 1, tgt_seq_len, 1)
    tgt_subsequent_mask = model.generate_square_subsequent_mask(tgt_seq_len).unsqueeze(
        0)  # (1, tgt_seq_len, tgt_seq_len)

    # 将掩码转换为布尔类型，然后进行逻辑与运算
    tgt_mask = (tgt_mask.bool() & tgt_subsequent_mask.bool()).float()

    return src_mask, tgt_mask


def demo():
    # 定义模型参数
    src_vocab_size = 10  # 源语言词汇表大小
    tgt_vocab_size = 10  # 目标语言词汇表大小
    d_model = 512  # 模型维度
    num_heads = 8  # 注意力头数
    num_encoder_layers = 3  # 编码器层数
    num_decoder_layers = 3  # 解码器层数
    d_ff = 2048  # 前馈网络的维度
    max_seq_length = 100  # 最大序列长度
    dropout = 0.1  # dropout率

    # 创建模型实例
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers,
                        d_ff, max_seq_length, dropout)

    # 假设我们有以下输入数据（这里使用随机数字作为示例）
    # 源语言序列（假设每个句子长度为4）
    src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    # 目标语言序列（预测时，通常使用一个特殊的开始符号，如<bos>，这里用0表示）
    tgt = torch.tensor([[0, 2, 3, 4], [0, 6, 7, 8]])  # 第一个词是<bos>

    # 创建掩码
    # 注意：在实际应用中，你可能需要根据序列的实际长度来创建掩码
    src_mask, tgt_mask = create_mask(src, tgt, pad_idx=0, model=model)

    # 模型前向传播
    output = model(src, tgt, src_mask, tgt_mask)

    # 输出结果是每个词在目标语言词汇表上的预测得分
    print("输出预测得分的形状:", output.shape)
    # 输出类似于： torch.Size([2, 4, 10])，表示两个序列，每个序列4个词，每个词有10个类别（词汇表大小）

    # 如果你想看到预测的词：
    predicted_ids = output.argmax(dim=-1)
    print("预测的词ID:", predicted_ids)

    # 损失函数（这里使用交叉熵损失）
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 假设0是填充词或<bos>标记
    tgt_true = torch.tensor([[2, 3, 4, 0], [6, 7, 8, 0]])  # 真实目标序列，填充为0

    # 计算损失
    loss = criterion(output.view(-1, tgt_vocab_size), tgt_true.view(-1))
    print("损失:", loss.item())


if __name__ == '__main__':
    demo()
