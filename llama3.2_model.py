import torch
import torch.nn as nn


# 定义1B参数量的LLaMA模型配置
LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,           # 词汇表大小，模型能理解的词汇数量
    "context_length": 8192,          # 模型在推理时使用的最大上下文长度（为了节省内存进行了缩减）
    "orig_context_length": 131_072,  # 模型训练时使用的原始上下文长度
    "emb_dim": 2048,                 # 词嵌入的维度，词向量的大小
    "n_heads": 32,                   # 多头注意力机制中头的数量
    "n_layers": 16,                  # Transformer层的层数
    "hidden_dim": 8192,              # 前馈神经网络（FeedForward）中中间层的维度
    "n_kv_groups": 8,                # 分组查询注意力机制中的键值组数量，用于优化计算
    "rope_base": 500_000.0,          # RoPE（旋转位置编码）中的基础参数θ
    "dtype": torch.bfloat16,         # 模型参数和计算使用的低精度数据类型，以减少内存占用
    "rope_freq": {                   # RoPE频率缩放的相关参数
        "factor": 32.0,              # 频率缩放的因子
        "low_freq_factor": 1.0,      # 低频部分的缩放因子
        "high_freq_factor": 4.0,     # 高频部分的缩放因子
        "original_context_length": 8192,  # RoPE计算时使用的原始上下文长度
    }
}


# 定义3B参数量的LLaMA模型配置
LLAMA32_CONFIG_3B = {
    "vocab_size": 128_256,           # 词汇表大小
    "context_length": 8192,          # 最大上下文长度（缩减以节省内存）
    "orig_context_length": 131_072,  # 原始上下文长度
    "emb_dim": 3072,                 # 词嵌入的维度
    "n_heads": 24,                   # 多头注意力机制中头的数量
    "n_layers": 28,                  # Transformer层的层数
    "hidden_dim": 8192,              # 前馈神经网络中间层的维度
    "n_kv_groups": 8,                # 分组查询注意力机制中的键值组数量
    "rope_base": 500_000.0,          # RoPE的基础参数θ
    "dtype": torch.bfloat16,         # 低精度数据类型
    "rope_freq": {                   # RoPE频率缩放的相关参数
        "factor": 32.0,              # 频率缩放的因子
        "low_freq_factor": 1.0,      # 低频部分的缩放因子
        "high_freq_factor": 4.0,     # 高频部分的缩放因子
        "original_context_length": 8192,   # RoPE计算时使用的原始上下文长度
    }
}


# 定义LLaMA模型的主类
class Llama3Model(nn.Module):
    def __init__(self, cfg):
        """
        初始化LLaMA模型。

        参数:
            cfg (dict): 模型配置，包含以下键:
                - vocab_size (int): 词汇表大小
                - context_length (int): 最大上下文长度
                - orig_context_length (int): 原始上下文长度
                - emb_dim (int): 词嵌入的维度
                - n_heads (int): 多头注意力机制中头的数量
                - n_layers (int): Transformer层的层数
                - hidden_dim (int): 前馈神经网络中间层的维度
                - n_kv_groups (int): 分组查询注意力机制中的键值组数量
                - rope_base (float): RoPE的基础参数θ
                - dtype (torch.dtype): 模型参数和计算使用的低精度数据类型
                - rope_freq (dict): RoPE频率缩放的相关参数
        """
        super().__init__()

        # 主模型参数
        # 词嵌入层，将词汇索引转换为词向量
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        # Transformer层列表，包含多个TransformerBlock模块
        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # 最终的RMS归一化层，用于对Transformer输出进行归一化
        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])

        # 输出线性层，将Transformer输出映射回词汇表维度，用于预测下一个词
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # 可重用的工具
        # 创建注意力掩码，上三角矩阵用于掩盖未来的信息，防止模型看到未来的词
        self.register_buffer(
            "mask", torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1).bool(),
            persistent=False
        )

        # 如果原始上下文长度与当前上下文长度不同，则重新缩放RoPE的基础参数θ
        if cfg["orig_context_length"] != cfg["context_length"]:
            cfg["rope_base"] = rescale_theta(
                            cfg["rope_base"],
                            cfg["orig_context_length"],
                            cfg["context_length"]
                        )
            
        # 计算RoPE的余弦和正弦值，用于位置编码
        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"]
        )

        # 将计算得到的余弦和正弦值注册为缓冲区，不作为模型参数保存
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # 将配置保存为模型属性
        self.cfg = cfg

    def forward(self, in_idx):
        """
        前向传播函数。

        参数:
            in_idx (torch.Tensor): 输入的词汇索引张量，形状为(batch_size, sequence_length)

        返回:
            torch.Tensor: 模型输出的logits，形状为(batch_size, sequence_length, vocab_size)
        """
        # 前向传播
        # 将输入词汇索引转换为词向量，形状为(batch_size, sequence_length, emb_dim)
        tok_embeds = self.tok_emb(in_idx)

        # 初始化Transformer的输入
        x = tok_embeds

        for block in self.trf_blocks:
            # 通过每个TransformerBlock进行处理，传入当前输入、掩码、RoPE的余弦和正弦值
            x = block(x, self.mask, self.cos, self.sin)

        # 对Transformer的输出进行RMS归一化
        x = self.final_norm(x)

        # 将归一化后的输出映射回词汇表维度，得到logits，形状为(batch_size, sequence_length, vocab_size)
        logits = self.out_head(x.to(self.cfg["dtype"]))

        # 返回logits
        return logits


# 定义TransformerBlock类
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        """
        初始化TransformerBlock。

        参数:
            cfg (dict): 模型配置，包含以下键:
                - emb_dim (int): 词嵌入的维度，词向量的大小
                - n_heads (int): 多头注意力机制中头的数量
                - n_kv_groups (int): 分组查询注意力机制中的键值组数量
                - dtype (torch.dtype): 模型参数和计算使用的低精度数据类型
        """
        super().__init__()

        # 自注意力模块，使用分组查询注意力机制
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],               # 输入维度，等于词嵌入的维度
            d_out=cfg["emb_dim"],              # 输出维度，等于词嵌入的维度
            num_heads=cfg["n_heads"],          # 多头注意力机制中头的数量
            num_kv_groups=cfg["n_kv_groups"],  # 分组查询注意力机制中的键值组数量
            dtype=cfg["dtype"]                 # 模型参数和计算使用的低精度数据类型
        )

        # 前馈神经网络模块
        self.ff = FeedForward(cfg)

        # 层归一化，用于对自注意力和前馈神经网络的输入进行归一化
        # RMS归一化，eps为防止除零的小常数
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])

    def forward(self, x, mask, cos, sin):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为(batch_size, sequence_length, emb_dim)
            mask (torch.Tensor): 注意力掩码，形状为(sequence_length, sequence_length)
            cos (torch.Tensor): RoPE的余弦值，形状为(1, head_dim, context_length)
            sin (torch.Tensor): RoPE的正弦值，形状为(1, head_dim, context_length)

        返回:
            torch.Tensor: 输出张量，形状为(batch_size, sequence_length, emb_dim)
        """
        # 残差连接的快捷路径
        shortcut = x

        # 对输入进行归一化
        x = self.norm1(x)
        # 通过自注意力模块处理
        x = self.att(x, mask, cos, sin)  # 输出形状 [batch_size, num_tokens, emb_size]
        # 残差连接，将自注意力输出与原始输入相加
        x = x + shortcut  

        # 前馈神经网络块的残差连接快捷路径
        shortcut = x
        # 对自注意力输出进行归一化
        x = self.norm2(x)
        # 通过前馈神经网络处理
        x = self.ff(x)
        # 残差连接，将前馈神经网络输出与自注意力输出相加
        x = x + shortcut

        return x


class FeedForward(nn.Module):
    def __init__(self, cfg):
        """
        初始化前馈神经网络模块。

        参数:
            cfg (dict): 模型配置，包含以下键:
                - emb_dim (int): 词嵌入的维度，词向量的大小
                - hidden_dim (int): 前馈神经网络中间层的维度
                - dtype (torch.dtype): 模型参数和计算使用的低精度数据类型
        """
        super().__init__()
        # 第一个线性层，将输入维度映射到隐藏层维度
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        # 第二个线性层，将输入维度映射到隐藏层维度
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        # 第三个线性层，将隐藏层维度映射回输出维度
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为(batch_size, sequence_length, emb_dim)

        返回:
            torch.Tensor: 输出张量，形状为(batch_size, sequence_length, emb_dim)
        """
        # 通过第一个线性层
        x_fc1 = self.fc1(x)
        # 通过第二个线性层
        x_fc2 = self.fc2(x)
        # 使用SiLU激活函数对第一个线性层输出进行激活，并与第二个线性层输出相乘
        x = nn.functional.silu(x_fc1) * x_fc2
        # 通过第三个线性层
        return self.fc3(x)


class GroupedQueryAttention(nn.Module):
    def __init__(
            self, d_in, d_out, num_heads,
            num_kv_groups,
            dtype=None
    ):
        """
        初始化分组查询注意力模块。

        分组查询注意力机制通过将键和值分组来减少计算量，同时保持多头注意力的多样性。

        参数:
            d_in (int): 输入向量的维度。
            d_out (int): 输出向量的维度，通常等于输入维度。
            num_heads (int): 多头注意力机制中头的数量，用于并行计算不同的注意力表示。
            num_kv_groups (int): 键值组的数量，用于将键和值分组以减少计算量。
            dtype (torch.dtype, optional): 模型参数和计算使用的低精度数据类型。默认为None。
        """
        super().__init__()
        # 确保输出维度可以被头的数量整除，且头的数量可以被键值组的数量整除
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        # 每个头的维度
        self.head_dim = d_out // num_heads

        # 键线性变换层，将输入维度映射到键的维度
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        # 值线性变换层，将输入维度映射到值的维度
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        # 每个键值组对应的头的数量
        self.group_size = num_heads // num_kv_groups

        # 查询线性变换层，将输入维度映射到查询的维度
        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        # 输出线性变换层，将多头注意力输出映射回输出维度
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

    def forward(self, x, mask, cos, sin):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, num_tokens, d_in)。
            mask (torch.Tensor): 注意力掩码，形状为 (sequence_length, sequence_length)。
            cos (torch.Tensor): RoPE 的余弦值，形状为 (1, head_dim, context_length)。
            sin (torch.Tensor): RoPE 的正弦值，形状为 (1, head_dim, context_length)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, num_tokens, d_out)。
        """
        # 获取输入张量的维度
        b, num_tokens, d_in = x.shape  # b: batch_size, num_tokens: 序列长度, d_in: 输入维度

        # 通过查询线性变换层计算查询向量，形状为 (b, num_tokens, d_out)
        queries = self.W_query(x)
        # 通过键线性变换层计算键向量，形状为 (b, num_tokens, num_kv_groups * head_dim)
        keys = self.W_key(x)
        # 通过值线性变换层计算值向量，形状为 (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)

        # 重塑查询、键、值张量以分离多头
        # 查询形状: (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        # 键形状: (b, num_tokens, num_kv_groups, head_dim)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        # 值形状: (b, num_tokens, num_kv_groups, head_dim)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)

        # 转置键和值张量以匹配多头维度，形状: (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # 转置查询张量以匹配多头维度，形状: (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)

        # 应用 RoPE 位置编码
        keys = apply_rope(keys, cos, sin)
        queries = apply_rope(queries, cos, sin)

        # 扩展键和值以匹配头的数量
        # 例如，如果 group_size = 2，则每个键和值将被重复两次
        # 键形状: (b, num_heads, num_tokens, head_dim)
        keys = keys.repeat_interleave(self.group_size, dim=1)
        # 值形状: (b, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1)

        # 计算缩放点积注意力（自注意力），形状: (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)

        # 使用掩码填充注意力得分，屏蔽未来的信息
        attn_scores = attn_scores.masked_fill(mask[:num_tokens, :num_tokens], -torch.inf)

        # 对注意力得分进行 softmax 归一化
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        assert keys.shape[-1] == self.head_dim

        # 计算上下文向量，形状: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并多头，形状: (b, num_tokens, d_out)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        # 可选的输出投影
        context_vec = self.out_proj(context_vec) 

        return context_vec


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None, dtype=torch.float32):
    """
    计算RoPE（旋转位置编码）的余弦和正弦值。

    参数:
        head_dim (int): 每个头的维度，必须为偶数。
        theta_base (float, optional): RoPE的基础参数θ，默认为10,000。
        context_length (int, optional): 上下文长度，默认为4096。
        freq_config (dict, optional): 频率配置参数，包含以下键:
            - original_context_length (int): 原始上下文长度。
            - low_freq_factor (float): 低频部分的缩放因子。
            - high_freq_factor (float): 高频部分的缩放因子。
            - factor (float): 频率缩放的因子。
        dtype (torch.dtype, optional): 张量数据类型，默认为torch.float32。

    返回:
        (torch.Tensor, torch.Tensor): RoPE的余弦和正弦值，形状均为 (context_length, head_dim)。
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # 计算逆频率，公式为 1 / (θ_base ^ (2i / head_dim))，其中 i 从0到(head_dim // 2 - 1)
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # 频率调整
    if freq_config is not None:
        # 计算低频和高频的波长
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        # 计算当前频率对应的波长
        wavelen = 2 * torch.pi / inv_freq

        # 根据低频因子调整逆频率
        # 如果波长大于低频波长，则逆频率除以因子，否则保持不变
        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        # 计算平滑因子，用于在低频和高频之间平滑过渡
        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        # 计算平滑后的逆频率
        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        # 判断当前频率是否在中等频率范围内
        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        # 如果是中等频率，则使用平滑后的逆频率，否则使用调整后的逆频率
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        # 最终的逆频率
        inv_freq = inv_freq_llama

    # 生成从0到context_length-1的位置索引
    positions = torch.arange(context_length, dtype=dtype)

    # 计算角度，公式为 position * inv_freq
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # 扩展角度以匹配head_dim
    # 将角度重复一次，以匹配head_dim（因为head_dim是偶数）
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # 预计算正弦和余弦
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    # 返回RoPE的余弦和正弦值
    return cos, sin


def apply_rope(x, cos, sin):
    """
    应用RoPE（旋转位置编码）到输入张量。

    参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, num_heads, seq_len, head_dim)。
        cos (torch.Tensor): RoPE的余弦值，形状为 (seq_len, head_dim)。
        sin (torch.Tensor): RoPE的正弦值，形状为 (seq_len, head_dim)。

    返回:
        torch.Tensor: 应用了RoPE后的张量，形状为 (batch_size, num_heads, seq_len, head_dim)。
    """
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # 将输入张量拆分为前半部分和后半部分
    x1 = x[..., : head_dim // 2]  # 前半部分
    x2 = x[..., head_dim // 2:]   # 后半部分

    # 调整cos和sin的形状以匹配输入张量
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # 应用旋转变换
    rotated = torch.cat((-x2, x1), dim=-1)  # 旋转后的张量
    x_rotated = (x * cos) + (rotated * sin) # 应用余弦和正弦进行旋转

    # 在应用cos和sin旋转后，可以使用较低精度
    # 返回应用了RoPE后的张量
    return x_rotated.to(dtype=x.dtype)


def rescale_theta(theta_old, context_length_old, context_length_new):
    """
    重新缩放θ参数，用于调整RoPE（旋转位置编码）的基础参数θ。

    参数:
        theta_old (float): 原始的θ值。
        context_length_old (int): 原始的上下文长度。
        context_length_new (int): 新的上下文长度。

    返回:
        float: 重新缩放后的θ值。
    """
    # 计算缩放因子，公式为 新上下文长度 / 旧上下文长度
    scaling_factor = context_length_new / context_length_old

    # 通过缩放因子重新计算θ值
    theta_new = theta_old * scaling_factor

    # 返回重新缩放后的θ值
    return theta_new


def text_to_token_ids(text, tokenizer):
    """
    将文本转换为对应的token ID序列。

    参数:
        text (str): 输入的文本字符串。
        tokenizer (transformers.PreTrainedTokenizer): 用于编码的tokenizer对象。

    返回:
        torch.Tensor: 包含token ID的张量，形状为 (1, num_tokens)。
    """
    # 使用tokenizer对输入文本进行编码，得到token ID列表
    encoded = tokenizer.encode(text)  

    # 将token ID列表转换为张量，并在最前面添加一个维度作为批量维度
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    # 返回包含token ID的张量
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    将token ID序列转换回文本字符串。

    参数:
        token_ids (torch.Tensor): 包含token ID的张量，形状为 (batch_size, num_tokens)。
        tokenizer (transformers.PreTrainedTokenizer): 用于解码的tokenizer对象。

    返回:
        str: 转换回的原文本字符串。
    """
    # 如果有批量维度，则将其移除，只保留token ID序列
    flat = token_ids.squeeze(0) 

    # 使用tokenizer将token ID列表解码回文本字符串
    # 返回解码后的文本字符串
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    使用模型生成文本。

    参数:
        model (nn.Module): 要使用的预训练模型。
        idx (torch.Tensor): 输入的token ID序列，形状为 (batch_size, num_tokens)。
        max_new_tokens (int): 要生成的token数量。
        context_size (int): 模型在生成时考虑的上下文长度。
        temperature (float, optional): 采样温度，用于控制生成的多样性。默认为0.0（贪婪搜索）。
        top_k (int, optional): top-k采样参数，默认为None（不使用top-k采样）。
        eos_id (int, optional): 结束序列的token ID，默认为None（不提前停止生成）。

    返回:
        torch.Tensor: 生成后的token ID序列，形状为 (batch_size, num_tokens + max_new_tokens)。
    """
    # 对于生成过程，使用循环逐步生成新的token
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        # 获取当前输入序列的最后context_size个token作为条件
        with torch.no_grad():
            logits = model(idx_cond)
        # 使用模型计算logits，不计算梯度以节省内存
        # 只关注最后一个时间步的logits，因为我们是逐个生成新的token
        logits = logits[:, -1, :]

        # 使用top-k采样过滤logits
        if top_k is not None:
            # 获取前top_k个最大的logits值
            top_logits, _ = torch.topk(logits, top_k)
            # 获取这些logits中的最小值作为阈值
            min_val = top_logits[:, -1]
            # 将小于阈值的logits设为负无穷，以便在后续的softmax中忽略这些值
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # # 应用temperature缩放
        if temperature > 0.0:
            # 对logits进行温度缩放，降低或增加生成的多样性
            logits = logits / temperature

            # 对缩放后的logits应用softmax，得到概率分布
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # 根据概率分布进行多项式采样，得到下一个token的ID
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # 如果不使用温度缩放，则使用贪婪搜索选择概率最高的token
        else:
            # 选择logits中最大的值的索引作为下一个token的ID
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # 如果指定了结束序列的token ID，并且生成的token中有结束序列的token，则提前停止生成
        if idx_next == eos_id:
            break

        # 将生成的下一个token ID拼接到当前的序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

     # 返回最终生成的token ID序列
    return idx
