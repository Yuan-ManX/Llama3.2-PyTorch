import os
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe


class Llama3Tokenizer:
    """
    Llama3Tokenizer 类用于将文本编码为模型可接受的token序列，以及将token序列解码回文本。
    它使用了tiktoken库进行字节对编码（BPE）。

    """
    def __init__(self, model_path):
        """
        初始化Llama3Tokenizer实例。

        参数:
            model_path (str): 模型的路径，用于加载BPE词汇表和模型配置。
        
        异常:
            AssertionError: 如果提供的model_path不是一个有效的文件路径，则抛出此异常。
        """
        # 检查提供的model_path是否指向一个存在的文件
        assert os.path.isfile(model_path), f"Model file {model_path} not found"

        # 从指定的model_path加载BPE词汇表
        mergeable_ranks = load_tiktoken_bpe(model_path)

        # 定义特殊token及其对应的token ID
        self.special_tokens = {
            "<|begin_of_text|>": 128000,    # 文本开始标记
            "<|end_of_text|>": 128001,      # 文本结束标记
            "<|start_header_id|>": 128006,  # 标题开始标记
            "<|end_header_id|>": 128007,    # 标题结束标记
            "<|eot_id|>": 128009,           # EOT（End of Token）标记
        }

        # 为保留的特殊token分配token ID，从128002开始，最多256个
        self.special_tokens.update({
            f"<|reserved_{i}|>": 128002 + i for i in range(256) if (128002 + i) not in self.special_tokens.values()
        })

        # 创建tiktoken.Encoding对象，用于编码和解码文本
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,  # 编码名称，通常为模型文件名
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,  # 可合并的rank，用于字节对编码
            special_tokens=self.special_tokens  # 特殊token及其对应的token ID
        )

    def encode(self, text, bos=False, eos=False, allowed_special=set(), disallowed_special=()):
        """
        将输入文本编码为token序列。

        参数:
            text (str): 要编码的文本。
            bos (bool): 是否在编码的token序列前添加文本开始标记（<|begin_of_text|>）。默认为False。
            eos (bool): 是否在编码的token序列后添加文本结束标记（<|end_of_text|>）。默认为False。
            allowed_special (set): 允许的特殊token集合。默认为空集合，表示不允许任何特殊token。
            disallowed_special (tuple): 不允许的特殊token元组。默认为空元组，表示不限制任何特殊token。
        
        返回:
            list: 编码后的token序列。
        """
        # 如果bos为True，在token序列前添加文本开始标记
        if bos:
            tokens = [self.special_tokens["<|begin_of_text|>"]]
        else:
            tokens = []

        # 使用tiktoken的encode方法对文本进行编码，并添加允许的特殊token
        tokens += self.model.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special)

        # 如果eos为True，在token序列后添加文本结束标记
        if eos:
            tokens.append(self.special_tokens["<|end_of_text|>"])
        return tokens

    def decode(self, tokens):
        """
        将token序列解码回文本。

        参数:
            tokens (list): 要解码的token序列。
        
        返回:
            str: 解码后的文本。
        """
        return self.model.decode(tokens)


class ChatFormat:
    """
    ChatFormat 类用于将聊天消息格式化为模型可接受的token序列，以及将token序列解码回文本。
    它依赖于一个Tokenizer实例来进行编码和解码。

    """
    def __init__(self, tokenizer):
        """
        初始化ChatFormat实例。

        参数:
            tokenizer (Tokenizer): 一个已经初始化的Tokenizer实例，用于编码和解码文本。
        
        属性:
            self.tokenizer (Tokenizer): 存储传入的Tokenizer实例。
        """
        self.tokenizer = tokenizer

    def encode_header(self, message):
        """
        编码消息的头部信息，包括角色（如用户或助手）和其他元数据。

        参数:
            message (dict): 包含消息信息的字典，通常包括 'role' 和 'content' 键。
        
        返回:
            list: 编码后的token序列，代表消息的头部信息。
        
        流程:
            1. 初始化一个空的token列表。
            2. 在token列表前添加开始头部标记 `<|start_header_id|>`。
            3. 使用Tokenizer编码消息的 'role' 字段，不添加开始和结束标记。
            4. 在token列表中添加结束头部标记 `<|end_header_id|>`。
            5. 添加换行符的编码，以分隔头部和内容。
            6. 返回最终的token列表。
        """

        tokens = []
        # 添加开始头部标记
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        # 编码消息的角色（如 "user" 或 "assistant"）并添加到tokens中
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        # 添加结束头部标记
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        # 添加换行符的编码，用于分隔头部和内容
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))

        return tokens

    def encode(self, text):
        """
        编码完整的聊天消息，包括头部和内容。

        参数:
            text (str): 要编码的消息内容。
        
        返回:
            list: 编码后的完整token序列，包括头部和内容。
        
        流程:
            1. 创建一个包含 'role' 和 'content' 的消息字典，默认为用户消息。
            2. 调用 `encode_header` 方法编码消息的头部信息。
            3. 使用Tokenizer编码消息的内容，并去除首尾空白字符。
            4. 在token序列后添加结束标记 `<|eot_id|>`。
            5. 返回最终的token序列。
        """

        message = {
            "role": "user",  # 默认消息角色为用户
            "content": text
        }

        # 编码消息头部
        tokens = self.encode_header(message)

        # 编码消息内容，去除首尾空白字符，不添加开始和结束标记
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )

        # 添加结束标记
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])

        return tokens

    def decode(self, token_ids):
        """
        解码token序列回文本。

        参数:
            token_ids (list): 要解码的token序列。
        
        返回:
            str: 解码后的文本。
        
        流程:
            1. 使用Tokenizer的decode方法将token序列解码回文本。
            2. 返回解码后的文本。
        """
        return self.tokenizer.decode(token_ids)


def clean_text(text, header_end="assistant<|end_header_id|>\n\n"):
    """
    清理文本，去除消息头部信息，只保留内容部分。

    参数:
        text (str): 要清理的原始文本，通常包含头部信息和内容。
        header_end (str): 头部结束标记的字符串，默认为 "assistant<|end_header_id|>\n\n"。
    
    返回:
        str: 清理后的文本，只包含内容部分。
    
    流程:
        1. 使用字符串的find方法查找头部结束标记在文本中的索引位置。
        2. 如果找到头部结束标记：
            - 返回从标记之后开始到文本末尾的子字符串，并去除首尾空白字符。
        3. 如果未找到头部结束标记：
            - 返回原始文本。
    """
    # 查找头部结束标记 "<|end_header_id|>" 的索引位置
    index = text.find(header_end)

    if index != -1:
        # 返回从标记之后开始的子字符串，并去除首尾空白字符
        return text[index + len(header_end):].strip()
    
    else:
        # 如果未找到标记，返回原始文本
        return text
