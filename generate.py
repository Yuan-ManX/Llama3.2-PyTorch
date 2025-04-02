import os
import time
import urllib.request
import torch

from llama_model import Llama3Model, generate, text_to_token_ids, token_ids_to_text
from tokenizer import Llama3Tokenizer, ChatFormat, clean_text


######################################################### Model settings #########################################################
# 选择要使用的模型文件
MODEL_FILE = "llama3.2-1B-instruct.pth"
# MODEL_FILE = "llama3.2-1B-base.pth"   
# MODEL_FILE = "llama3.2-3B-instruct.pth"
# MODEL_FILE = "llama3.2-3B-base.pth"

# 设置模型的最大上下文长度。LLaMA3模型支持的最大上下文长度为131,072，但这里设置为8192以平衡性能和资源消耗。
MODEL_CONTEXT_LENGTH = 8192 

# 文本生成设置，根据模型文件名称选择不同的提示语。
# 如果模型文件名称中包含"instruct"，则使用指令提示；否则，使用简短的提示。
if "instruct" in MODEL_FILE:
    PROMPT = "What do cats eat?"
else:
    PROMPT = "Cats eat"

# 设置生成的最大新token数量。生成过程中模型将生成最多150个新token。
MAX_NEW_TOKENS = 150

# 设置生成温度。温度控制生成文本的随机性。温度为0表示贪婪搜索，生成最可能的下一个token。
TEMPERATURE = 0.

# 设置Top-K值。Top-K用于在生成过程中选择最可能的K个token。这里设置为1表示只选择最可能的下一个token。
TOP_K = 1


######################################################### Initialize model #########################################################

# 构建模型文件的URL。如果模型文件不存在，将从该URL下载模型。
url = f"{MODEL_FILE}"

if not os.path.exists(MODEL_FILE):
    print(f"Downloading {MODEL_FILE}...")
    # 从指定的URL下载模型文件并保存为MODEL_FILE
    urllib.request.urlretrieve(url, MODEL_FILE)
    print(f"Downloaded to {MODEL_FILE}")

# 根据模型文件名称导入相应的模型配置
# 如果模型文件名称中包含"1B"，则导入1B参数的模型配置；否则，如果包含"3B"，则导入3B参数的模型配置。
if "1B" in MODEL_FILE:
    from llama_model import LLAMA32_CONFIG_1B as LLAMA32_CONFIG
elif "3B" in MODEL_FILE:
    from llama_model import LLAMA32_CONFIG_3B as LLAMA32_CONFIG
else:
    raise ValueError("Incorrect model file name")

# 设置模型的上下文长度
LLAMA32_CONFIG["context_length"] = MODEL_CONTEXT_LENGTH

# 创建模型实例，并加载模型参数。
model = Llama3Model(LLAMA32_CONFIG)
# 从MODEL_FILE加载模型参数，weights_only=True表示只加载权重，不加载其他状态
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))

# 根据硬件环境设置设备。如果有可用的CUDA设备，则使用GPU；
# 如果支持MPS（Apple的Metal），则使用MPS；否则，使用CPU。
device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device)


######################################################### Initialize tokenizer #########################################################

# 定义分词器模型文件的名称
TOKENIZER_FILE = "tokenizer.model"

# 构建分词器文件的URL
url = f"{TOKENIZER_FILE}"

# 检查分词器文件是否已经存在。如果不存在，则从指定的URL下载分词器文件
if not os.path.exists(TOKENIZER_FILE):
    # 从URL下载分词器文件并保存为TOKENIZER_FILE
    urllib.request.urlretrieve(url, TOKENIZER_FILE)
    print(f"Downloaded to {TOKENIZER_FILE}")

# 使用下载的分词器文件初始化Llama3Tokenizer实例
tokenizer = Llama3Tokenizer("tokenizer.model")

# 如果模型文件名称中包含"instruct"，则将tokenizer包装在ChatFormat类中，以便处理指令格式的文本
if "instruct" in MODEL_FILE:
    tokenizer = ChatFormat(tokenizer)


######################################################### Generate text #########################################################

# 设置随机种子，以确保生成的文本具有可重复性
torch.manual_seed(123)

# 记录文本生成开始的时间
start = time.time()

# 将提示语Prompt转换为token ID，并移动到指定的设备（CPU或GPU）
token_ids = generate(
    model=model,  # 要使用的模型实例
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),  # 将提示语转换为token ID，并移动到设备上
    max_new_tokens=MAX_NEW_TOKENS,  # 生成的最大新token数量
    context_size=LLAMA32_CONFIG["context_length"],  # 模型的上下文长度
    top_k=TOP_K,  # Top-K值，用于在生成过程中选择最可能的K个token
    temperature=TEMPERATURE  # 温度，用于控制生成文本的随机性
)

# 输出生成文本所花费的时间
print(f"Time: {time.time() - start:.2f} sec")

# 如果有可用的CUDA设备，则计算生成过程中分配的最大内存，并转换为GB单位
if torch.cuda.is_available():
    # 获取生成过程中分配的最大内存（字节）
    max_mem_bytes = torch.cuda.max_memory_allocated()
    # 将字节转换为GB
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    # 输出最大内存使用量
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

# 将生成的token ID序列转换回文本
output_text = token_ids_to_text(token_ids, tokenizer)

# 如果模型文件名称中包含"instruct"，则清理输出文本，去除头部信息，只保留内容部分
if "instruct" in MODEL_FILE:
    output_text = clean_text(output_text)

# 输出最终的生成文本
print("\n\nOutput text:\n\n", output_text)
