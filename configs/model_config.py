import torch


# 要运行的 LLM 名称，只使用本地模型。列表中本地模型将在启动项目时全部加载。
# 列表中第一个模型将作为 API 和 WEBUI 的默认模型。
LLM_MODELS = ["Qwen2-0.5B-Instruct"]

LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型名称 -> 绝对路径
MODEL_PATH = {
    "Qwen2-0.5B-Instruct": r"D:\models\Qwen2-0.5B-Instruct"
}

# 历史对话长度
HISTORY_LEN = 3

# 生成的最大token
MAX_TOKENS = 2048

# temperature < 1.0, 输出更具确定性和重复性 > 1.0, 增加随机性
TEMPERATURE = 0.5


# 【-2~2】
# 控制生成文本中特定单词或短语的存在频率
# 较高的presence_penalty 会模型更倾向于在生成的文本中包含多样性更大的单词和短语，减少重复性。
# 较低的presence_penalty模型可能更倾向于生成包含和输入相近的文本，重复性
PRESENCE_PENALTY = 1.05