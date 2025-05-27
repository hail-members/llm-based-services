from gpt4all import GPT4All

# load / save
model = GPT4All(
    "Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf",
)

# huggingface cli login
from huggingface_hub import login
login(token="XXXXXXXXXXXXXXXXXXXXXXXXX") # 여러분 토큰으로 바꿔보세요