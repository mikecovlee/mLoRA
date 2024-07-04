from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
    "~/workspace/models/THUDM/chatglm3-6b",
    device_map=
)