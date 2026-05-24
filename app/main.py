import os
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Запуск приложения...")

HF_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")
if HF_TOKEN is None:
    print("Внимание: переменная HF_TOKEN не задана. Возможно, модель потребует аутентификации.")

PROMPT = "Language modeling is "
MODEL_PATH = "allenai/Bolmo-1B"
SEED = 42

print(f"Установка генератора случайных чисел с фиксированным SEED={SEED} для воспроизводимости результатов.")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cudnn.deterministic = True
print(f"Используется устройство: {device}")

print(f"Загрузка модели из {MODEL_PATH}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    token=HF_TOKEN
).to(device)
print(f"Модель загружена из {MODEL_PATH} и перемещена на устройство {device}.")

print("Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    token=HF_TOKEN
)
print("Токенизатор загружен.")

print(f"Генерация текста на основе промпта: '{PROMPT}'...")
inputs = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(device)
print(f"Промпт токенизирован и перемещен на устройство {device}.")

print("Генерация текста...")
outputs = model.generate(inputs, max_new_tokens=256, do_sample=False)
print("Текст сгенерирован.")

print("Декодирование сгенерированного текста...")
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Сгенерированный текст декодирован.")

print("Prompt:", PROMPT)
print("Generated:", text)
