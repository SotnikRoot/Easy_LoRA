import os
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# ============================================================
# НАСТРОЙКИ
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_DIR = "./models"
JSON_FILE = "result_LoRA.json"
OUTPUT_DIR = "./qwen-lora-finetuned_2.0"

# Режимы скорости (выберите один):
SPEED_MODE = "ULTRA_FAST"  # "ULTRA_FAST", "FAST", "BALANCED", "QUALITY"

SPEED_CONFIGS = {
    "ULTRA_FAST": {
        "epochs": 1,
        "batch_size": 4,
        "grad_accum": 2,
        "max_length": 256,
        "lora_r": 4,
        "learning_rate": 3e-4,
        "logging_steps": 50,
        "description": "Максимальная скорость (~4-5 часов на intel i5)"
    },
    "FAST": {
        "epochs": 2,
        "batch_size": 2,
        "grad_accum": 4,
        "max_length": 512,
        "lora_r": 8,
        "learning_rate": 2e-4,
        "logging_steps": 25,
        "description": "Быстро с хорошим качеством (~15-17 часов на intel i5)"
    },
    "BALANCED": {
        "epochs": 3,
        "batch_size": 1,
        "grad_accum": 8,
        "max_length": 512,
        "lora_r": 12,
        "learning_rate": 1e-4,
        "logging_steps": 20,
        "description": "Баланс скорости и качества (~29-35 часов на intel i5)"
    },
    "QUALITY": {
        "epochs": 5,
        "batch_size": 1,
        "grad_accum": 8,
        "max_length": 1024,
        "lora_r": 16,
        "learning_rate": 5e-5,
        "logging_steps": 10,
        "description": "Максимальное качество (~60-70 часов на intel i5)"
    }
}

MODE = SPEED_CONFIGS[SPEED_MODE]

# ============================================================
# УЛУЧШЕННАЯ ЗАГРУЗКА ДАННЫХ
# ============================================================

def load_and_format_json_data(json_file_path):
    """Загружает и форматирует JSON данные"""
    print(f"📖 Загружаю данные из {json_file_path}...")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        exit()
    
    training_examples = []
    
    for item in data["data"]:
        instruction = item.get('instruction', '').strip()
        output = item.get('output', '').strip()
        input_text = item.get('input', '').strip()
        
        # Пропускаем пустые примеры
        if not instruction or not output:
            continue
        
        # Формат в стиле ChatML (более чистый)
        if input_text:
            text = f"<|im_start|>system\nТы - эксперт по 1С:Предприятие.Элемент. Отвечай точно и по делу.<|im_end|>\n<|im_start|>user\nКонтекст: {input_text}\n\nВопрос: {instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        else:
            text = f"<|im_start|>system\nТы - эксперт по 1С:Предприятие.Элемент. Отвечай точно и по делу.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        
        training_examples.append(text)
    
    print(f"📊 Загружено {len(training_examples)} примеров")
    
    # Показываем пример
    if training_examples:
        print("\n✅ Пример обучающих данных:")
        print("-" * 60)
        print(training_examples[0][:500])
        print("-" * 60 + "\n")
    
    return training_examples

# ============================================================
# СОЗДАНИЕ ПАПОК
# ============================================================

print("📁 Создаю папки...")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================

print(f"📥 Загружаю модель {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_DIR,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_DIR,
    torch_dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)

print("✅ Модель загружена!")

# ============================================================
# ПОДГОТОВКА ДАННЫХ
# ============================================================

text_examples = load_and_format_json_data(JSON_FILE)

if len(text_examples) < 10:
    print("⚠️ ВНИМАНИЕ: Слишком мало данных для обучения!")
    print(f"Найдено только {len(text_examples)} примеров")
    print("Рекомендуется минимум 100+ примеров для качественного обучения")

dataset_dict = {"text": text_examples}
dataset = Dataset.from_dict(dataset_dict)

def tokenize_function(examples):
    """Токенизация с правильными параметрами"""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MODE["max_length"],  # Увеличено для более длинных ответов
        padding="max_length",
        return_tensors=None
    )
    result["labels"] = result["input_ids"].copy()
    return result

print("🔄 Токенизирую данные...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Токенизация"
)

# Разделяем на train и eval
split_dataset = tokenized_dataset.train_test_split(test_size=0.005, seed=42)

print(f"📊 Train: {len(split_dataset['train'])}, Eval: {len(split_dataset['test'])}")

# ============================================================
# НАСТРОЙКА LoRA (более агрессивная)
# ============================================================

print("🔧 Настраиваю LoRA...")

lora_config = LoraConfig(
    r=MODE["lora_r"],
    lora_alpha=32,  # Увеличено с 16 до 32
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# УЛУЧШЕННЫЕ ПАРАМЕТРЫ ОБУЧЕНИЯ
# ============================================================

print("⚙️ Настраиваю обучение...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # КРИТИЧНО: больше эпох!
    num_train_epochs=MODE["epochs"],
    
    # Меньше batch size для лучшего обучения
    per_device_train_batch_size=MODE["batch_size"],
    per_device_eval_batch_size=MODE["batch_size"],
    gradient_accumulation_steps=MODE["grad_accum"],
    
    # Learning rate
    learning_rate=MODE["learning_rate"],  # Было 2e-4, снижено для стабильности
    warmup_steps=MODE["logging_steps"],  # Было 10
    
    # Логирование и сохранение
    logging_steps=MODE["logging_steps"],
    save_steps=MODE["logging_steps"],
    eval_steps=MODE["logging_steps"],
    save_total_limit=3,
    
    # Оптимизация
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    # Техническое
    fp16=False,
    dataloader_num_workers=0,
    report_to="none",
    
    # Важно!
    remove_unused_columns=False,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=data_collator,
)

# ============================================================
# ОБУЧЕНИЕ
# ============================================================

print("\n" + "="*60)
print("🚀 НАЧИНАЮ ОБУЧЕНИЕ")
print("="*60 + "\n")

trainer.train()

print("\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!\n")

# ============================================================
# СОХРАНЕНИЕ
# ============================================================

print("💾 Сохраняю модель...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Модель сохранена: {OUTPUT_DIR}\n")

# ============================================================
# УЛУЧШЕННОЕ ТЕСТИРОВАНИЕ
# ============================================================

print("="*60)
print("🧪 ТЕСТИРУЮ МОДЕЛЬ")
print("="*60 + "\n")

model.eval()

def generate_text(prompt, max_new_tokens=256):
    """Улучшенная генерация"""
    # Формируем промпт в формате обучения
    formatted_prompt = f"<|im_start|>system\nТы - эксперт по 1С:Предприятие.Элемент. Отвечай точно и по делу.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # Генерируем только новые токены
            temperature=0.3,  # Снижено с 0.7 для более точных ответов
            top_p=0.85,  # Снижено с 0.9
            top_k=40,  # Добавлено
            repetition_penalty=1.15,  # Увеличено с 1.1
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,  # Предотвращаем повторение фраз
        )
    
    # Декодируем только новую часть
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Извлекаем ответ после assistant
    if "<|im_start|>assistant" in full_output:
        response = full_output.split("<|im_start|>assistant")[-1]
        response = response.replace("<|im_end|>", "").strip()
    else:
        response = full_output[len(formatted_prompt):].strip()
    
    return response

# Тестовые вопросы
test_prompts = [
    "Что такое 1С:Предприятие.Элемент?",
    "Как создать новое приложение?",
    "Расскажи о панели управления"
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"Тест {i}: {prompt}")
    print("-" * 60)
    try:
        response = generate_text(prompt)
        print(response)
    except Exception as e:
        print(f"Ошибка: {e}")
    print("-" * 60 + "\n")

print("🎉 Готово!")
