import os
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    StoppingCriteria,
    StoppingCriteriaList
)
from typing import List, Optional


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_DIR = "./models"

# Загружаем токенизатор с правильными настройками
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_DIR,
    trust_remote_code=True,
    padding_side="left"  # Важно для генерации
)

# Устанавливаем pad token если его нет
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Загружаем модель
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=MODEL_DIR,
    torch_dtype=torch.float32,
    device_map="auto" if torch.cuda.is_available() else "cpu",
    trust_remote_code=True
)

# Критерии остановки для целостного завершения
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: List[int], min_length: int = 20):
        self.stop_token_ids = stop_token_ids
        self.min_length = min_length
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] < self.min_length:
            return False
        
        # Проверяем последний токен
        last_token = input_ids[0, -1].item()
        if last_token in self.stop_token_ids:
            return True
            
        # Также проверяем конец предложения
        last_tokens = input_ids[0, -3:].tolist()
        if tokenizer.eos_token_id in last_tokens:
            return True
            
        return False


def format_prompt(prompt: str, strict_mode: bool = True) -> str:
    """Форматирование промпта с акцентом на точность и релевантность"""
    if strict_mode:
        system_message = """Ты - эксперт по 1С:Предприятие.Элемент. 
Твои ответы должны быть:
1. Максимально точными и фактологическими
2. Только по теме вопроса без отклонений
3. Полными и законченными по смыслу
4. Без выдумывания несуществующей информации
5. С четкой структурой и логикой

Если не знаешь точного ответа - так и скажи."""
    else:
        system_message = "Ты - эксперт по 1С:Предприятие.Элемент. Отвечай точно и по делу."
    
    return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""


def generate_precise_response(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.3,  # Низкая температура для минимального творчества
    top_p: float = 0.85,
    repetition_penalty: float = 1.2,
    num_beams: int = 3,
    do_sample: bool = True,
    min_new_tokens: int = 30,
    stop_sequences: Optional[List[str]] = None
) -> str:
    """
    Генерация максимально точных и целостных ответов
    """
    # Форматируем промпт
    formatted_prompt = format_prompt(prompt, strict_mode=True)
    
    # Токенизация
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048 - max_new_tokens
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Настраиваем стоп-токены
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
    
    # Добавляем кастомные стоп-последовательности
    if stop_sequences:
        for seq in stop_sequences:
            seq_ids = tokenizer.encode(seq, add_special_tokens=False)
            if len(seq_ids) > 0:
                stop_token_ids.append(seq_ids[0])
    
    stopping_criteria = StoppingCriteriaList([
        StopOnTokens(stop_token_ids, min_length=min_new_tokens)
    ])
    
    # Генерация с оптимальными параметрами для точности
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                
                # Параметры контроля качества
                temperature=temperature,
                top_p=top_p,
                top_k=40,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                
                # Контроль повторений
                no_repeat_ngram_size=4,
                encoder_no_repeat_ngram_size=3,
                bad_words_ids=[[tokenizer.pad_token_id]],
                
                # Beam search для лучшей когерентности
                num_beams=num_beams if not do_sample else 1,
                early_stopping=True,
                
                # Параметры завершения
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                
                # Критерии остановки
                stopping_criteria=stopping_criteria,
                
                # Параметры для плавного завершения
                length_penalty=0.8 if do_sample else 1.0,
                
                # Убираем предупреждения
                suppress_tokens=None,
            )
            
            # Декодируем ответ
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Извлекаем только ответ ассистента
            if "<|im_start|>assistant" in full_text:
                response = full_text.split("<|im_start|>assistant")[-1]
                response = response.split("<|im_end|>")[0].strip()
            else:
                # Альтернативный метод извлечения
                prompt_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
                response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                          skip_special_tokens=True)
            
            # Пост-обработка для целостности
            response = clean_response(response)
            
            return response
            
        except Exception as e:
            print(f"Ошибка генерации: {e}")
            return "Произошла ошибка при генерации ответа."


def clean_response(text: str) -> str:
    """Очистка и форматирование ответа для целостности"""
    # Убираем лишние пробелы
    text = ' '.join(text.split())
    
    # Проверяем завершенность предложений
    sentences = text.split('. ')
    if len(sentences) > 1:
        # Проверяем последнее предложение
        last_sentence = sentences[-1].strip()
        if last_sentence and not last_sentence.endswith(('.', '!', '?', ':', ';')):
            # Если последнее предложение обрывается, убираем его
            text = '. '.join(sentences[:-1]) + '.'
    
    # Убираем незаконченные строки
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.endswith(('...', '..', ' -', ' –')):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def generate_technical_answer(prompt: str) -> str:
    """Специализированная генерация для технических вопросов"""
    # Добавляем контекст для технических вопросов
    tech_context = """ВНИМАНИЕ: Отвечай ТОЛЬКО о 1С:Элемент. 
Используй ТОЛЬКО реальный синтаксис и возможности платформы.
Если вопрос не относится к 1С:Элемент - вежливо откажись отвечать.
Приводи реальные примеры кода с комментариями."""
    
    enhanced_prompt = f"{tech_context}\n\nВопрос: {prompt}"
    
    return generate_precise_response(
        enhanced_prompt,
        temperature=0.1,  # Очень низкая температура для кода
        top_p=0.7,
        do_sample=False,  # Детерминированный вывод для кода
        num_beams=5,
        repetition_penalty=1.3,
        max_new_tokens=768,
        stop_sequences=["```", "Вопрос:", "Замечание:", "Примечание:"]
    )


def generate_explanation(prompt: str) -> str:
    """Генерация для объяснений и описаний"""
    return generate_precise_response(
        prompt,
        temperature=0.4,
        top_p=0.9,
        do_sample=True,
        num_beams=3,
        repetition_penalty=1.15,
        max_new_tokens=1024,
        min_new_tokens=100
    )


# Тестирование с улучшенными промптами
test_prompts = [
    {
        "prompt": "Что такое язык Элемент?",
        "type": "explanation",
        "description": "Точное определение без лишней информации"
    },
    {
        "prompt": "Как создать новое приложение в 1С:Элемент?",
        "type": "technical",
        "description": "Пошаговая инструкция"
    },
    {
        "prompt": "Расскажи о панели управления в 1С:Элемент",
        "type": "explanation",
        "description": "Структурированное описание"
    },
    {
        "prompt": "Напиши запрос на языке Элемент который извлечёт из таблицы Сотрудники только тех у кого в названии должности есть приписка 'Директор'",
        "type": "technical",
        "description": "Конкретный пример кода"
    },
    {
        "prompt": "Чем отличается синтаксис Элемента от 1С:Предприятия?",
        "type": "technical",
        "description": "Сравнительный анализ"
    },
    {
        "prompt": "Напиши метод отправляющий уведомления по получаемым номерам телефона в 1С:Элемент",
        "type": "technical",
        "description": "Практический пример"
    }
]


def run_tests():
    """Запуск комплексного тестирования"""
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ УЛУЧШЕННОЙ ГЕНЕРАЦИИ")
    print("Максимальная точность, релевантность и целостность ответов")
    print("=" * 80)
    
    for i, test in enumerate(test_prompts, 1):
        print(f"\n{'='*60}")
        print(f"ТЕСТ {i}: {test['description']}")
        print(f"Тип: {test['type'].upper()}")
        print(f"Вопрос: {test['prompt']}")
        print(f"{'='*60}\n")
        
        try:
            if test['type'] == 'technical':
                response = generate_technical_answer(test['prompt'])
            else:
                response = generate_explanation(test['prompt'])
            
            print("ОТВЕТ:")
            print(response)
            
            # Анализ качества ответа
            print(f"\n{'─'*40}")
            print("АНАЛИЗ ОТВЕТА:")
            print(f"Длина: {len(response)} символов")
            print(f"Завершенность: {'✓' if response.strip().endswith(('.', '!', '?')) else '⚠'}")
            
            # Проверка релевантности
            keywords = ["элемент", "1с", "платформа", "приложение"]
            relevance = sum(1 for kw in keywords if kw in response.lower())
            print(f"Релевантность: {relevance}/4 ключевых слов")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*80)


def generate_with_retry(prompt: str, max_retries: int = 2) -> str:
    """Генерация с повторными попытками для улучшения качества"""
    best_response = ""
    best_score = 0
    
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                # Первая попытка: строгий режим
                response = generate_precise_response(prompt, temperature=0.2)
            else:
                # Последующие попыки: немного более креативный режим
                response = generate_precise_response(prompt, temperature=0.35)
            
            # Оценка качества ответа
            score = evaluate_response_quality(response, prompt)
            
            if score > best_score:
                best_score = score
                best_response = response
                
            if score > 0.8:  # Достаточно хороший ответ
                break
                
        except Exception as e:
            print(f"Попытка {attempt + 1} не удалась: {e}")
    
    return best_response if best_response else "Не удалось сгенерировать качественный ответ."


def evaluate_response_quality(response: str, prompt: str) -> float:
    """Простая оценка качества ответа"""
    score = 0.0
    
    # 1. Длина ответа (не слишком короткий)
    if len(response) > 50:
        score += 0.2
    
    # 2. Завершенность
    if response.strip().endswith(('.', '!', '?', ':')):
        score += 0.3
    
    # 3. Релевантность (проверка ключевых слов из промпта)
    prompt_words = set(prompt.lower().split())
    response_words = set(response.lower().split())
    common_words = len(prompt_words.intersection(response_words))
    if common_words > 0:
        score += min(0.3, common_words * 0.1)
    
    # 4. Структура (наличие абзацев или списков)
    if '\n' in response or '•' in response or '1.' in response[:50]:
        score += 0.2
    
    return min(score, 1.0)


if __name__ == "__main__":
    # Основной запуск
    print("Инициализация модели для точной генерации...")
    print(f"Модель: {MODEL_NAME}")
    print(f"Устройство: {model.device}")
    print(f"Токенизатор готов: {tokenizer is not None}")
    
    # Запуск тестов
    run_tests()
    
    # Пример интерактивного режима
    print("\n" + "="*80)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("Введите вопрос о 1С:Элемент или 'выход' для завершения")
    print("="*80)
    
    while True:
        try:
            user_input = input("\nВопрос: ").strip()
            
            if user_input.lower() in ['выход', 'exit', 'quit']:
                break
            
            if not user_input:
                continue
            
            print("\n" + "─"*60)
            print("Генерация оптимального ответа...")
            
            # Выбор стратегии в зависимости от вопроса
            if any(word in user_input.lower() for word in ['код', 'запрос', 'метод', 'синтаксис', 'как сделать']):
                response = generate_technical_answer(user_input)
            elif any(word in user_input.lower() for word in ['что такое', 'объясни', 'расскажи', 'описание']):
                response = generate_explanation(user_input)
            else:
                response = generate_with_retry(user_input)
            
            print("\nОТВЕТ:")
            print(response)
            print("─"*60)
            
        except KeyboardInterrupt:
            print("\n\nЗавершение работы...")
            break
        except Exception as e:
            print(f"\nОшибка: {e}")