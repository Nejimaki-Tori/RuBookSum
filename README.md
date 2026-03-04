# BoookSum

**Реферирование художественной литературы посредством больших языковых моделей**

---

## Содержание

1. [Описание](#описание)
2. [Датасет](#датасет)
3. [Структура репозитория](#структура-репозитория)
4. [Установка](#установка)
5. [Параметры](#параметры)
6. [Метрики](#метрики)

---

## Описание

Проект посвящён исследованию подходов к автоматическому реферированию художественных текстов с использованием больших языковых моделей. Художественные тексты отличаются сложной стилистикой и семантикой, а ограниченное контекстное окно современных LLM не позволяет обрабатывать целый роман за раз. BoookSum предоставляет фреймворк для сравнения нескольких методов компактного представления содержания произведений без существенной потери смысла.

Реализованы и исследованы следующие методы:

* **Иерархический метод** — рекурсивное объединение аннотаций чанков с опциональной фильтрацией дублирующих узлов.
* **«Чертёжный» метод (blueprint)** — генерация плана в форме вопрос-ответ и последующая сборка аннотации из ответов.
* **Иерархический с фильтрацией узлов** — улучшенный иерархический метод с удалением семантически близких фрагментов.
* **«Чертёжный» с кластеризацией вопросов** — вариант чертёжного метода, где вопросы группируются по смыслу.

Для оценки качества используются метрики **ROUGE-L** и **BERTScore**.

---

## Датасет

В json-файле `combined_data.json` находятся собранные тексты книг и их аннотаций, а также информация об авторе произведения и его названия.
Датасет можно скачать по ссылке: https://huggingface.co/datasets/NejimakiTori/literature_sum

---

## Структура репозитория

```text
.
├── combined_data.json          # Данные (книги и аннотации)
├── src/                    # Реализации методов
│   ├── hierarchical.py         # Иерархический метод
│   ├── iterative.py            # Итеративный метод
│   ├── pseudo.py               # Псевдо-генерация по названию
│   ├── blueprint.py            # Text-Blueprint
│   ├── prompts.py              # Промпты для модели
│   └── methods.py              # Объединяет в себе все реализованные методы
├── utils.py                    # LLM-клиент и вспомогательные функции (разбиение на чанки и др.)
├── metrics.py                  # Метрики: ROUGE-L, BERTScore
├── local_benchmark_vllm.ipynb    # Ноутбук для локального запуска через VLLM
├── main.ipynb                  # Демонстрация работы фреймворка
└── requirements.txt            # Зависимости
```

---

## Установка

```bash
# 1. Клонируем репозиторий
git clone https://github.com/Nejimaki-Tori/BoookSum.git
cd BoookSum

# 2. Устанавливаем зависимости
pip install -r requirements.txt

# 3. Желательно установить torch с поддержкой вычислений на GPU
pip install torch==1.12.1+cu114 torchvision==0.13.1+cu114 torchaudio==0.12.1 \
--extra-index-url https://download.pytorch.org/whl/cu114
```

---

## Пример использования

```python
import torch
from sentence_transformers import SentenceTransformer
from run_bench import run_benchmark

url = 'YOUR_URL'
key = 'YOUR_KEY'
model_name = 'YOUR_MODEL_NAME'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = SentenceTransformer('deepvk/USER-bge-m3').to(device)

result = await run_benchmark(
    api=url,
    key=key,
    model_name=model_name,
    concurrency=40,
    output_dir='output_hierarchical_default',
    number_of_books=40,
    encoder_name = 'deepvk/USER-bge-m3',
    device = 'auto',
    method = 'hierarchical',
    mode = 'default',
    initial_word_limit = 500,
    cap_chars = 80000,
    shared_encoder=encoder,
    shared_device=device,
)
```

## Параметры

| Метод                           | Параметр             | Значение по умолчанию | Описание                                                           |
| ------------------------------- | -------------------- | --------------------- | ------------------------------------------------------------------ |
| `Hierarchical.run`              | `initial_word_limit` | `500`                 | Максимальное число слов в аннотации                                |
|                                 | `mode`               | `default`             | Фильтрация семантически близких узлов                              |
| `Blueprint.run`                 | `initial_word_limit` | `500`                 | Лимит слов в итоговой аннотации                                    |
|                                 | `mode`               | `'default'`           | Режим работы Text-Blueprint: `default` или `cluster`               |
| `Evaluater.evaluate_annotation` | —                    | —                     | Использует метрики ROUGE-L, BERTScore                              |

---

## Метрики

* **ROUGE-L** — лексическое совпадение по наибольшей общей подпоследовательности.

* **BERTScore** — семантическое совпадение с эталоном с помощью SentenceTransformer.
