# BoookSum

**Реферирование художественной литературы посредством больших языковых моделей**

---

## Содержание

1. [Описание](#описание)
2. [Требования](#требования)
3. [Датасет](#датасет)
4. [Структура репозитория](#структура-репозитория)
5. [Установка](#установка)
6. [Параметры](#параметры)
7. [Метрики](#метрики)

---

## Описание

Проект посвящён исследованию подходов к автоматическому реферированию художественных текстов с использованием больших языковых моделей. Художественные тексты отличаются сложной стилистикой и семантикой, а ограниченное контекстное окно современных LLM не позволяет обрабатывать целый роман за раз. BoookSum предоставляет фреймворк для сравнения нескольких методов компактного представления содержания произведений без существенной потери смысла.

Реализованы и исследованы следующие методы:

* **Иерархический метод** — рекурсивное объединение аннотаций чанков с опциональной фильтрацией дублирующих узлов.
* **«Чертёжный» метод (blueprint)** — генерация плана в форме вопрос-ответ и последующая сборка аннотации из ответов.
* **Иерархический с фильтрацией узлов** — улучшенный иерархический метод с удалением семантически близких фрагментов.
* **«Чертёжный» с кластеризацией вопросов** — вариант чертёжного метода, где вопросы группируются по смыслу.

Для оценки качества используются метрики **ROUGE-L**, **BERTScore**, **Coverage** и **Answer Similarity**.

---

## Требования

* Python 3.8+
* asyncio
* openai
* scipy
* nltk
* transformers
* tqdm
* evaluate
* sentence\_transformers
* numpy
* razdel
* rouge
* scikit-learn

Для взаимодействия с LLM требуется клиент, реализованный в `utils.py`, который обращается к серверу по API-ключу и URL.

---

## Датасет

В json-файле `combined_data.json` находятся собранные тексты книг и их аннотаций, а также информация об авторе произведения и его названия.

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
├── metrics.py                  # Метрики: ROUGE-L, BERTScore, Coverage, Answer Similarity
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
import sys
import torch
from sentence_transformers import SentenceTransformer
sys.path.append('src')
from methods import Summarisation

with open('Access_key.txt', 'r', encoding='utf-8') as file: # тут можно указать эндпоинты
    url, key = file.read().split()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = SentenceTransformer('deepvk/USER-bge-m3').to(device)
bench = Summarisation(url=url, key=key, model_name='ruadapt-qwen3-4b', device=device, encoder=encoder) # здесь указывается название модели
bench.prepare_enviroment()

result = await bench.run_benchmark_one_method(
    is_evalutation_needed=True, # нужен ли подсчет метрик
    number_of_books=1, # сколько книг будет обработано
    method='hierarchical', # метод сжатия
    mode='default', # режим для метода сжатия
    initial_word_limit=500, # максимальная длина аннотации (в символах)
    text_length_cap=80000, # слишком длинные тексты не будут обрабатываться
    save_json_path='benchmark_results.json' # куда сохранять результаты
)
```

## Параметры

| Метод                           | Параметр             | Значение по умолчанию | Описание                                                           |
| ------------------------------- | -------------------- | --------------------- | ------------------------------------------------------------------ |
| `Hierarchical.run`              | `initial_word_limit` | `500`                 | Максимальное число слов в аннотации                                |
|                                 | `mode`               | `default`             | Фильтрация семантически близких узлов                              |
| `Iterative.run`                 | `initial_word_limit` | `500`                 | Лимит слов при итеративном обновлении аннотации                    |
| `Blueprint.run`                 | `initial_word_limit` | `500`                 | Лимит слов в итоговой аннотации                                    |
|                                 | `mode`               | `'default'`           | Режим работы Text-Blueprint: `default` или `cluster`               |
| `Evaluater.evaluate_annotation` | —                    | —                     | Использует метрики ROUGE-L, BERTScore                              |

---

## Метрики

* **ROUGE-L** — лексическое совпадение по наибольшей общей подпоследовательности.

* **BERTScore** — семантическое совпадение с эталоном с помощью SentenceTransformer.
