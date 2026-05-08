# Breast Cancer Classifier (ML Pipeline)

Классификатор злокачественных опухолей груди на основе логистической регрессии.

В проекте есть полный цикл:
- предобработка данных
- обучение модели
- FastAPI для инференса
- DVC для версионирования артефактов
- pytest для тестирования

---

# Dataset

Используется датасет Breast Cancer Wisconsin из scikit-learn.

Характеристики:
- 569 образцов
- 30 признаков (радиус, текстура, периметр, площадь и т.д.)
- 2 класса: злокачественная (1) и доброкачественная (0) опухоль
- данные предобработаны и сохранены в `data/raw/`

---

# Модель

Логистическая регрессия:
- максимальное количество итераций: 10000
- стандартизация признаков: StandardScaler

---

# Pipeline

DVC pipeline состоит из двух этапов:

1. preprocess - разбиение данных на train/test
   - вход: `data/raw/X.csv`, `data/raw/y.csv`
   - выход: `data/processed/X_train.csv`, `data/processed/y_train.csv`, `data/processed/X_test.csv`, `data/processed/y_test.csv`

2. train - обучение классификатора
   - вход: `data/processed/*.csv`
   - выход: `experiments/log_reg.sav`, `experiments/scaler.pkl`

Запуск pipeline:
```bash
dvc repro
```

---

# API

Простой FastAPI сервис с одним эндпоинтом для предсказания.

## Эндпоинты

- `/health` -- проверка состояния сервиса
- `/predict` -- предсказание класса опухоли
- `/train` -- обучение модели
- `/evaluate` -- оценка качества модели

## /predict

POST запрос с параметрами: [tests/test_0.json](tests/test_0.json)

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @tests/test_0.json
```

Ответ:
```json
{
  "prediction": 0,
  "model": "LOG_REG"
}
```

## /train

POST запрос для обучения модели данными из Cassandra.

```bash
curl -X POST "http://localhost:8000/train"
```

Ответ:
```json
{
  "model": "LOG_REG",
  "trained": true,
  "train_samples": 455,
  "message": "Model LOG_REG trained successfully on 455 samples"
}
```

## /evaluate

POST запрос для оценки качества модели на тестовых данных из Cassandra.

```bash
curl -X POST "http://localhost:8000/evaluate"
```

Ответ:
```json
{
  "model": "LOG_REG",
  "accuracy": 0.9737,
  "precision": 0.9722,
  "recall": 0.9722,
  "f1": 0.9722,
  "eval_samples": 114,
  "message": "Evaluation complete on 114 samples"
}
```

---

# Хранение данных

Train и eval данные загружаются в Cassandra при старте контейнера (через `cassandra-init.sh`) в таблицы `train_data` и `eval_data`. Ручки `/train` и `/evaluate` читают данные оттуда.

---

# Установка и запуск

## Создание виртуального окружения

```bash
python -m venv .venv
source .venv/bin/activate
```

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Запуск DVC pipeline

```bash
dvc repro
```

## Запуск API

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI: http://localhost:8000/docs

## Запуск через Docker

```bash
docker-compose up --build
```

При старте Cassandra автоматически загружает train/eval данные в таблицы `train_data` / `eval_data`.

## Тестирование

```bash
pytest tests/
```
