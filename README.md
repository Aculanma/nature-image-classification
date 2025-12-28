## Описание проекта

Данный проект посвящён задаче классификации изображений природных сцен.
Целью является обучение модели машинного обучения, способной автоматически
определять класс изображения, соответствующий типу природного ландшафта 
(например: лес, горы, море и т.д.).  

Проект выполнен в рамках курса по MLOps.

В рамках проекта реализованы:
- подготовка и разбиение данных,
- обучение модели,
- управление параметрами через YAML-конфигурации,
- хранение данных и результатов обучения с помощью DVC.

## Структура проекта

nature-image-classification/  
├── configs/ # Конфигурации экспериментов  
│ ├── baseline.yaml  
│ └── train.yaml  
├── nature_image_classification/ # Исходный код проекта  
│ ├── init.py  
│ ├── data_split.py  
│ └── train.py  
├── outputs.dvc # DVC-файл с результатами обучения  
├── main.py # Точка входа для запуска обучения  
├── pyproject.toml # Зависимости и метаданные проекта  
├── README.md  
├── .dvcignore  
└── .pre-commit-config.yaml  

## Setup

Опишем процедуру запуска проекта:

### 1.Клонирование репозитория

```bash
git clone https://github.com/Aculanma/nature-image-classification.git
cd nature-image-classification
```

### 2. Установка uv

Если uv еще не установлен:
```bash
pip install uv
```

### 3. Создаим виртуальное окружение
```bash
uv venv
.venv\Scripts\activate      # Windows
#или
source .venv/bin/activate   # должно сработать для Linux / macOS
```

### 4. Установка зависимостей

```bash
uv sync
```

### 5. Настройка dvc
В проекте используется DVC для хранения данных и артефактов обучения.
Удаленное хранилище настраивается локально.
Пример настройки для Yandex Object Storage:
```bash
# Добавляем удалённое хранилище по умолчанию
dvc remote add -d storage s3://mlops-first-steps
# Указываем URL эндпоинта
dvc remote modify storage endpointurl https://storage.yandexcloud.net
```

DVC будет использовать AWS-конфиг или переменные окружения для авторизации.
Пример локального AWS-конфига:
```bash
[default]
aws_access_key_id = <ВАШ_КЛЮЧ>
aws_secret_access_key = <ВАШ_СЕКРЕТ>
```
Или просто через переменные окружения:
```bash
export AWS_ACCESS_KEY_ID=<ВАШ_КЛЮЧ>
export AWS_SECRET_ACCESS_KEY=<ВАШ_СЕКРЕТ>
```
### 6. Загрузка данных
```bash
dvc pull
```
После команды все данные должны появиться локально.

## Train  
В этом разделе описан процесс обучения модели.  
__Этапы обучения:__
1. загрузку и разбиение датасета,
2. предобработку данных,
3. обучение модели,
4. сохранение результатов обучения через dvc.

Для запуска обучения используем Hydra и скрипт train.py из папки nature_image_classification:
1. Обучение baseline модели:
```bash
python nature_image_classification/train.py model.type=baseline
```
2. Обучение ResNet50 модели:
```bash
python nature_image_classification/train.py model.type=resnet50
```
После завершения обучения метрики сохраняются в формате .json в папку outputs.

