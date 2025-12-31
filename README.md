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
Данные хранятся в публичном Object Storage (Yandex Cloud).  
Настройка удалённого хранилища:
```bash
# Добавляем удалённое хранилище 
dvc remote add -d storage https://storage.yandexcloud.net/mlops-first-steps
```
После этого можно загрузить данные локально:
```bash
dvc pull
```
DVC автоматически восстановит структуру данных в папке data/ (включая train, test и val) в соответствии с .dvc-файлами проекта.

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

