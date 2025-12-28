from pathlib import Path
import random
import shutil

RANDOM_SEED = 42
VAL_RATIO = 0.1


def split_train_val(train_dir: Path, val_dir: Path) -> None:
    random.seed(RANDOM_SEED)

    for class_dir in train_dir.iterdir():
        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob("*.jpg"))
        random.shuffle(images)

        val_size = int(len(images) * VAL_RATIO)
        val_images = images[:val_size]

        (val_dir / class_dir.name).mkdir(parents=True, exist_ok=True)

        for img in val_images:
            shutil.move(img, val_dir / class_dir.name / img.name)


if __name__ == "__main__":
    split_train_val(
        train_dir=Path("data/raw/train"),
        val_dir=Path("data/raw/val"),
    )
