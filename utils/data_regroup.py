import pathlib
import shutil

import pandas as pd

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
ORIGIN_IMAGE_DIR = DATA_DIR / "train_image/train_image"
TARGET_IMAGE_DIR = DATA_DIR / "train_image/regrouped"
LABEL_DATA_DIR = DATA_DIR / "train_label.csv"

label_df = pd.read_csv(LABEL_DATA_DIR)
label_df.set_index("ID", inplace=True)

for file in ORIGIN_IMAGE_DIR.iterdir():
    if file.is_file():
        label = label_df.loc[int(file.stem), "Label"]
        new_path = TARGET_IMAGE_DIR / str(label)
        new_path.resolve().mkdir(parents=True, exist_ok=True)
        if not (new_path / file.name).exists():
            shutil.copyfile(file, new_path / file.name)
