import pathlib

WORKSPACE_DIR = pathlib.Path(__file__).parent.parent

# Data related configuration
DATA_DIR = WORKSPACE_DIR / "data"
ORIGIN_TRAIN_DATA_DIR = DATA_DIR / "train_image/train_image"
ORIGIN_TEST_DATA_DIR = DATA_DIR / "test_image/test_image"
TRAIN_LABEL_DATA_PATH = DATA_DIR / "train_label.csv"
TRAIN_DATA_DIR = DATA_DIR / "train_image/processed"
TEST_DATA_DIR = DATA_DIR / "test_image/processed"

# Preprocessing related configuration
CONCURRENT_PROCESSING_THRESHOLD = 4
