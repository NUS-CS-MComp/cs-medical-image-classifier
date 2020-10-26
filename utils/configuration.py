import pathlib

from utils.gpu_control import assign_gpu

WORKSPACE_DIR = pathlib.Path(__file__).parent.parent

# Data related configuration
DATA_DIR = WORKSPACE_DIR / "data"
ORIGIN_TRAIN_DATA_DIR = DATA_DIR / "train_image/train_image"
ORIGIN_TEST_DATA_DIR = DATA_DIR / "test_image/test_image"
TRAIN_LABEL_DATA_PATH = DATA_DIR / "train_label.csv"
TRAIN_DATA_DIR = DATA_DIR / "train_image/regrouped"
TEST_DATA_DIR = DATA_DIR / "test_image/test_image"

# Preprocessing related configuration
CONCURRENT_PROCESSING_THRESHOLD = 4

# Training related configuration
BATCH_SIZE = 32
SEED = 137
IMAGE_SIZE = (224, 224)
IMAGE_SHAPE = IMAGE_SIZE + (3,)
TRAIN_EPOCH = 100
FINE_TUNE_EPOCH = 50
VALIDATION_SPLIT = 0.2
MODEL_CHECKPOINT_PATH = WORKSPACE_DIR / "models/checkpoints"

# GPU related configuration
GPU_MEMORY_LIMIT = 1024 * 4
assign_gpu(memory_limit=GPU_MEMORY_LIMIT)
