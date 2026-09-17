from pathlib import Path

# --- Dosya Yolları ---
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customer_subscription_churn_usage_patterns.csv"

# --- Veri Ön İşleme Parametreleri ---
DROP_COLUMNS = ["user_id", "signup_date"]
CATEGORICAL_COLUMNS = ["plan_type"]
TARGET_COLUMN = "churn"

# --- Veri Bölme (Split) ---
TEST_SIZE = 0.20
RANDOM_SEED = 42

# --- DataLoader Parametreleri ---
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 64

# --- Model Mimarisi & Eğitim Parametreleri ---
INPUT_DIM = 8
HIDDEN_DIM_1 = 64
HIDDEN_DIM_2 = 32
DROPOUT_1 = 0.3
DROPOUT_2 = 0.2
OUTPUT_DIM = 1

LEARNING_RATE = 0.005
EPOCHS = 35

# --- Eşik Değerleri ---
DEFAULT_THRESHOLD = 0.50
OPTIMAL_THRESHOLD = 0.40