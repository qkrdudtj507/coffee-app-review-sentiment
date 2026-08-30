from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

#경로
AIHUB_CORPUS_PATH = ROOT_DIR / "data" / "감성대화말뭉치(최종데이터)_Training.json"
REVIEWS_PATH = ROOT_DIR / "data" / "reviews_merged_all.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "kobert_model_ver6.pt"

#모델
BASE_MODEL_NAME = "monologg/kobert"
NUM_LABELS = 2
MAX_LEN = 128

#감정코드 매핑
EMOTION_PREFIX_MAP = {
    "E1": "분노",
    "E2": "슬픔",
    "E3": "불안",
    "E4": "상처",
    "E5": "당황",
    "E6": "기쁨",
}

EMOTION_TO_POLARITY = {
    "기쁨": "positive",
    "분노": "negative",
    "슬픔": "negative",
    "불안": "negative",
    "당황": "negative",
    "상처": "negative",
}

#학습 파라미터 설정
SAMPLES_PER_CLASS = 10000  
RANDOM_SEED = 42
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
