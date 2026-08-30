"""
학습된 KoBERT 감정 분류 모델(models/kobert_model_ver6.pt)을 불러와
리뷰 텍스트의 긍/부정을 예측하는 스크립트.

실행:
    python src/predict.py
"""
import pandas as pd
import torch
import torch.nn.functional as F
from kobert_transformers import get_tokenizer
from transformers import BertModel

from config import BASE_MODEL_NAME, MAX_LEN, MODEL_PATH, REVIEWS_PATH
from model import KoBERTClassifier

# label encoding 시 'negative' < 'positive' 알파벳 순으로 0, 1이 부여됨 (train.py의 LabelEncoder 기준)
ID2LABEL = {0: "negative", 1: "positive"}


def load_model(model_path=MODEL_PATH, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = BertModel.from_pretrained(BASE_MODEL_NAME)
    model = KoBERTClassifier(base_model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def predict_sentiment(text: str, model, tokenizer, device) -> str:
    """텍스트 하나의 감정 라벨(positive/negative)을 반환"""
    encoding = tokenizer(
        text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        _, logits = model(input_ids, attention_mask)
        pred = torch.argmax(logits, dim=1).item()
    return ID2LABEL[pred]


def predict_sentiment_scores(text: str, model, tokenizer, device) -> dict:
    """텍스트 하나에 대해 {'부정': p0, '긍정': p1} 확률 딕셔너리를 반환"""
    encoding = tokenizer(
        text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        _, logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    return dict(zip(["부정", "긍정"], probs))


def run_on_reviews(reviews_path=REVIEWS_PATH):
    """리뷰 CSV 전체에 감정 예측을 적용해 결과를 반환"""
    model, device = load_model()
    tokenizer = get_tokenizer()

    reviews = pd.read_csv(reviews_path)
    reviews["sentiment"] = reviews["content"].apply(
        lambda t: predict_sentiment_scores(t, model, tokenizer, device)
    )
    reviews[["부정", "긍정"]] = reviews["sentiment"].apply(pd.Series)
    reviews = reviews.drop(columns="sentiment")
    return reviews


if __name__ == "__main__":
    model, device = load_model()
    tokenizer = get_tokenizer()

    print(predict_sentiment("오늘 너무 기분 좋아!", model, tokenizer, device))  # positive
    print(predict_sentiment("짜증나고 우울하다.", model, tokenizer, device))    # negative
    print(predict_sentiment_scores("민원", model, tokenizer, device))
