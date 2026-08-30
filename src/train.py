"""
KoBERT 감정(긍/부정) 분류 모델 학습 스크립트.

데이터: AI-Hub 감성대화 말뭉치(2022) 중 사용자 발화(HS)
라벨: 기쁨(E6) → positive, 나머지 5개 감정 → negative
파라미터: epoch=3, batch_size=16, lr=5e-5, optimizer=AdamW 

실행:
    python src/train.py
"""
import pandas as pd
import torch
from kobert_transformers import get_tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import BertModel

from config import (
    BASE_MODEL_NAME, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    MODEL_DIR, MODEL_PATH, RANDOM_SEED, SAMPLES_PER_CLASS, TEST_SIZE,
)
from dataset import SentimentDataset, load_aihub_corpus, map_to_polarity
from model import KoBERTClassifier


def prepare_data():
    """말뭉치 로딩 -> 긍/부정 매핑 -> 클래스별 균등 샘플링 -> train/val 분리."""
    df = load_aihub_corpus()
    df = map_to_polarity(df)

    df_sample = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(n=SAMPLES_PER_CLASS, random_state=RANDOM_SEED))
        .reset_index(drop=True)
    )
    print("클래스별 샘플 수:\n", df_sample["label"].value_counts())

    texts = df_sample["sentence"].tolist()
    le = LabelEncoder()
    labels = le.fit_transform(df_sample["label"].tolist())

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    return train_texts, val_texts, train_labels, val_labels, le


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_texts, val_texts, train_labels, val_labels, le = prepare_data()

    tokenizer = get_tokenizer()
    base_model = BertModel.from_pretrained(BASE_MODEL_NAME)
    model = KoBERTClassifier(base_model).to(device)

    train_loader = DataLoader(
        SentimentDataset(train_texts, train_labels, tokenizer),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        SentimentDataset(val_texts, val_labels, tokenizer),
        batch_size=BATCH_SIZE,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, _ = model(input_ids, attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {avg_loss:.4f}")

        # 검증
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                _, logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(f"Validation Accuracy: {correct / total:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n모델 저장 완료: {MODEL_PATH}")


if __name__ == "__main__":
    train()
