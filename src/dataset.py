"""
1. AI-Hub 감성대화 말뭉치 로딩 및 라벨 전처리
2. PyTorch Dataset 클래스
"""
import json

import pandas as pd
import torch
from torch.utils.data import Dataset

from config import AIHUB_CORPUS_PATH, EMOTION_PREFIX_MAP, EMOTION_TO_POLARITY, MAX_LEN


def load_aihub_corpus(path=AIHUB_CORPUS_PATH) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        dialogues = json.load(f)

    samples = []
    for d in dialogues:
        try:
            emotion = d["profile"]["emotion"]["type"]
            utterances = d["talk"]["content"]
            for key, sentence in utterances.items():
                if key.startswith("HS"):  # 사용자 발화만 사용
                    sentence = sentence.strip()
                    if sentence:
                        samples.append({"sentence": sentence, "emotion": emotion})
        except (KeyError, TypeError):
            continue

    return pd.DataFrame(samples)


def map_to_polarity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _map_emotion(code: str):
        return EMOTION_PREFIX_MAP.get(code[:2])

    out["emotion_name"] = out["emotion"].apply(_map_emotion)
    out["label"] = out["emotion_name"].map(EMOTION_TO_POLARITY)
    return out.dropna(subset=["label"])


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item
