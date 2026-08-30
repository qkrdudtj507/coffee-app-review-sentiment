"""
KoBERT 기반 감정(긍/부정) 이진 분류 모델.
사전학습 KoBERT에 선형 분류 헤드를 추가한 구조입니다.
"""
from torch import nn


class KoBERTClassifier(nn.Module):
    def __init__(self, base_model, num_labels: int = 2):
        super().__init__()
        self.bert = base_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return loss, logits
