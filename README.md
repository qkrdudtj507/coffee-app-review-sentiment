# ☕ 커피 프랜차이즈 앱 리뷰 감정분석 모델 (KoBERT)

> 국내 커피 프랜차이즈 12개 브랜드의 앱스토어 리뷰 24,828건을 대상으로,
> KoBERT를 파인튜닝해 리뷰의 긍/부정 감정을 분류하는 모델입니다.
>
> 논문 [「AI Agent 기반 커피 프랜차이즈 앱스토어 리뷰 분석: 토픽 모델링 및 감정 분석을 통한 고객 경험(CX) 인사이트 도출」]의 **감정 분석 모델** 을 담당했습니다.
> 🏆 한국빅데이터학회 추계학술대회 **우수논문상**

[![Python](https://img.shields.io/badge/Python-3.x-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white)]()
[![KoBERT](https://img.shields.io/badge/KoBERT-monologg-lightgrey)]()

## 📌 담당 부분

이 저장소는 논문 전체가 아니라 **제가 직접 만든 감정 분석(Sentiment Analysis) 모델 부분만** 정리했습니다. 리뷰 수집, 토픽 모델링(LLM 기반 클러스터링)은 팀 프로젝트의 다른 파트이며, 이 저장소의 범위가 아닙니다.

- **입력**: 커피 프랜차이즈 앱 리뷰 텍스트 (24,828건)
- **출력**: 리뷰별 긍정/부정 감정 라벨 및 확률 점수
- **역할**: KoBERT 감정분석 모델 훈련 및 fine tunning, 개별 리뷰 및 토픽 모델링으로 분류된 CX 이슈 클러스터에 감정 점수를 결합해, 브랜드별·기능별 CX 개선 우선순위를 도출하는 데 사용됨

## 🔍 모델 구조 & 학습 방법론

### 1. 학습 데이터
- **AI-Hub 감성대화 말뭉치(2022)** 의 사용자 발화(`HS`)만 추출 (약 146,000 문장)
- 6개 세부 감정(E1 분노, E2 슬픔, E3 불안, E4 상처, E5 당황, E6 기쁨)을 이진 라벨로 재분류
  - `기쁨(E6)` → **positive**
  - 나머지 5개 → **negative**
- 클래스 불균형 해소를 위해 각 라벨에서 **10,000건씩** 균등 샘플링

### 2. 모델 구조
`monologg/kobert` 사전학습 모델의 `[CLS]` 토큰 임베딩에 선형 분류 헤드(`Linear`)를 추가한 이진 분류기입니다. ([`src/model.py`](src/model.py))

### 3. 학습 설정

| 파라미터 | 값 |
|---|---|
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 5e-5 |
| Optimizer | AdamW |
| Loss | CrossEntropyLoss |
| Max sequence length | 128 |
| Validation accuracy | **약 0.84** |

### 4. 적용
학습된 분류기를 실제 앱 리뷰 24,828건(`content` 컬럼)에 적용해 리뷰별 긍/부정 확률을 산출했습니다. 이 결과는 토픽 클러스터(10개)와 교차 분석되어, 브랜드별·기능 영역별 CX 감정 점수(예: `Technical and System Issues` 영역 평균 -0.8 이하)를 도출하는 데 사용되었습니다. 

## 📁 폴더 구조

```
coffee-app-review-sentiment/
├── README.md
├── .gitignore
├── requirements.txt
├── src/
│   ├── config.py       # 경로 · 라벨 매핑 · 하이퍼파라미터
│   ├── model.py         # KoBERTClassifier 정의
│   ├── dataset.py       # AI-Hub 말뭉치 로딩/전처리, PyTorch Dataset
│   ├── train.py          # 학습 스크립트
│   └── predict.py        # 학습된 모델로 리뷰 감정 예측
├── notebooks/
│   ├── train(original).ipynb        # 원본 학습 노트북 (참고용)
│   └── predict(original).ipynb    # 원본 추론 노트북 (참고용)
├── data/
│   └── README.md         # 데이터 다운로드/준비 안내 (용량·저작권상 원본 미포함)
├── models/
│   └── README.md         # 학습된 가중치 파일 안내 (용량상 원본 미포함)
└── reports/
    └── AI_Agent_기반_커피_프랜차이즈_앱스토어_리뷰_분석.pdf   # 논문 원문
```

`src/` 아래 스크립트들은 원본 노트북(`notebooks/`)의 로직을 그대로 기능 단위로 정리한 것입니다. 코드 자체는 동일합니다.

## ⚠️ 한계 및 향후 개선 방향

논문 Ⅴ장에서 밝힌 한계점입니다.

- 리뷰 데이터 특성상 부정 리뷰 비중이 높아, 부정 감성 점수가 과대평가됐을 가능성이 있습니다.
- 긍정/부정 이진 분류만 수행해, 중립 감정을 반영하지 못했습니다.
- 향후에는 중립 클래스를 추가한 다중 분류, 설문·소셜미디어 등 데이터 소스 확장을 고려할 수 있습니다.

## 📄 인용

```
AI Agent 기반 커피 프랜차이즈 앱스토어 리뷰 분석: 토픽 모델링 및 감정 분석을 통한 고객 경험(CX) 인사이트 도출
한국빅데이터학회지, 우수논문상 (한국빅데이터학회 추계학술대회)
```
