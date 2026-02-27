
안녕하세요 부울경 특화 2반 여러분! 
오늘부터 시작한 AI 특강 강의의 실습 과제 제출에 대해 안내 드립니다. 
AI 특강 강의 (화 ~ 금)를 다 보고, 해당 스켈레톤 과제를 돌려 본 후 
로그 내용을 캡쳐해서 첨부한 ppt를 참고해서 올려주세요! 
오늘 하루도 고생 많으셨습니다! :)

- 과제 제출 기한 : 2월 27일(금) 자정까지
- 스켈레톤 프로젝트 필수 제출
- 제출 포멧 : 프로젝트번호_학번.zip 제출

1. SSAFY GIT -> 프로젝트 개발 -> 특화 AI공용명세서 확인 후 스켈레톤 코드 (1~4) 실행
2. 스켈레톤 코드 url :  https://lab.ssafy.com/instruction/14th/skeleton-projects/ai-special-skeleton
3. 실행 결과를 test 후 각 스켈레톤 코드의 로그 캡쳐
4. 프로젝트번호_학번.zip 제출 








# Skeleton01 - Knowledge Distillation을 위한 LLM 기반 Pseudo Labeling 데이터 생성

## 1. 프로젝트 개요

Teacher 모델(Upstage Solar)을 API로 호출하여 정답 데이터가 없는 데이터셋에 Pseudo Label을 생성합니다. 생성된 데이터는 Student 모델(소형 LLM) 학습에 활용되며, 이를 통해 Knowledge Distillation의 핵심 개념을 실습합니다.

## 2. 실행 환경

| 항목 | 사양 |
|------|------|
| OS | Windows 10/11 |
| Python | 3.13 |
| RAM | 최소 8GB |
| 저장공간 | 최소 2GB |
| 주요 패키지 | openai, pandas, dotenv, jupyter |
| Teacher 모델 | Upstage Solar (API) |

## 3. 핵심 학습 내용

### 3.1 Solar API 설정
- Upstage Console에서 Solar API 키 발급 및 환경변수 설정
- openai 클라이언트를 통한 Solar 모델 호출 방법 학습

### 3.2 데이터 불러오기 및 확인
- pandas를 이용하여 기존 데이터셋 로드
- 정답 레이블이 없는 unlabeled 데이터 특성 파악

### 3.3 프롬프트 엔지니어링
- Teacher 모델이 고품질 Pseudo Label을 생성하도록 프롬프트 설계
- System / User 역할 분리 및 출력 형식 유도

### 3.4 LLM 기반 Pseudo Labeling
- Solar API를 호출하여 데이터 전체에 Pseudo Label 자동 생성
- 생성된 레이블을 데이터셋에 병합하여 학습 데이터 구성

## 4. 학습 키워드

- **Knowledge Distillation**: 대형 Teacher 모델의 지식을 소형 Student 모델로 이식하는 기법
- **Pseudo Labeling**: 정답이 없는 데이터에 Teacher 모델이 생성한 예측값을 정답으로 사용하는 방법
- **Prompt Engineering**: LLM이 원하는 형식과 품질의 출력을 생성하도록 입력을 설계하는 기술
- **Solar API**: Upstage에서 제공하는 고성능 LLM API 서비스

---

# Skeleton02 - Text-to-SQL Task에 대한 Knowledge Distillation 데이터셋 통한 LoRA 학습

## 1. 프로젝트 개요

Skeleton01에서 생성한 Pseudo Labeling 데이터셋을 활용하여 SmolLM2 소형 모델에 Text-to-SQL 태스크 특화 LoRA(Low-Rank Adaptation) 파인튜닝을 수행합니다. 학습 후 SQL 생성 정확도를 실행 기반 평가(Execution Accuracy)로 측정합니다.

## 2. 실행 환경

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA GeForce RTX 4050 Laptop (6GB VRAM) |
| OS | Windows 10/11 |
| Python | 3.13 |
| CUDA | 12.9.1 |
| 주요 패키지 | peft, transformers, trl, datasets, accelerate |
| 모델 | HuggingFaceTB/SmolLM2-360M-Instruct |

## 3. 핵심 학습 내용

### 3.1 데이터셋 EDA 및 전처리
- Spider 데이터셋 로드 및 구조 파악
- Text-to-SQL 입력 형식(스키마 + 자연어 질의 → SQL)에 맞게 전처리

### 3.2 Base 모델 및 토크나이저 로딩
- Hugging Face Hub에서 SmolLM2-360M-Instruct 모델 다운로드
- Chat template 기반 프롬프트 포맷팅

### 3.3 LoRA Fine-tuning
- PEFT 라이브러리의 LoraConfig 설정 (rank, alpha, target modules 등)
- TRL SFTTrainer를 활용한 효율적인 학습 파이프라인 구성
- `outputs/checkpoint-100/`에 adapter 가중치 저장

### 3.4 모델 평가
- `inference.py`로 학습된 LoRA 모델 추론 → `predict.txt` 생성
- test-suite-sql-eval 기반 Execution Accuracy 평가

## 4. 평가 결과 (기본 베이스라인)

```
                     easy                 medium               all
count                146                  106                  252
=====================   EXECUTION ACCURACY     =====================
execution            0.164                0.038                0.111
```

## 5. 학습 키워드

- **LoRA (Low-Rank Adaptation)**: 원본 모델 파라미터를 고정하고 소수의 추가 파라미터만 학습하여 효율적으로 파인튜닝하는 기법
- **Text-to-SQL**: 자연어 질의를 SQL 쿼리로 변환하는 NLP 태스크
- **EDA (Exploratory Data Analysis)**: 학습 전 데이터의 분포와 특성을 탐색하는 과정
- **Execution Accuracy**: 생성된 SQL 쿼리를 실제 DB에서 실행하여 결과가 일치하는지 평가하는 지표

---

# Skeleton03 - QLoRA로 모델을 양자화해서 학습하기

## 1. 프로젝트 개요

4-bit 양자화(Quantization)와 LoRA를 결합한 QLoRA 기법을 적용하여, 제한된 GPU 메모리(6GB)에서도 대형 모델에 준하는 성능의 파인튜닝을 수행합니다. bitsandbytes를 활용하여 모델을 NF4 4-bit로 로드하고 Text-to-SQL 태스크에 QLoRA 학습을 진행합니다.

## 2. 실행 환경

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA GeForce RTX 4050 Laptop (6GB VRAM) |
| OS | Windows 10/11 |
| Python | 3.13 |
| CUDA | 12.9.1 |
| 주요 패키지 | peft, transformers, trl, bitsandbytes, accelerate |
| 모델 | HuggingFaceTB/SmolLM2-360M-Instruct |

## 3. 핵심 학습 내용

### 3.1 4-bit 양자화 모델 로딩
- BitsAndBytesConfig를 통한 NF4 4-bit 양자화 설정
- fp16 대비 메모리 사용량 약 4배 절감 원리 이해

### 3.2 QLoRA Fine-tuning
- 양자화된 모델에 LoRA adapter를 결합한 QLoRA 학습 파이프라인 구성
- prepare_model_for_kbit_training을 통한 양자화 모델 학습 준비
- Gradient Checkpointing을 활용한 메모리 최적화

### 3.3 모델 평가
- `inference.py`로 QLoRA 학습 모델 추론 → `predict.txt` 생성
- test-suite-sql-eval 기반 Execution Accuracy 평가

## 4. 평가 결과 (Base 모델 기준)

```
                     easy                 medium               all                  joint_all
count                44                   16                   60                   1
=====================   EXECUTION ACCURACY     =====================
execution            0.682                0.250                0.567
```

## 5. 학습 키워드

- **Quantization (양자화)**: 모델 가중치의 자료형을 fp16 → 4-bit로 줄여 메모리를 절감하는 기법
- **QLoRA**: 4-bit 양자화된 모델에 LoRA를 적용하여 성능 하락을 최소화하면서 파인튜닝하는 기법
- **bitsandbytes**: GPU에서 4-bit/8-bit 양자화 연산을 지원하는 라이브러리
- **NF4 (NormalFloat4)**: QLoRA에서 사용하는 정보 손실을 최소화한 4-bit 부동소수점 자료형

---

# Skeleton04 - vLLM을 이용한 효율적인 LLM 추론과 성능 벤치마킹

## 1. 프로젝트 개요

vLLM을 활용하여 LLM 추론 파이프라인을 구축하고, Transformers 대비 성능 우위를 정량적으로 비교 분석하였습니다. Skeleton02에서 학습한 LoRA adapter를 vLLM에 연동하여 실제 서빙 환경에서의 추론까지 수행하였습니다.

## 2. 실행 환경

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA GeForce RTX 4050 Laptop (6GB VRAM) |
| OS | Windows 11 + WSL2 Ubuntu |
| Python | 3.12 |
| vLLM | 0.6.6 |
| PyTorch | 2.5.1+cu124 |
| 모델 | HuggingFaceTB/SmolLM2-360M-Instruct |

## 3. 핵심 학습 내용

### 3.1 PagedAttention 시뮬레이션
- vLLM의 핵심 메모리 관리 기술인 PagedAttention의 동작 원리를 Python 코드로 시뮬레이션
- Logical Block → Physical Block 매핑을 통한 KV Cache 메모리 효율화 원리 이해
- Block Table을 활용한 가상 메모리 주소 변환 과정 구현

### 3.2 vLLM 모델 로딩 및 추론
- vLLM LLM 엔진 초기화 파라미터 학습 (tensor_parallel_size, gpu_memory_utilization, enforce_eager 등)
- SamplingParams를 활용한 추론 파라미터 설정
- Chat template 기반 프롬프트 포맷팅 및 텍스트 생성

### 3.3 LoRA Adapter 연동 추론
- Skeleton02에서 학습한 LoRA adapter를 vLLM Runtime LoRA로 로딩
- LoRARequest를 통한 동적 adapter 적용
- Text-to-SQL 태스크에 대한 LoRA 적용 추론 수행

### 3.4 Transformers vs vLLM 성능 비교
- 동일 모델·동일 프롬프트 조건에서 Transformers와 vLLM의 추론 성능을 정량 비교
- 추론 시간, 처리량(Throughput), 메모리 사용량 등 벤치마크 수행

## 4. 제출 폴더 구조

```
ai-special-skeleton/
├── 2. skeleton02/
│   ├── outputs/                    # LoRA 학습 결과
│   │   └── checkpoint-100/         # LoRA adapter (adapter_config.json, adapter_model.safetensors)
│   └── ...
│
├── 4. skeleton04/
│   ├── vllm.ipynb                  # 실행 노트북 (출력 결과 포함)
│   ├── merged_model/               # Merge된 모델 (선택)
│   ├── checkpoint-100/             # 체크포인트
│   ├── README.md                   # 결과 정리 문서 (본 파일)
│   └── ...
```

## 5. 실행 방법

### 사전 준비 (WSL2 Ubuntu 환경)

```bash
# 1. 가상환경 활성화
source ~/vllm-env/bin/activate

# 2. 프로젝트 폴더로 이동
cd "/mnt/c/Users/SSAFY/Desktop/ai-special-skeleton/4. skeleton04"

# 3. Jupyter 서버 실행
jupyter notebook --no-browser

# 4. VSCode에서 WSL 모드로 열기 (Ctrl+Shift+P → Reopen Folder in WSL)
# 5. 커널을 vLLM (WSL)로 선택 후 노트북 순서대로 실행
```

### 주요 패키지

```bash
pip install vllm==0.6.6 transformers peft ipykernel notebook
```

## 6. 결과 요약

| 비교 항목 | Transformers | vLLM |
|-----------|-------------|------|
| 추론 방식 | Autoregressive (토큰 단위) | Continuous Batching + PagedAttention |
| 메모리 관리 | 정적 할당 | PagedAttention (동적 블록 할당) |
| KV Cache | 연속 메모리 필요 | 블록 단위 비연속 할당 가능 |
| 배치 처리 | Static Batching | Continuous Batching |
| 주요 장점 | 간편한 사용, 유연한 커스터마이징 | 높은 처리량, 효율적 메모리 사용 |

> 구체적인 벤치마크 수치는 vllm.ipynb 노트북의 실행 결과를 참고해주세요.

## 7. 학습 키워드

- **PagedAttention**: OS의 가상 메모리 페이징 기법을 KV Cache에 적용하여 메모리 낭비를 최소화하는 기술
- **KV Cache**: Transformer 추론 시 이전 토큰의 Key/Value를 캐싱하여 중복 연산을 방지하는 기법
- **vLLM**: PagedAttention 기반의 고성능 LLM 추론 엔진
- **LoRA Adapter**: 소수의 파라미터만 학습하여 LLM을 효율적으로 파인튜닝하는 기법
- **Performance Benchmarking**: 추론 속도, 메모리 사용량 등을 정량적으로 측정·비교하는 과정
