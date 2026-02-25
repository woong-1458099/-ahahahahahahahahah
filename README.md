
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
