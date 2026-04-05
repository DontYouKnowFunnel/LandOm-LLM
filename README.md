# LandOm Funnel Pipeline

랜딩페이지 HTML을 입력으로 받아, 페이지를 세그멘팅하고 각 세그먼트에 퍼널 단계를 부여하는 프로젝트입니다.

## 현재 메인 파이프라인

현재 고정된 파이프라인은 아래 순서로 동작합니다.

1. HTML 본문 추출 및 정제
2. page-level 세그멘테이션
3. 세그먼트 압축 표현 생성
4. 공통 프롬프트 기반 LLM 분류
5. 결과 정규화 및 selector 기반 출력 생성

평가 기준은 DOM unit 기반의 두 지표를 사용합니다.

- `Boundary F1`: 세그먼트 경계 일치도
- `Label Accuracy`: DOM unit 단위 퍼널 단계 일치도

## 주요 구성 요소

- `html_tools/segments.py`
  - 기본 세그멘테이션과 세그먼트 압축 유틸리티
- `html_tools/segments_targeted_refine.py`
  - 메인 세그멘테이션 경로
- `prompts/html_to_funnel_prompt.txt`
  - llama와 OpenAI가 공통으로 사용하는 메인 프롬프트
- `scripts/run_bench_batch.py`
  - provider와 model을 받아 benchmark 추론을 수행하는 메인 실행 스크립트
- `funnel_pipeline/run_funnel_langgraph.py`
  - 공통 LLM 호출, 출력 정규화 유틸리티

## 설치

```bash
python3 -m pip install -r requirements.txt
```

## 환경 변수

루트에 `.env` 파일을 만들고 필요한 API 키를 설정합니다.

```env
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

선택적으로 공통 LLM 설정을 둘 수 있습니다.

```env
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
LLM_BASE_URL=https://api.groq.com/openai/v1
```

## 실행 예시

### llama benchmark 실행

```bash
python3 scripts/run_bench_batch.py \
  --provider groq \
  --model llama-3.3-70b-versatile \
  --input-dir LandOm-LLM-Bench/input \
  --gold-dir LandOm-LLM-Bench/output \
  --output-root run/bench_eval
```

### OpenAI benchmark 실행

```bash
python3 scripts/run_bench_batch.py \
  --provider openai \
  --model gpt-5.4 \
  --input-dir LandOm-LLM-Bench/input \
  --gold-dir LandOm-LLM-Bench/output \
  --output-root run/bench_eval
```

평가는 benchmark workspace 내부의 공식 grader가 수행합니다.
