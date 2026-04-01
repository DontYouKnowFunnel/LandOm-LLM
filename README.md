# LandOm Funnel Pipeline

HTML을 입력으로 받아 퍼널 분석 결과(JSON)와 시각화 오버레이 JS를 생성하는 프로젝트입니다.

## 구성

- `funnel_pipeline/run_funnel_langgraph.py`
  - HTML -> compressed HTML -> LLM 분석 -> `funnel.json` 생성
- `funnel_pipeline/run_funnel_selector_mapping.py`
  - `funnel.json`의 id를 CSS selector로 매핑
  - `funnel_selector_output.json`, `funnel_overlay.js` 생성
- `prompts/html_to_funnel_prompt.txt`
  - 퍼널 분석 프롬프트
- `examples/input/input.html`
  - 예시 입력 HTML
- `run/*`
  - 실행 결과 출력 파일

## 설치

```bash
python3 -m pip install -r requirements.txt
```

## 환경 변수

루트에 `.env` 파일을 만들고 최소 아래 값을 설정하세요.

```env
OPENAI_API_KEY=your_openai_api_key
```

Groq를 사용할 때는 아래 키를 설정하세요.

```env
GROQ_API_KEY=your_groq_api_key
```

선택적으로 공통 LLM 설정도 `.env`에 둘 수 있습니다.

```env
LLM_PROVIDER=groq
LLM_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
LLM_BASE_URL=https://api.groq.com/openai/v1
```

선택(LangSmith 추적):

```env
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=projectname
```

## 사용 방법

### 1) 퍼널 JSON 생성 (LangGraph)

```bash
python3 funnel_pipeline/run_funnel_langgraph.py \
  --input-html examples/input/input.html \
  --output run/funnel.json
```

Groq로 실행하려면:

```bash
python3 funnel_pipeline/run_funnel_langgraph.py \
  --provider groq \
  --model meta-llama/llama-4-scout-17b-16e-instruct \
  --input-html examples/input/input.html \
  --output run/funnel.json
```

### 2) selector 매핑 + 오버레이 JS 생성

```bash
python3 funnel_pipeline/run_funnel_selector_mapping.py \
  --input-html examples/input/input.html \
  --funnel-json run/funnel.json \
  --output-json run/funnel_selector_output.json \
  --output-js run/funnel_overlay.js
```

## 브라우저 오버레이 확인

1. 대상 페이지를 브라우저에서 연다.
2. `run/funnel_overlay.js` 내용을 콘솔에 붙여넣어 실행한다.
3. 퍼널 박스가 섹션 위에 렌더링된다.

## 출력 파일

- `run/funnel.json`: LLM 퍼널 분류 결과
- `run/funnel_selector_output.json`: selector 매핑 결과
- `run/funnel_overlay.js`: 브라우저 주입용 시각화 스크립트
- `examples/output/output.txt`: compressed HTML 예시
