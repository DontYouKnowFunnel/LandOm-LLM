# LandOm LLM Server

랜딩페이지 퍼널 분석과 섹션별 개선안 생성을 수행하는 LLM 서버입니다.

## Runtime API

### 퍼널 분석

`POST /api/v1/funnels/analyze`

전체 랜딩페이지 HTML을 입력받아 섹션별 퍼널 단계와 selector를 백엔드 callback으로 전송합니다.

### 개선안 생성

`POST /api/v1/funnels/optimize`

백엔드가 전달한 특정 퍼널 단계의 섹션 HTML, 서비스 페르소나, 방문자 행동 로그를 사용해 개선안을 생성하고, 결과를 백엔드 callback으로 저장합니다.

개선안 생성 파이프라인은 아래 순서로 동작합니다.

1. 섹션 HTML 전처리
2. LLM 기반 구조화 특징 추출
3. Problem RAG 검색 및 field-aware reranking
4. Revision RAG 검색 및 reranking
5. 검색 근거 기반 개선안 생성
6. `PATCH /api/v1/projects/{projectId}/optimizations/{sectionId}` callback

## 주요 구조

- `app/server.py`
  - FastAPI endpoint와 백엔드 callback 처리
- `app/pipeline.py`
  - 기존 퍼널 분석 runtime
- `rag_pipeline/`
  - 개선안 생성을 위한 runtime RAG 파이프라인
- `data/rag/`
  - Qdrant 업로드용 Problem DB / Revision DB JSONL
- `scripts/build_rag_vector_db.py`
  - JSONL을 embedding하여 Qdrant collection에 업로드
- `scripts/query_rag_vector_db.py`
  - Qdrant 검색 smoke test

수집, 평가, grid search 등 연구용 일회성 스크립트는 runtime branch에 포함하지 않습니다.

## 설치

```bash
python3 -m pip install -r requirements.txt
```

## 환경 변수

루트에 `.env` 파일을 만들고 `.env.example`을 기준으로 값을 채웁니다.

필수 값:

```env
OPENAI_API_KEY=your_openai_api_key
BACKEND_BASE_URL=https://backend.example.com
```

원격 Qdrant 서버를 사용할 경우:

```env
QDRANT_URL=https://your-qdrant.example.com
QDRANT_API_KEY=your_qdrant_api_key
```

`QDRANT_URL`을 비워두면 local embedded Qdrant가 `runs/qdrant`를 사용합니다.
런타임 collection 이름은 `problem_patterns_en`, `intervention_evidence_en`으로 고정되어 있습니다.
RAG 파이프라인 모델 설정은 `rag_pipeline/config.py`의 `AI_MODELS`에서, 퍼널 분석 모델 설정은 `funnel_pipeline/config.py`의 `AI_MODELS`에서 관리합니다.

## Vector DB 업로드

```bash
python3 scripts/build_rag_vector_db.py --recreate
```

원격 Qdrant는 `.env`의 `QDRANT_URL`, `QDRANT_API_KEY`를 사용하거나 CLI 인자로 직접 지정할 수 있습니다.

```bash
python3 scripts/build_rag_vector_db.py \
  --qdrant-url https://your-qdrant.example.com \
  --qdrant-api-key your_qdrant_api_key \
  --recreate
```

## 로컬 서버 실행

```bash
uvicorn app.server:app --reload
```
