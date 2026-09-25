# llm/v1/rag — KBO 직관 안내 RAG (도메인 분리 구조)

2026-09-15 · 담당: club·course = 형준, venue = 현준

## 구조

```
llm/
├─ chat_service.py       성호 채팅 서비스. `self.chain = chat_chain() or self.get_chain()` 한 줄로 RAG 가 붙는다
├─ rag_views.py          POST /chat/ 뷰 (curl·Postman 확인용. 실사용 경로는 chat/sessions/…)
└─ rag/
   ├─ pipeline.py        ★ 백엔드 진입점. answer() + 성호 chain 규격을 만족하는 RagChatChain
   ├─ dispatcher.py      안내데스크. 질문을 보고 course / club / venue / 둘 다 / 범위 밖 으로 나눔 (LLM 0회)
   ├─ persona.py         말투 규칙 + 후처리(soften/clean). ★ 두 도메인이 같이 쓰는 유일한 공통 파일
   ├─ club/              구단·야구: 순위·일정·가격·예매·좌석·반입·재입장·규칙   — 형준만 수정
   │    agent.py         진입점 answer() · 가드 · 검색 조합 · LangChain ChatOpenAI 호출 · 후처리
   │    router.py        구장·팀 별칭, 카테고리 키워드 (rag_test/router.py 와 동일)
   │    retrieval.py     임베딩 · pgvector 검색 · 키워드 재정렬
   │    structured.py    순위·일정 DB 직접 조회 (LLM 0회)
   │    prompts.py       내용 규칙 · few-shot · 고정 문구
   ├─ venue/             구장 안팎: 먹거리·시설·교통·포토존·주변  — 현준만 수정
   │    agent.py         test.py 이식본 (LangChain create_agent + search_documents_tool · 검색어 변환 · 안/밖 필터 · 키워드 fallback)
   │    prompts.py       내용 규칙 · 검색어 변환 프롬프트 (말투는 persona 에서 가져옴)
   └─ course/            ★ 직관 코스 추천 (메인 기능) — 형준만 수정
        agent.py         진입점 answer() · 경기 조회 → 후보 검색 → LLM 1회(키만 고름) → places[] 조립
        prompts.py       코스 프롬프트 (JSON 출력 규칙. LLM 은 키 고르기 + 인트로만)
        slots.py         취향 · 동행(아이·부모님·연인·혼자·회식·친구) · 여유시간 · 재추천 (LLM 0회)
        timeline.py      경기 시작에서 역산한 도착·출발 시각, 경기 종료 예상, 체류시간 (LLM 0회)
        geo.py           하버사인 거리 · 8방위 · 총 도보 · 동선 나쁘면 가까운 후보로 교체 (LLM 0회)
        save.py          places[] → POST /courses/ payload (travel.Course/CourseStop 계약)
```

## 챗봇에 어떻게 붙어 있나

성호 `ChatService` 는 `self.chain` 에 두 가지만 요구한다.

```python
self.chain.invoke({"question": q, "chat_history": messages}) -> str
self.chain.stream({"question": q, "chat_history": messages}) -> str 청크들
```

`pipeline.chat_chain()` 이 그 규격을 그대로 만족하는 Runnable 을 돌려주므로, chat_service.py 변경은
import 1줄 + chain 고르는 1줄이 전부다. 회원(`/chat/sessions/{id}/messages/`)·게스트(`/chat/guest/`)
두 경로 모두 여기로 들어온다.

- 스위치 없음 (2026-09-15): 챗봇은 항상 `assistant` 파이프라인(프롬프트 · RAG · 에이전트[야구 DB 읽기 전용 도구] · 파서)으로 답한다. 테스트 러너 안에서만 `chat_chain()` 이 None
- Django 테스트 DB(`test_*`) 로 도는 동안은 자동으로 꺼진다 (기존 회귀 테스트 보호)
- RAG 는 답을 한 번에 만들므로 `.stream()` 은 완성된 답을 잘라서 흘린다 → SSE 계약·프론트 그대로
- `places` · `coursePayload` 처럼 문자열 말고 전부가 필요하면 `chain.detail(...)` 또는 `pipeline.answer(...)`

## 두 도메인이 지키는 약속 (이것만 맞으면 안은 자유)

```python
READY: bool                                   # False 면 디스패처가 이 도메인으로 안 보냄
def answer(question: str, history: list[dict] | None, hint_stadium: str | None) -> dict:
    return {"answer": str, "sources": list[dict], "route": str}
```

- `history`: `[{"role": "user"|"assistant", "content": str}, ...]` 이번 질문 제외, 오래된 것부터
- `hint_stadium`: 프론트에서 고른 구장 코드(`JAMSIL` 등) 또는 None
- `sources[i]`: `{"doc_id", "grade", "category", "stadium", ...}` — grade 는 OFFICIAL / UNCERTAIN / UNOFFICIAL / THIRD_PARTY

## 연결 (백엔드 담당)

프론트 계약(`docs/handoffs/FRONTEND_BACKEND_HANDOFF.md` 7장: `{messages, context}` → `{reply}`)을 구현한 뷰가 `llm/v1/rag_views.py` 에 있다. `config/urls.py` 에 두 줄:

```python
from llm.v1.rag_views import ChatView
    path("chat/", ChatView.as_view()),     # urlpatterns 안, chat/sessions/ 보다 위나 아래 상관없음
```

기존 `chat_service.py`(채팅방 저장 API)에 붙이고 싶으면 RAG 는 이 한 줄이면 된다.

```python
from llm.v1.rag.pipeline import answer          # from llm.v1.rag import answer 와 같은 함수

result = answer(question,                        # 마지막 user 메시지
                history=[{"role": "user"|"assistant", "content": "..."}, ...],   # 이번 질문 제외
                stadium_name=context.get("stadium"))  # "잠실야구장" 같은 값, 없으면 None
reply = result["answer"]                          # 프론트 reply 에 그대로
# result["sources"] (근거 목록) · result["route"] (어느 길로 갔는지, 디버깅용) 는 선택
```

- 인증: `ChatView` 는 JWT 없이 받는다 (Next 서버 간 중계용. 기존 `chat/sessions/…` 는 JWT 그대로).
- 프론트 설정: `frontend/.env.local` 에 `CHAT_PROVIDER=backend`, `CHAT_BACKEND_URL=http://backend:8000/api/v1/chat/` → 프론트 컨테이너 재시작.
- 환경변수: `OPENAI_API_KEY`, `EMBEDDING_MODEL`(적재 때와 동일) 은 compose 에 이미 있음. `LLM_MODEL` 은 없어도 기본값 gpt-5.6-luna.
- 패키지: `requirements.txt` 에 `langchain` 추가됨(venue 의 create_agent) → `docker compose up -d --build backend` 로 재빌드 필요.
- 예외: `answer()` 는 도메인 내부 예외를 잡아 사용자용 문구를 돌려주므로 뷰에서 500 이 나지 않는다. 네트워크·키 문제는 로그로 확인.

## 질문이 흐르는 길

```
프론트 /chat-api → Next 서버 → Django 뷰(백엔드 담당)
   → rag.answer(question, history, stadium_name)
   → dispatcher.route()   "scope" | "venue" | "club" | "both"
       scope : 야구 무관 → 고정 문구
       venue : venue.READY 면 venue.answer, 아니면 club.answer
       club  : club.answer
       both  : 둘 다 부르고 이어 붙임 (LLM 최대 2회)
   → persona.finalize()   말투 통일 후처리
   → {"reply": ...}
```

## 규칙

1. **자기 폴더만 고친다.** club/ 은 형준, venue/ 는 현준. 상대 폴더는 PR 리뷰로만.
2. 말투를 바꾸고 싶으면 `persona.py` — 두 도메인에 동시에 적용된다. 바꾸기 전에 팀 채널에 한 줄.
3. 디스패처 키워드(`VENUE_WORDS`·`CLUB_WORDS`)는 "어느 도메인이냐"만 정한다. 카테고리 세부 사전은 각자 폴더 안에.
4. LLM 은 `langchain_openai.ChatOpenAI` (팀 기준). 모델은 `LLM_MODEL` 환경변수 (기본 gpt-5.6-luna). club 은 체인 1회 호출, venue 는 `create_agent`(도구 호출 판단 + 검색어 변환 + 답변 = 2~3회) — 기법은 각자 유지하고 골든셋으로 비교한다.
5. 임베딩 모델은 `EMBEDDING_MODEL` — 적재(build_index) 때와 같아야 한다.

## 로컬 실행 (뷰 없이 패키지만 확인)

```bash
# 1. DB 에 청크가 있어야 한다 (3,839건). 없으면:
docker compose exec backend python manage.py build_index

# 2. Django shell 에서 바로 호출
docker compose exec backend python manage.py shell
>>> from llm.v1.rag import answer
>>> answer("삼성 몇 위야?")["answer"]                       # 순위: DB 직접조회, LLM 0회
>>> r = answer("잠실 주차 얼마야?"); r["route"], r["answer"]  # RAG: 임베딩 1회 + LLM 1회
>>> answer("재입장은?", history=[{"role":"user","content":"잠실 주차 얼마야?"},{"role":"assistant","content":"..."}])["route"]
```

`route` 값으로 어느 길로 갔는지 바로 알 수 있다 (`structured` = 순위·일정 직접조회, `guard:clarify` = 되묻기, `venue(not ready)>club>…` = venue 준비 전이라 club 이 대신 답함).

## 평가

`rag_test/`(step4·step5)는 그대로 채점 도구로 쓴다. club 로직을 여기서 고치면 `rag_test/`의 같은 파일(router·structured·chain)도 맞춰 둔다 — 지금은 수동 동기화(4차에서 rag_test 가 이 패키지를 import 하도록 정리 예정).
