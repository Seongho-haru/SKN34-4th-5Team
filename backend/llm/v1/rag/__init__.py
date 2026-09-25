"""KBO 직관 안내 RAG — 도메인별 분리 구조.

    from llm.v1.rag import answer                    # 또는 from llm.v1.rag.pipeline import answer, rag_chain
    answer("잠실 주차 얼마야?", history=[...], stadium_name="잠실야구장")
    → {"answer": "...", "sources": [...], "route": "club>rag:JAMSIL:TRANSPORT"}

구조·규칙은 README.md 참고.
"""
from .dispatcher import route  # noqa: F401
from .pipeline import answer, chat_chain, normalize_history, rag_chain, use_rag  # noqa: F401
