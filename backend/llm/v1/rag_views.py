"""프론트 챗봇 계약 — POST /chat/  (Next 서버가 CHAT_PROVIDER=backend 일 때 여기로 중계)

계약 (docs/handoffs/FRONTEND_BACKEND_HANDOFF.md 7장 · frontend/lib/chat/types.ts):
  요청  {"messages": [{"role": "user"|"assistant", "content": "..."}], "sessionId"?: int,
         "context": {"stadium"?: "잠실야구장", "intent"?: "route"|"baseball"|"stadium"}}
  응답  {"reply": "...", "sources": [...], "route": "..."}   ← 프론트는 reply 만 읽는다

기존 채팅방 API(chat/sessions/…, JWT 필요)는 그대로 두고, 이 뷰는 프론트 중계용으로 인증 없이 받는다.
대화 기록은 프론트가 messages 로 넘겨주므로 여기서는 저장하지 않는다 (계정별 저장은 4차).
"""
import logging

from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .rag import answer as rag_answer

log = logging.getLogger(__name__)

MAX_HISTORY = 12          # 프론트 MAX_HISTORY_MESSAGES 와 동일
MAX_MESSAGE_LENGTH = 2000
MAX_REPLY_LENGTH = 8000


class ChatView(APIView):
    permission_classes = (AllowAny,)
    authentication_classes = ()          # JWT 미적용 — Next 서버 간 중계

    def post(self, request):
        messages = request.data.get("messages")
        if not isinstance(messages, list) or not messages:
            return Response({"detail": "messages 가 비어 있습니다."}, status=status.HTTP_400_BAD_REQUEST)
        messages = messages[-MAX_HISTORY:]
        for m in messages:
            if not isinstance(m, dict) or m.get("role") not in ("user", "assistant") or not isinstance(m.get("content"), str):
                return Response({"detail": "messages 형식이 올바르지 않습니다."}, status=status.HTTP_400_BAD_REQUEST)
        last = messages[-1]
        if last["role"] != "user":
            return Response({"detail": "마지막 메시지는 user 여야 합니다."}, status=status.HTTP_400_BAD_REQUEST)
        question = last["content"].strip()
        if not question or len(question) > MAX_MESSAGE_LENGTH:
            return Response({"detail": f"질문은 1~{MAX_MESSAGE_LENGTH}자여야 합니다."}, status=status.HTTP_400_BAD_REQUEST)

        context = request.data.get("context") or {}
        stadium_name = context.get("stadium") if isinstance(context, dict) else None
        history = [{"role": m["role"], "content": m["content"]} for m in messages[:-1]]

        try:
            result = rag_answer(question, history=history, stadium_name=stadium_name)
        except Exception:
            log.exception("chat failed")
            return Response({"detail": "답변 생성에 실패했습니다. 다시 시도해 주세요."},
                            status=status.HTTP_502_BAD_GATEWAY)

        return Response({
            "reply": result["answer"][:MAX_REPLY_LENGTH],
            "sources": result.get("sources", []),
            "route": result.get("route", ""),
            "sessionId": request.data.get("sessionId"),
        })
