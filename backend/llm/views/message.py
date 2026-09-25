from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.generics import GenericAPIView
from rest_framework import mixins
from llm.models import ChatMessage
from llm.serializer.message import ChatMessageSerializer
from llm.service.chat import ChatMessageService

class ChatMessageShowView(
    mixins.ListModelMixin,      # GET       요청 : 채팅방 리스트전달
    GenericAPIView
    ):
    """채팅방 메세지 리스트조회 및, 수정, 삭제 기능.

    URL: /api/v2/chat/sessions/<session_id>/messages/

    GET: 해당 대화방의 저장된 메시지를 조회합니다.
    
    """
    # IsAuthenticated 회원 사용자만 조회 할수있습니다.
    permission_classes = [IsAuthenticated]
    serializer_class = ChatMessageSerializer

    # 채팅방 대화 목록을 가져옵니다.
    def get_queryset(self):
        # 채팅방 세션메세지
        session_id = self.kwargs["session_id"]
        return ChatMessage.objects.filter(
            session_id = session_id,
            session__user=self.request.user,
        ).order_by("sequence_no")

    # GET: /api/v2/chat/sessions/<session_id>/messages/
    def get(self, request, *args, **kwargs):
        """해당 세션의 채팅목록을 가져온다."""
        return self.list(request, *args, **kwargs)

class ChatMessageView(
    GenericAPIView
):
    """채팅을 비회원 / 회원 둘다 동시에 할수있도록 처리한다.
    URL: /api/v2/chat/sessions/<session_id>/messages/

    POST: 사용자 메시지를 보내고 AI 답변을 생성합니다.
            Accept: application/json이면 JSON, text/event-stream이면 SSE로 응답합니다.
    """
    permission_classes = [AllowAny]
    serializer_class = ChatMessageSerializer

    # POST: /api/v2/chat/sessions/<session_id>/messages/
    def post(self, request, *args, **kwargs):
        """LLM 호출"""
        return ChatMessageService.stream(request, *args, **kwargs)

    def put(self, request, *args, **kwargs):
        """이후 채팅목록을 수정 + LLM 호출"""
        return ChatMessageService.message_update(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):
        """이후 채팅목록을 삭제한다."""
        return ChatMessageService.message_delete(request, *args, **kwargs)