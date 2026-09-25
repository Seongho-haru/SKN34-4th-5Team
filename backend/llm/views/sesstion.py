from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.request import Request
from rest_framework.pagination import PageNumberPagination
from rest_framework.generics import GenericAPIView
from rest_framework import mixins
from llm.models import ChatSession
from llm.serializer.sesstion import ChatSessionSerializer
import uuid

class ChatRoomListView(
    mixins.ListModelMixin,      # GET       요청 : 채팅방 리스트전달
    GenericAPIView
):
    """ 채팅방 리스트 조회 View
        URL: /api/v2/chat/sessions/

        GET: 로그인한 사용자의 대화방 목록을 조회합니다.
                return list_대화방
    """
    permission_classes = [IsAuthenticated]
    serializer_class = ChatSessionSerializer
    pagination_class = PageNumberPagination

    def get_queryset(self):
        return ChatSession.objects.filter(
            user=self.request.user
        ).order_by('-update_at')

    # GET /api/v2/chat/sessions/ : 로그인 한 사용자의 대화방 목록을 조회 합니다.
    def get(self, request : Request, *args, **kwargs):
        return self.list(request, *args, **kwargs)

class ChatRoomCreateView(
    mixins.CreateModelMixin,    # POST      요청 : 채팅방 생성,
    GenericAPIView
    ):
    """대화방 목록 생성 View.

    URL: /api/v2/chat/sessions/

    요구사항:
        1. 비회원 / 회원 채팅을 가능하도록한다. 
        2. 비회원은 UUID으로 임시 Id으로 설정한다.

    POST: 로그인한 사용자의 새 대화방을 생성합니다.
        return : UUID
    """
    # AllowAny : 비회원 + 회원 사용가능 권한 설정 
    permission_classes = [AllowAny]
    serializer_class = ChatSessionSerializer

    def perform_create(self, serializer):
        # 회원시
        if self.request.user.is_authenticated:
            serializer.save(
                user=self.request.user,
                guest=None
            )
        # 비회원시
        serializer.save(
                user=None,
                guest=self.guest_id
            )

    # POST /api/v2/chat/sessions/
    def post(self, request : Request, *args, **kwargs):
        # 1. 만약 회원 사용자일시 
        if request.user.is_authenticated:
            return self.create(request, *args, **kwargs)
        # 2. 만약 비회왼이라면
        # 3. 쿠키에서 Guest_id 를 찾는다.
        self.guest_id = request.COOKIES.get("guest_id")
        # 4. 쿠기에 guest_id가없으면 생성한다.
        if not self.guest_id:
            self.guest_id = str(uuid.uuid4())
        # 5. 비회원 전용 채팅방 생성
        response = self.create(request, *args, **kwargs)

        # 브라우저가 다음 요청에도 동일 guest를 식별하도록 저장
        response.set_cookie(
            "guest_id",
            str(self.guest_id),
            httponly=True,
            samesite="Lax",
        )

        return response
    
class ChatRoomDetailView(
    mixins.UpdateModelMixin,    # PATCH     요청 : 채팅방 수정(예: 체팅 제목)
    mixins.DestroyModelMixin,   # DELETE    요청 : 채팅방 삭제
    GenericAPIView
):
    """
    URL: /api/v2/chat/sessions/<session_id>/
    
    PATCH: 대화방 정보를 수정합니다.
    DELETE: 대화방을 삭제합니다.
    """

    permission_classes = [IsAuthenticated]
    serializer_class = ChatSessionSerializer

    def get_queryset(self):
        return ChatSession.objects.filter(
            user=self.request.user
        ).order_by('-update_at')

    def put(self, request, *args, **kwargs):
        return self.update(request, *args, **kwargs)
    
    def patch(self, request, *args, **kwargs):
        return self.partial_update(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)


