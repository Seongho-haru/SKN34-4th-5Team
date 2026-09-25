from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.views import APIView
from rest_framework.request import Request
from rest_framework.pagination import PageNumberPagination
from rest_framework.generics import GenericAPIView
from rest_framework import mixins
from llm.models import ChatSession

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
    serializer_class = ChatSession
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

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


    # POST /api/v2/chat/sessions/
    def post(self, request : Request, *args, **kwargs):
        # 1. 만약 회원 사용자일시 
        if request.user.is_authenticated:
            return self.create(request, *args, **kwargs)
        # 2. 만약 비회왼이라면
        # 3. 쿠키에서 Guest_id 를 찾는다.
        guest_id = request.COOKIES.get("guest_id")
        # 4. 쿠기에 guest_id가없으면 생성한다.
        if not guest_id:
            guest_id = str(uuid.uuid4())
        

        
    
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
    serializer_class = ChatSession

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


class ChatMessageView(APIView):
    """대화방 메시지 조회 및 AI 답변 생성 API.

    URL: /api/v2/chat/sessions/<session_id>/messages/

    GET: 해당 대화방의 저장된 메시지를 조회합니다.
    POST: 사용자 메시지를 보내고 AI 답변을 생성합니다.
         Accept: application/json이면 JSON, text/event-stream이면 SSE로 응답합니다.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        pass

    def post(self, request, *args, **kwargs):
        pass