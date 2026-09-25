from accounts.models import CustomUser
from llm.models import ChatSession
from django.db.models import QuerySet


class Sessions:
    """
    채팅방(세션) CRUD
    1. 채팅방 가져오기 
        - 페이지 제너레이션으로 가져와야함 
    2. 채팅방 생성하기 
        - 반환값은 UUID
        - 비회원 / 회원 나눠서 설정할수있도록
    3. 채팅방 변경하기
        - 보통 채팅방 제목을 수정하는것에대해서 사용하게 됨
    4. 채팅방 삭제하기 
    """

    @staticmethod
    def get_sessions(user :CustomUser) ->QuerySet[ChatSession]:
        """
            무한 스크롤이 가능하도록 해야함 
        """
        
        
        return 

    def create_sesstions():
        pass

    def edit_sesstions():
        pass

    def delete_sesstions():
        pass