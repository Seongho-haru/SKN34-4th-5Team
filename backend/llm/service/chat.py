from chat_service import ChatService

class ChatMessageService:
    """
    V2 채팅 메시지 비즈니스 로직
    """
    version='2'
    # chain = 랭체인 오는곳
    @staticmethod
    def stream(self, session, content):
        """새 질문 전송"""
        pass

    @staticmethod
    def message_update(self, session, message_id, content):
        """메세지 + langchain 실행"""
        pass

    @staticmethod
    def message_delete(self):
        """메세지 삭제 처리"""
        pass

    def stop(self, session, message_id):
        """LLM 중단"""
        pass