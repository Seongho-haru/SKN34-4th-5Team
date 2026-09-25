from typing import Sequence

from django.db import transaction
from django.db.models import Max
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ..models import ChatSession, ChatMessage


class DjangoChatMessageHistory(BaseChatMessageHistory):

    def __init__(self, user_id: int, session_id: int):
        self.user_id = user_id
        self.session_id = session_id

    def _get_session(self):
        return ChatSession.objects.get(
            id=self.session_id,
            user_id=self.user_id,
        )

    @property
    def messages(self) -> list[BaseMessage]:
        session = self._get_session()

        records = (
            ChatMessage.objects
            .filter(session=session)
            .order_by("id")
        )

        return [
            (HumanMessage if record.role == "human" else AIMessage)(content=record.message)
            for record in records
        ]

    @transaction.atomic
    def add_messages(
        self,
        messages: Sequence[BaseMessage],
        *,
        assistant_status: str = "completed",
    ) -> list[ChatMessage]:

        session = ChatSession.objects.select_for_update().get(
            id=self.session_id, user_id=self.user_id
        )
        last_sequence = session.messages.aggregate(last=Max("sequence_no"))["last"] or 0

        rows = [
            ChatMessage(
                session=session,
                sequence_no=last_sequence + index,
                role=message.type,
                message=message.content,
                status=assistant_status if message.type == "ai" else "",
            )
            for index, message in enumerate(messages, start=1)
        ]

        return ChatMessage.objects.bulk_create(rows)

    def clear(self) -> None:
        session = self._get_session()

        ChatMessage.objects.filter(
            session=session
        ).delete()
