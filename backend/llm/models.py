import uuid

from django.conf import settings
from django.db import models
from django.db.models import Q
from pgvector.django import VectorField, HnswIndex
from llm.enum import ToolStatus , ChatRole , MessageStatus

class Document(models.Model):
    title = models.CharField(max_length=255)
    source = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)


class DocumentChunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    content = models.TextField()
    chunk_index = models.IntegerField()
    metadata = models.JSONField(default=dict)
    embedding = VectorField(dimensions=1536)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            HnswIndex(
                name="chunk_embedding_hnsw",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]


# 채팅방 테이블
class ChatSession(models.Model):
    """
    NOTE: 9월 25일 비회원 로직추가 
    1. User에 Null, Black을 추가함
    2. Guest을 추가
    3. 제약조건: 둘중하나만 반드시 존재해야함.  
    """
    id = models.UUIDField(
        primary_key=True,default=uuid.uuid4,editable=False
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="chat_sessions",
        null=True,
        blank=True
    )
    guest = models.UUIDField(
        null= True,
        blank=True,
        db_index=True
    )
    title = models.CharField(max_length=255, blank=True, default="메세지 제목")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    class Meta:
        constraints = [
            models.CheckConstraint(
                # 제약조건 회원(User) or 비회원(guest) 중 하나는 반드시 존재야한다. 라는 제약조건
                condition=( 
                    Q(user__isnull=False, guest__isnull=True)
                    | Q(user__isnull=True, guest__isnull=False)
                ),
                name="chat_session_has_one_owner",
            )
        ]


# 채팅 메세지 테이블
class ChatMessage(models.Model):
    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    sequence_no = models.PositiveIntegerField()
    role = models.CharField(
        max_length=10,
        choices=ChatRole.choices,
    )
    message = models.TextField()
    status = models.CharField(
        max_length=12,
        choices=MessageStatus.choices,
        default=MessageStatus.COMPLETED,
    )
    is_active = models.BooleanField(
        default=True,
        db_index=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["session", "sequence_no"],
                condition=Q(is_active=True),
                name="unique_active_message_sequence",
            )
        ]

        indexes = [
            models.Index(
                fields=["session", "is_active", "sequence_no"],
            )
        ]


class ChatToolCall(models.Model):
    message = models.ForeignKey(
        ChatMessage,
        on_delete=models.CASCADE,
        related_name="tools",
    )

    tool_name = models.CharField(max_length=100)

    status = models.CharField(
        max_length=12,
        choices=ToolStatus.choices,
        default=ToolStatus.STARTED,
    )

    arguments = models.JSONField(
        null=True,
        blank=True,
    )

    result = models.JSONField(
        null=True,
        blank=True,
    )

    truncated = models.BooleanField(
        default=False,
    )

    # Tool 호출 시작 시간
    created_at = models.DateTimeField(
        auto_now_add=True,
    )

    # Tool 실행 종료 시간
    finished_at = models.DateTimeField(
        null=True,
        blank=True,
    )