import uuid

from django.conf import settings
from django.db import models
from pgvector.django import VectorField, HnswIndex


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
    id = models.UUIDField(
        primary_key=True,default=uuid.uuid4,editable=False
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="chat_sessions",
    )
    title = models.CharField(max_length=255, blank=True, default="메세지 제목")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


# 채팅 메세지 테이블
class ChatMessage(models.Model):
    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    sequence_no = models.IntegerField()
    role = models.CharField(
        max_length=10, choices=[("human", "사용자"), ("ai", "AI")], default="human"
    )
    message = models.TextField()
    status = models.CharField(
        max_length=10,
        choices=[("completed", "완료"), ("stopped", "중단")],
        blank=True,
        default="",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class ChatTurn(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="turns")
    question = models.TextField()
    base_sequence = models.PositiveIntegerField(default=0)
    status = models.CharField(
        max_length=10,
        choices=[
            ("pending", "대기"),
            ("completed", "완료"),
            ("stopped", "중단"),
            ("failed", "실패"),
        ],
        default="pending",
    )
    human_message = models.OneToOneField(
        ChatMessage, on_delete=models.SET_NULL, null=True, blank=True, related_name="+"
    )
    assistant_message = models.OneToOneField(
        ChatMessage, on_delete=models.SET_NULL, null=True, blank=True, related_name="+"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class ChatProgressEvent(models.Model):
    turn = models.ForeignKey(ChatTurn, on_delete=models.CASCADE, related_name="progress_events")
    sequence_no = models.PositiveIntegerField()
    operation_id = models.UUIDField()
    parent_operation_id = models.UUIDField(null=True, blank=True)
    kind = models.CharField(
        max_length=10,
        choices=[("phase", "처리"), ("retrieval", "검색"), ("tool", "도구")],
    )
    status = models.CharField(
        max_length=12,
        choices=[
            ("started", "시작"),
            ("completed", "완료"),
            ("failed", "실패"),
            ("interrupted", "중단"),
            ("unknown", "확인 불가"),
        ],
    )
    label = models.CharField(max_length=160)
    tool_name = models.CharField(max_length=80, null=True, blank=True)
    tool_call_id = models.CharField(max_length=255, null=True, blank=True)
    arguments = models.JSONField(null=True, blank=True)
    result = models.JSONField(null=True, blank=True)
    truncated = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("sequence_no",)
        constraints = [
            models.UniqueConstraint(
                fields=("turn", "sequence_no"), name="unique_chat_progress_sequence"
            )
        ]
