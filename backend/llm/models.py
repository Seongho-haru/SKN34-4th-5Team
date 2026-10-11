import uuid

from django.conf import settings
from django.db import models
from django.db.models import Q
from pgvector.django import VectorField, HnswIndex


class GuestChatUsage(models.Model):
    """Legacy v1 usage records, retained for history only; v2 uses usage wallets."""
    identity = models.CharField(max_length=64, primary_key=True)
    used = models.PositiveSmallIntegerField(default=0)


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


class ChatAttachment(models.Model):
    """비공개 파일/명시적 URL. 소유권은 세션과 같고 원본은 공용 private object storage 에 둔다."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="attachments")
    kind = models.CharField(max_length=5, choices=[("image", "image"), ("text", "text"), ("url", "url")])
    name = models.CharField(max_length=255)
    object_key = models.CharField(max_length=255, blank=True)
    content_type = models.CharField(max_length=64, blank=True)
    size = models.PositiveIntegerField(default=0)
    width = models.PositiveIntegerField(null=True)
    height = models.PositiveIntegerField(null=True)
    source_url = models.URLField(max_length=2048, blank=True)
    extracted_text = models.TextField(blank=True)  # URL 캐시만. 파일은 private storage 에서 읽는다
    url_read_result = models.JSONField(default=dict, blank=True)  # 실패/부분 관찰 보존; 새 첨부만 재조회
    created_at = models.DateTimeField(auto_now_add=True)


class ChatAttachmentDeletion(models.Model):
    object_key = models.CharField(max_length=255, primary_key=True)
    created_at = models.DateTimeField(auto_now_add=True)


class ChatThreadDeletion(models.Model):
    """삭제된 ChatSession 의 checkpoint 삭제 outbox. 세션 삭제와 같은 트랜잭션에 쓰고, checkpoint 삭제 성공 뒤 지운다.

    FK 가 아니다: 세션 행은 이미 없다. 남아 있는 행 = 아직 지우지 못한 thread (llm.service.chat_thread.purge_deleted_threads).
    """
    thread_id = models.UUIDField(primary_key=True)
    token = models.UUIDField(default=uuid.uuid4)  # 예약마다 새 값. drain 은 읽은 token 일 때만 행을 지운다
    created_at = models.DateTimeField(auto_now_add=True)


class AnswerFeedback(models.Model):
    """소유 세션의 완료 답변 평가. 편집 전 스냅샷은 유지하되 세션/회원 삭제 시 함께 삭제한다."""
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="feedback")
    answer_id = models.CharField(max_length=255)  # checkpoint AIMessage.id (legacy IDs included)
    message_id = models.PositiveBigIntegerField()  # public wire ID; never reused within the session
    rating = models.CharField(max_length=4, choices=[("up", "좋아요"), ("down", "아쉬워요")])
    reason = models.CharField(max_length=24, blank=True)
    comment = models.CharField(max_length=1000, blank=True)
    question = models.TextField()
    answer = models.TextField()
    metadata = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["session", "answer_id"], name="feedback_session_answer_unique"),
            models.CheckConstraint(condition=Q(rating__in=["up", "down"]), name="feedback_valid_rating"),
        ]
        ordering = ["-updated_at", "-id"]


class UsageWallet(models.Model):
    """토큰 사용량 지갑. 회원(user) 또는 비회원(guest = ChatSession.guest UUID) 중 하나만 주인이다.

    period: 회원은 USAGE_TIMEZONE 기준 달("2026-09"), 비회원은 "lifetime"(충전 없음). 달이 바뀌면 used 를 0 으로 되돌린다.
    """
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True,
                                related_name="usage_wallet")
    guest = models.UUIDField(null=True, blank=True, unique=True)
    period = models.CharField(max_length=16)
    used_tokens = models.BigIntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.CheckConstraint(
                condition=Q(user__isnull=False, guest__isnull=True) | Q(user__isnull=True, guest__isnull=False),
                name="usage_wallet_has_one_owner",
            ),
        ]


class UsageCharge(models.Model):
    """턴 하나의 후불 원장. reserved는 지갑 active guard, unknown_calls는 미확인 사용량."""
    RESERVED, SETTLED, SETTLED_UNKNOWN = "reserved", "settled", "settled_unknown"
    wallet = models.ForeignKey(UsageWallet, on_delete=models.CASCADE, related_name="charges")
    session_id = models.UUIDField(null=True)  # FK 아님: 세션이 지워져도 원장은 남는다
    period = models.CharField(max_length=16)
    status = models.CharField(max_length=16, default=RESERVED)
    reserved_tokens = models.BigIntegerField()
    charged_tokens = models.BigIntegerField(default=0)
    input_tokens = models.BigIntegerField(default=0)
    output_tokens = models.BigIntegerField(default=0)
    calls = models.IntegerField(default=0)
    unknown_calls = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    settled_at = models.DateTimeField(null=True)

    class Meta:
        indexes = [models.Index(fields=["wallet", "status"], name="usage_charge_wallet_status")]
