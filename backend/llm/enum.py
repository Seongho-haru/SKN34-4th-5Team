from django.db import models


class ChatRole(models.TextChoices):
    USER = "user", "사용자"
    ASSISTANT = "assistant", "AI"


class MessageStatus(models.TextChoices):
    PENDING = "pending", "진행 중"
    COMPLETED = "completed", "완료"
    FAILED = "failed", "실패"
    STOPPED = "stopped", "중단"


class ToolStatus(models.TextChoices):
    STARTED = "started", "시작"
    COMPLETED = "completed", "완료"
    FAILED = "failed", "실패"