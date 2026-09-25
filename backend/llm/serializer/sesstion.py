from datetime import timedelta

from drf_spectacular.utils import extend_schema_field
from rest_framework import serializers
from ..models import ChatMessage, ChatSession
from django.utils import timezone

class ChatSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatSession
        fields = ("id", "title", "created_at", "updated_at")
        read_only_fields = ("id", "created_at", "updated_at")


class ChatMessageSerializer(serializers.ModelSerializer):
    content = serializers.CharField(source="message", max_length=2200)

    class Meta:
        model = ChatMessage
        fields = ("id", "sequence_no", "role", "content", "status", "created_at", "updated_at")
        read_only_fields = ("id", "sequence_no", "role", "status", "created_at", "updated_at")