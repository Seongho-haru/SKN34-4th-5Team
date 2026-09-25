from rest_framework import serializers
from llm.models import ChatMessage, ChatToolCall
import json
class ChatToolCallSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatToolCall
        fields = (
            "id",
            "tool_name",
            "status",
            "created_at",
        )
        read_only_fields = fields
class ChatMessageSerializer(serializers.ModelSerializer):
    content = serializers.CharField(
        source="message",
        max_length=2200,
    )

    tools = ChatToolCallSerializer(
        many=True,
        read_only=True,
    )

    class Meta:
        model = ChatMessage
        fields = (
            "id",
            "sequence_no",
            "role",
            "content",
            "status",
            "tools",
            "created_at",
            "updated_at",
        )

        read_only_fields = (
            "id",
            "sequence_no",
            "role",
            "status",
            "tools",
            "created_at",
            "updated_at",
        )


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"