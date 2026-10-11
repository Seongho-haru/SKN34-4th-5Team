import logging
import uuid
from collections.abc import Mapping

from django.db import transaction
from django.db.models import Q
from django.http import Http404, HttpResponse
from rest_framework.exceptions import ValidationError
from rest_framework.parsers import JSONParser, MultiPartParser
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from community import image_storage
from llm.models import ChatAttachment, ChatSession
from llm.service.attachments import MAX_IMAGE, metadata, normalize_file, read_file, reference_url, reserve_attachment_deletion
from llm.service.chat_thread import ChatThread
from llm.service.ownership import get_owned_session
from llm.views.message import GuestChatThrottle
log = logging.getLogger(__name__)

TOOL_GROUP_LABELS = {
    "web_research": "웹 조사",
    "schedule": "경기 일정",
    "standings": "순위",
    "players": "선수 정보",
    "baseball_stats": "야구 기록",
    "rules": "규정",
    "stadium_info": "구장 정보",
    "carry_in": "반입 규정",
    "parking_transport": "주차·교통",
    "community": "커뮤니티",
    "nearby_places": "주변 장소",
    "tourism": "관광",
    "directions": "길찾기",
    "courses": "기존 코스",
    "weather": "날씨",
    "day_plan": "직관 코스 계획",
}


class ToolGroupsView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        return Response([{"id": key, "label": value} for key, value in TOOL_GROUP_LABELS.items()])


class ChatAttachmentView(APIView):
    permission_classes = [AllowAny]
    parser_classes = [JSONParser, MultiPartParser]
    throttle_classes = [GuestChatThrottle]

    def get(self, request, session_id):
        session = get_owned_session(request, session_id)
        return Response([metadata(row) for row in session.attachments.order_by("created_at")])

    def post(self, request, session_id):
        session = get_owned_session(request, session_id)
        is_file = request.content_type.startswith("multipart/form-data")
        if is_file:
            try:
                length = int(request.META["CONTENT_LENGTH"])
            except (KeyError, TypeError, ValueError):
                return Response({"detail": "Content-Length가 필요합니다."}, status=411)
            if length > MAX_IMAGE + 64 * 1024:
                return Response({"detail": "요청 본문이 너무 큽니다."}, status=413)
            if set(request.data) != {"file"} or len(request.FILES.getlist("file")) != 1:
                raise ValidationError({"file": "파일 하나만 전송해 주세요."})
            name, kind, body, mime, width, height = normalize_file(request.FILES["file"])
            values = dict(name=name, kind=kind, content_type=mime, size=len(body), width=width, height=height,
                          object_key=f"chat/{session.id}/{uuid.uuid4().hex}")
        else:
            if not isinstance(request.data, Mapping) or set(request.data) != {"url"} or not isinstance(request.data["url"], str):
                raise ValidationError({"url": "URL 하나만 전송해 주세요."})
            url = reference_url(request.data["url"])
            values = dict(kind="url", name=url[:255], source_url=url, content_type="text/uri-list", size=0)
        stored = False
        try:
            with transaction.atomic():
                if not ChatSession.objects.select_for_update().filter(pk=session.id).exists():
                    raise Http404
                if not is_file:
                    existing = session.attachments.filter(kind="url", source_url=url).filter(
                        Q(url_read_result={}) | ~Q(extracted_text="")).first()
                    if existing:
                        return Response(metadata(existing), status=201)
                if is_file:
                    image_storage.put_object(values["object_key"], body, mime)
                    stored = True
                row = ChatAttachment.objects.create(session=session, **values)
        except Exception as error:
            # Compensation must run after the failed metadata transaction has rolled back.
            if stored:
                try:
                    image_storage.delete_object(values["object_key"])
                except image_storage.StorageUnavailable:
                    reserve_attachment_deletion(values["object_key"])
                    return Response({"detail": "첨부 저장소를 사용할 수 없습니다."}, status=503)
            if isinstance(error, image_storage.StorageUnavailable):
                return Response({"detail": "첨부 저장소를 사용할 수 없습니다."}, status=503)
            raise
        return Response(metadata(row), status=201)


class ChatAttachmentDetailView(APIView):
    permission_classes = [AllowAny]

    def get(self, request, session_id, attachment_id):
        get_owned_session(request, session_id)
        row = ChatAttachment.objects.filter(session_id=session_id, pk=attachment_id).first()
        if row is None or row.kind == "url":
            raise Http404
        try:
            response = HttpResponse(read_file(row), content_type=row.content_type)
        except image_storage.ObjectNotFound:
            raise Http404
        except image_storage.StorageUnavailable:
            return Response({"detail": "첨부 저장소를 사용할 수 없습니다."}, status=503)
        response["Cache-Control"] = "private, no-store"
        response["X-Content-Type-Options"] = "nosniff"
        response["Content-Disposition"] = "inline" if row.kind == "image" else "attachment"
        return response

    def delete(self, request, session_id, attachment_id):
        session = get_owned_session(request, session_id)
        try:
            with transaction.atomic():
                ChatSession.objects.select_for_update().get(pk=session.pk)
                row = ChatAttachment.objects.filter(session=session, pk=attachment_id).first()
                if row is None:
                    raise Http404
                thread = ChatThread(session.id)
                # Historical checkpoints retain edit/retry references. Never delete their bytes.
                for snapshot in thread.history():
                    if any(str(row.id) in m.additional_kwargs.get("attachment_ids", [])
                           for m in snapshot.values.get("messages", [])):
                        return Response({"detail": "대화에 사용한 첨부는 삭제할 수 없습니다."}, status=409)
                row.delete()  # post_delete outbox cleans storage only after commit
        except image_storage.StorageUnavailable:
            return Response({"detail": "첨부 저장소를 사용할 수 없습니다."}, status=503)
        return Response(status=204)
