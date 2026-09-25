"""요청 한 건의 처리 단계·도구 실행을 수집해 SSE 와 DB 로 보내는 최소 계측.

수집기는 ContextVar 로 요청(그리고 그 요청의 생산자 스레드) 안에서만 살아 있다.
전역 agent 객체에 콜백을 영구 부착하지 않고, 호출마다 `config_kwargs()` 로 넘긴다.
"""
import contextvars
import json
import queue
import re
import threading
import uuid
from contextlib import contextmanager, nullcontext

from django.utils import timezone
from langchain_core.callbacks import BaseCallbackHandler

from ..models import ChatProgressEvent


MAX_EVENTS = 128
MAX_ARGUMENT_BYTES = 4096
MAX_RESULT_BYTES = 16384
MAX_TEXT = 200
MAX_DEPTH = 4
MAX_ITEMS = 20
# 공개해도 되는 값만 통과시킨다. 여기 없는 키(sql, schema, url, 원문 등)는 통째로 버린다.
SAFE_KEYS = {
    "team", "team_code", "stadium", "stadium_code", "date_from", "date_to",
    "start_date", "end_date", "status", "home_away", "limit", "as_of",
    "snapshot_date", "zone_keyword", "cheapest_first", "categories", "query",
    "kind", "keyword", "mode", "count", "name", "label", "warning",
    "distance", "seconds", "actual_date", "truncated", "count_is_partial",
    "games", "game_date", "game_time", "home_team", "away_team", "home_score",
    "away_score", "status_code", "standings", "rank", "team_name_ko", "wins",
    "losses", "draws", "games_behind", "prices", "zone_name_ko", "day_type",
    "customer_type", "price_tier", "price_krw", "discount_condition", "policies",
    "policy_type", "subtype", "max_tickets", "booking_channel", "channel_condition",
    "items", "type", "walk_min", "address",
}
RETRIEVAL_SAFE_KEYS = frozenset({"stadium", "stadium_code", "categories", "count", "truncated"})
TOOL_SAFE_KEYS = {
    "get_games": frozenset({
        "team", "team_code", "stadium", "stadium_code", "date_from", "date_to",
        "start_date", "end_date", "status", "home_away", "limit", "count",
        "actual_date", "truncated", "count_is_partial",
        "games", "game_date", "game_time", "home_team", "away_team", "home_score",
        "away_score", "status_code",
    }),
    "get_standings": frozenset({
        "team", "team_code", "as_of", "snapshot_date", "count", "standings", "rank",
        "team_name_ko", "wins", "losses", "draws", "games_behind",
    }),
    "get_ticket_prices": frozenset({
        "team", "team_code", "stadium_code", "zone_keyword", "cheapest_first", "count",
        "prices", "zone_name_ko", "day_type", "customer_type", "price_tier", "price_krw",
        "discount_condition",
    }),
    "get_ticket_policy": frozenset({
        "team", "team_code", "stadium_code", "status", "count", "policies", "policy_type",
        "subtype", "max_tickets", "booking_channel", "channel_condition",
    }),
    "get_ticket_policies": frozenset({
        "team", "team_code", "stadium_code", "status", "count", "policies", "policy_type",
        "subtype", "max_tickets", "booking_channel", "channel_condition",
    }),
    "search_kbo_documents": frozenset({"stadium_code", "categories", "count", "truncated"}),
    "search_nearby_places": frozenset({
        "stadium_code", "kind", "keyword", "count", "distance", "truncated", "items",
        "name", "type", "walk_min", "address",
    }),
    "search_places": frozenset({
        "stadium_code", "kind", "keyword", "categories", "count", "distance", "truncated",
        "items", "name", "type", "walk_min", "address",
    }),
    "get_directions": frozenset({"mode", "distance", "seconds", "truncated"}),
    "search_tourism": frozenset({"stadium_code", "categories", "count", "status", "truncated"}),
    "get_weather": frozenset({"stadium_code", "actual_date", "status", "label"}),
    "plan_course": frozenset({"stadium_code", "mode", "count", "distance", "seconds", "truncated"}),
    "get_baseball_schema": frozenset(),
    "execute_baseball_select": frozenset({"count", "truncated", "count_is_partial"}),
}
# 허용 키 안에 들어온 자유 문자열도 한 번 더 지운다.
SECRET = re.compile(
    r"sk-[A-Za-z0-9_\-]{8,}"
    r"|bearer\s+\S+"
    r"|(?:password|passwd|secret|token|api[_-]?key|credential)\s*[:=]\s*\S+"
    r"|\bselect\b[\s\S]*"
    r"|\b(?:insert|update|delete|drop)\b\s+\w+"
    r"|https?://\S+",
    re.IGNORECASE,
)
LABELS = {
    "retrieval": "관련 야구 안내 문서 확인 중",
    "model": "응답 생성 중",
    "get_games": "경기 일정 조회 중",
    "get_standings": "팀 순위 조회 중",
    "get_ticket_prices": "티켓 가격 조회 중",
    "get_ticket_policy": "예매 정책 조회 중",
    "get_ticket_policies": "예매 정책 조회 중",
    "search_kbo_documents": "야구 안내 문서 추가 조회 중",
    "search_nearby_places": "구장 주변 장소 조회 중",
    "search_places": "장소 조회 중",
    "get_directions": "이동 경로 조회 중",
    "search_tourism": "주변 관광지 조회 중",
    "get_weather": "경기 날씨 조회 중",
    "plan_course": "직관 코스 구성 중",
    "get_baseball_schema": "야구 데이터 항목 확인 중",
    "execute_baseball_select": "야구 데이터 조회 중",
    "assistant": "질문 처리 중",
}
CANCEL_LABEL = "응답 생성이 중단되었습니다"
_CURRENT = contextvars.ContextVar("chat_progress_collector", default=None)


class ProgressStorageError(RuntimeError):
    """진행 기록 저장 실패. 도메인 fallback 이 삼키지 않도록 따로 둔다."""


class ProgressCancelled(RuntimeError):
    """연결 종료 뒤 다음 모델·도구 실행을 시작하지 않기 위한 내부 제어 신호."""


def _text(value):
    redacted = SECRET.sub("[비공개]", value)
    return redacted[:MAX_TEXT], len(redacted) > MAX_TEXT


def _safe(value, allowed_keys, *, depth=0):
    if depth >= MAX_DEPTH:
        return None, value is not None
    if value is None or isinstance(value, (bool, int, float)):
        return value, False
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, dict):
        cleaned, truncated = {}, False
        for key, item in value.items():
            key = str(key)
            if key not in allowed_keys:
                continue
            safe_item, item_truncated = _safe(item, allowed_keys, depth=depth + 1)
            cleaned[key] = safe_item
            truncated |= item_truncated
        return cleaned, truncated
    if isinstance(value, (list, tuple)):
        cleaned, truncated = [], len(value) > MAX_ITEMS
        for item in value[:MAX_ITEMS]:
            safe_item, item_truncated = _safe(item, allowed_keys, depth=depth + 1)
            cleaned.append(safe_item)
            truncated |= item_truncated
        return cleaned, truncated
    return None, False


def sanitize(value, limit, allowed_keys=SAFE_KEYS):
    """허용 키만 남긴 뒤 UTF-8 바이트 상한을 적용한다. (정제값, 잘림여부)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            value = None
    cleaned, truncated = _safe(value, allowed_keys)
    if isinstance(cleaned, list):
        cleaned = {"items": cleaned}
    if not isinstance(cleaned, dict) or not cleaned:
        return None, truncated
    encoded = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":")).encode()
    if len(encoded) <= limit:
        return cleaned, truncated
    return {"truncated": True}, True


def is_error_result(value):
    """예외 대신 오류 문자열/딕셔너리를 돌려주는 기존 도구 계약을 성공으로 보지 않는다."""
    if isinstance(value, str):
        text = value.strip().lower()
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            decoded = None
        if isinstance(decoded, dict) and decoded.get("error"):
            return True
        return text.startswith(("조회 오류", "오류", "error", "도구 입력 형식", "허용되지"))
    return isinstance(value, dict) and bool(value.get("error"))


class ProgressCollector:
    """요청 한 건의 이벤트 순번·정제·전달을 담당한다.

    persistent 이벤트는 producer가 DB에 먼저 저장한 뒤 public event만 큐에 넣는다.
    따라서 소비자가 끊겨도 이미 실행된 started/terminal 감사 기록은 유실되지 않는다.
    """

    def __init__(self, turn_id, *, persistent=False, output=None):
        self.turn_id = uuid.UUID(str(turn_id))
        self.persistent = persistent
        self.output = output
        self.cancelled = False
        self._lock = threading.Lock()
        self._sequence = 0
        self._ordinary = 0

    # ── 전달 ────────────────────────────────────────────────────────────
    def put(self, item):
        """소비자가 살아 있는 동안만 기다린다 (느린/끊긴 소비자가 생산자를 막지 않게)."""
        while not self.cancelled:
            try:
                self.output.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    @staticmethod
    def save(row):
        if row is None:
            return
        try:
            ChatProgressEvent.objects.create(**row)
        except Exception as exc:
            raise ProgressStorageError("chat progress storage failed") from exc

    # ── 수집 ────────────────────────────────────────────────────────────
    def emit(self, *, kind, status, label, operation_id=None, parent_operation_id=None,
             tool_name=None, tool_call_id=None, arguments=None, result=None, direct=False):
        terminal = status != "started"
        with self._lock:
            # 취소 뒤에는 새 실행을 시작하지 않는다. 이미 시작한 실행의 종료는 항상 받는다.
            if not terminal:
                if self.cancelled or self._ordinary >= MAX_EVENTS:
                    return None
                self._ordinary += 1
            self._sequence += 1
            sequence = self._sequence
            raw_tool_name = str(tool_name) if tool_name else None
            allowed_keys = (
                TOOL_SAFE_KEYS.get(raw_tool_name, frozenset()) if kind == "tool"
                else RETRIEVAL_SAFE_KEYS if kind == "retrieval"
                else frozenset()
            )
            argument_snapshot, argument_truncated = sanitize(
                arguments, MAX_ARGUMENT_BYTES, allowed_keys
            )
            result_snapshot, result_truncated = sanitize(
                result, MAX_RESULT_BYTES, allowed_keys
            )
            operation_id = uuid.UUID(str(operation_id or uuid.uuid4()))
            parent_operation_id = (
                uuid.UUID(str(parent_operation_id)) if parent_operation_id else None
            )
            label = _text(str(label))[0][:160]
            tool_name = _text(raw_tool_name)[0][:80] if raw_tool_name else None
            row = {
                "turn_id": self.turn_id,
                "sequence_no": sequence,
                "operation_id": operation_id,
                "parent_operation_id": parent_operation_id,
                "kind": kind,
                "status": status,
                "label": label,
                "tool_name": tool_name,
                "tool_call_id": _text(str(tool_call_id))[0][:255] if tool_call_id else None,
                "arguments": argument_snapshot,
                "result": result_snapshot,
                "truncated": argument_truncated or result_truncated,
            } if self.persistent else None
            event = {
                "turn_id": str(self.turn_id),
                "sequence_no": sequence,
                "operation_id": str(operation_id),
                "parent_operation_id": str(parent_operation_id) if parent_operation_id else None,
                "kind": kind,
                "status": status,
                "label": label,
                "created_at": timezone.now().isoformat().replace("+00:00", "Z"),
                "tool_name": tool_name,
                "tool_call_id": _text(str(tool_call_id))[0][:255] if tool_call_id else None,
                "arguments": argument_snapshot,
                "result": result_snapshot,
                "truncated": argument_truncated or result_truncated,
                "summary": None,
            }
            if self.persistent:
                self.save(row)
            if self.output is not None and not direct:
                self.put(("progress", event))
            return operation_id

    def cancel(self, persist=True):
        """브라우저 연결 종료·조기 종료 신호. 이미 실행 중인 호출을 중단됐다고 적지 않는다."""
        if self.cancelled:
            return
        self.cancelled = True          # 먼저 세워야 큐에 막힌 생산자가 풀린다
        if persist:
            self.emit(kind="phase", status="interrupted", label=CANCEL_LABEL, direct=True)


@contextmanager
def collect(collector):
    token = _CURRENT.set(collector)
    try:
        yield collector
    finally:
        _CURRENT.reset(token)


def current():
    return _CURRENT.get()


@contextmanager
def operation(kind, name, *, parent_operation_id=None, arguments=None):
    """LangChain 콜백이 못 보는 수동 단계만 감싼다 (공통 콜백과 중복 기록하지 않는다)."""
    collector = current()
    if collector is None:
        yield None
        return
    if collector.cancelled:
        raise ProgressCancelled("chat progress cancelled")
    label = LABELS.get(name, "조회 중" if kind != "phase" else "처리 중")
    tool_name = name if kind == "tool" else None
    operation_id = collector.emit(
        kind=kind, status="started", label=label, parent_operation_id=parent_operation_id,
        tool_name=tool_name, arguments=arguments,
    )
    if operation_id is None:          # 이벤트 상한 초과 — 실행 자체는 계속한다
        yield None
        return
    if collector.cancelled:           # started 저장/큐 대기 사이 disconnect
        collector.emit(
            kind=kind, status="interrupted", label=CANCEL_LABEL,
            operation_id=operation_id, parent_operation_id=parent_operation_id,
            tool_name=tool_name,
        )
        raise ProgressCancelled("chat progress cancelled")
    try:
        yield operation_id
    except ProgressCancelled:
        collector.emit(
            kind=kind, status="interrupted", label=CANCEL_LABEL,
            operation_id=operation_id, parent_operation_id=parent_operation_id,
            tool_name=tool_name,
        )
        raise
    except Exception:
        collector.emit(
            kind=kind, status="failed", label=label.replace(" 중", " 실패"),
            operation_id=operation_id, parent_operation_id=parent_operation_id,
            tool_name=tool_name,
        )
        raise
    else:
        collector.emit(
            kind=kind, status="completed", label=label.replace(" 중", " 완료"),
            operation_id=operation_id, parent_operation_id=parent_operation_id,
            tool_name=tool_name,
        )


class ProgressCallback(BaseCallbackHandler):
    """설치된 langchain-core 콜백으로 도구/모델 실행 시작·종료를 짝지어 기록한다."""

    raise_error = True

    def __init__(self):
        self.operations = {}

    def _parent(self, parent_run_id):
        entry = self.operations.get(str(parent_run_id))
        return entry[0] if entry else None

    def _start(self, run_id, parent_run_id, kind, name, arguments=None, tool_call_id=None):
        collector = current()
        if collector is None:
            return
        if collector.cancelled:
            raise ProgressCancelled("chat progress cancelled")
        operation_id = collector.emit(
            kind=kind, status="started", label=LABELS.get(name, "조회 중"),
            parent_operation_id=self._parent(parent_run_id),
            tool_name=name if kind == "tool" else None,
            tool_call_id=tool_call_id, arguments=arguments,
        )
        if operation_id is not None:
            self.operations[str(run_id)] = (operation_id, name, kind, tool_call_id)
            if collector.cancelled:
                collector.emit(
                    kind=kind, status="interrupted", label=CANCEL_LABEL,
                    operation_id=operation_id, parent_operation_id=self._parent(parent_run_id),
                    tool_name=name if kind == "tool" else None,
                )
                raise ProgressCancelled("chat progress cancelled")

    def _end(self, run_id, parent_run_id, output=None, failed=False):
        collector = current()
        entry = self.operations.pop(str(run_id), None)
        if collector is None or entry is None:
            return
        operation_id, name, kind, tool_call_id = entry
        status = "failed" if failed or is_error_result(output) else "completed"
        label = LABELS.get(name, "조회 중").replace(" 중", " 실패" if status == "failed" else " 완료")
        collector.emit(
            kind=kind, status=status, label=label, operation_id=operation_id,
            parent_operation_id=self._parent(parent_run_id),
            tool_name=name if kind == "tool" else None,
            tool_call_id=tool_call_id,
            result=None if failed else output,
        )

    # 모델 호출: prompts/messages 는 수집하지 않는다.
    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        self._start(run_id, parent_run_id, "phase", "model")

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        self._start(run_id, parent_run_id, "phase", "model")

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        self._end(run_id, parent_run_id)

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._end(run_id, parent_run_id, failed=True)

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None,
                      inputs=None, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        self._start(run_id, parent_run_id, "tool", name,
                    inputs if inputs is not None else input_str,
                    tool_call_id=kwargs.get("tool_call_id"))

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        self._end(run_id, parent_run_id, output=getattr(output, "content", output))

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._end(run_id, parent_run_id, failed=True)


def callbacks():
    return [ProgressCallback()] if current() is not None else []


def config_kwargs(**extra):
    """수집기가 없으면 config 자체를 넘기지 않는다 (config 를 안 받는 기존 호출부 보호)."""
    config = dict(extra)
    handlers = callbacks()
    if handlers:
        config["callbacks"] = handlers
    return {"config": config} if config else {}


def collector_context(collector):
    """이미 바깥에서 수집 중이면 새 수집기를 덮어쓰지 않는다."""
    return nullcontext() if current() is not None else collect(collector)


def project_event(event, include_details=False):
    """서버 내부 event에서 권한별 wire payload를 만든다."""
    public = {
        key: event[key] for key in (
            "turn_id", "sequence_no", "operation_id", "parent_operation_id", "kind",
            "status", "label", "created_at", "tool_name", "summary",
        )
    }
    if include_details:
        public.update({key: event[key] for key in (
            "tool_call_id", "arguments", "result", "truncated",
        )})
    return public
