"""assistant 파이프라인(프롬프트 · RAG · 에이전트 · 파서) 단위 테스트 — 진짜 LLM·DB·카카오 없이 돈다.

    cd backend && python -m unittest llm.v1.rag.assistant.test_assistant
"""
import json
import unittest
from threading import Event
from types import SimpleNamespace
from unittest import mock

from django.test import override_settings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

from . import pipeline, tools
from ...tools import DOMAIN_TOOL_NAMES

ALL_TABLES = ["TEAM", "STADIUM", "HOME_CONTEXT", "POSTSEASON_STAGE", "GAME", "STANDING_HISTORY", "SEAT_ZONE",
              "TICKET_PRICE", "TICKET_POLICY"]


class ScriptedModel(GenericFakeChatModel):
    """정해진 AIMessage 를 차례로 돌려주는 모델 (create_agent 가 부르는 bind_tools 는 자기 자신)."""

    def bind_tools(self, tool_list, **kwargs):
        return self


def scripted(*messages):
    return ScriptedModel(messages=iter(messages))


def call(name, args, i):
    return {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}


class FakeService:
    def __init__(self, rows=None):
        self.sql, self.rows = [], rows

    def get_baseball_schema(self):
        return {"tables": [{"quoted_name": '"TEAM"'}]}

    def execute_baseball_select(self, sql, params, max_rows):
        self.sql.append((sql, params, max_rows))
        rows = self.rows if self.rows is not None else [["2026-09-16", "18:30", "KIA 타이거즈", "LG 트윈스", "광주", None, None, "PREV"]]
        cols = ["game_date", "game_time", "home_team", "away_team", "stadium", "home_score", "away_score", "status_code"]
        return {"columns": cols, "rows": rows, "truncated": False}


def no_docs(inputs):
    return {**inputs, "context": "검색 결과 없음", "stadium": None, "doc_count": 0}


class PromptAndRagTest(unittest.TestCase):
    def setUp(self):
        tools.new_state("JAMSIL")

    def test_prompt_has_context_date_hint_and_recent_history(self):
        history = [{"role": "user", "content": f"q{i}"} for i in range(12)] + [{"role": "system", "content": "무시"}]
        out = pipeline.build_prompt({"question": "주차 얼마야?", "history": history, "hint": "JAMSIL",
                                     "today": "2026-09-15", "context": "[1] 주차 876면", "route_hint": ""})
        msgs = out["messages"]
        self.assertIsInstance(msgs[0], SystemMessage)
        for text in ("2026-09-15", "잠실야구장(JAMSIL)", "[1] 주차 876면"):
            self.assertIn(text, msgs[0].content)
        for token in ("{today}", "{context}", "{route_hint}", "{stadium_hint}"):
            self.assertNotIn(token, msgs[0].content)
        self.assertEqual(len(msgs), 1 + pipeline.HISTORY_TURNS + 1)
        self.assertIsInstance(msgs[-1], HumanMessage)

    def test_route_hint_points_to_course_and_nearby_tools(self):
        self.assertIn("plan_course", pipeline.route_hint("친구랑 잠실 경기 전후 코스 짜줘"))
        self.assertIn('kind="stay"', pipeline.route_hint("챔피언스 필드 주변 숙박 추천"))
        self.assertEqual(pipeline.route_hint("LG 몇 위야?"), "")

    def test_visible_text_drops_reasoning_and_tool_blocks(self):
        self.assertEqual(pipeline._text([
            {"type": "reasoning", "text": "숨은 추론"},
            {"type": "tool_call", "text": "도구 인자"},
            {"type": "output_text", "text": "공개 답변"},
        ]), "공개 답변")

    def test_retrieve_uses_question_then_history_then_screen_stadium(self):
        seen = []

        def fake_search(vec, k, st, cats):
            seen.append((st, cats))
            return [{"doc_id": "T1", "stadium": st, "category": "TRANSPORT", "status": "PARTIAL",
                     "evidence_type": "OFFICIAL", "content": "주차 안내"}]
        embed = lambda q: [0.0]  # noqa: E731
        out = pipeline.retrieve({"question": "고척 주차 얼마야?", "hint": "JAMSIL"}, _search=fake_search, _embed=embed)
        self.assertEqual(seen[-1], ("GOCHEOK", ["TRANSPORT", "PRICE"]))
        self.assertIn("등급=UNCERTAIN", out["context"])
        pipeline.retrieve({"question": "주차는?", "history": [{"role": "user", "content": "광주 가요"}], "hint": "JAMSIL"},
                          _search=fake_search, _embed=embed)
        self.assertEqual(seen[-1][0], "GWANGJU")
        pipeline.retrieve({"question": "내일 경기 몇 시야?", "hint": "JAMSIL"}, _search=fake_search, _embed=embed)
        self.assertEqual(seen[-1], ("JAMSIL", None))           # 일정은 DB 가 정본 → 문서 카테고리로 좁히지 않는다

    def test_retrieve_failure_still_answers(self):
        def boom(*a):
            raise RuntimeError("db down")
        out = pipeline.retrieve({"question": "주차"}, _search=boom, _embed=lambda q: [0.0])
        self.assertEqual(out["context"], "검색 결과 없음")

    def test_parser_takes_last_plain_ai_message(self):
        result = {"messages": [AIMessage(content="", tool_calls=[call("x", {}, 1)]), AIMessage(content=[{"type": "text", "text": " 답 "}])]}
        self.assertEqual(pipeline.parse_output(result), "답")
        self.assertEqual(pipeline.parse_output({"messages": [AIMessage(content="", tool_calls=[call("x", {}, 1)])]}), "")


class DbToolTest(unittest.TestCase):
    def setUp(self):
        tools.new_state("GWANGJU")

    def test_get_games_filters_and_counts(self):
        svc = FakeService()
        out = json.loads(tools.get_games(team="기아", home_away="home", status="upcoming", date_from="2026-09-15", _service_obj=svc))
        sql, params, _ = svc.sql[0]
        self.assertEqual(params, {"d_from": "2026-09-15", "d_to": "2027-10-20", "team": "KIA", "status": "PREV"})
        self.assertIn("h.team_code = %(team)s", sql)
        self.assertNotIn("a.team_code = %(team)s", sql)
        self.assertEqual(out["count"], 1)
        self.assertIn("games", tools.state()["tools"])

    def test_bad_inputs_do_not_query(self):
        svc = FakeService()
        self.assertIn("형식", tools.get_games(date_from="9월 16일", _service_obj=svc))
        self.assertIn("팀을 찾지 못했", tools.get_ticket_prices(team="없는팀", _service_obj=svc))
        self.assertEqual(svc.sql, [])

    def test_ticket_prices_accept_stadium_when_team_is_unknown(self):
        svc = FakeService()
        tools.get_ticket_prices(stadium="잠실야구장", _service_obj=svc)
        sql, params, _ = svc.sql[0]
        self.assertEqual(params, {"stadium": "JAMSIL"})
        self.assertIn("s.stadium_code = %(stadium)s", sql)

    def test_free_sql_requires_schema_first_and_caps_rows(self):
        svc = FakeService()
        self.assertIn("먼저 get_baseball_schema", tools.execute_baseball_select("SELECT 1", _service_obj=svc))
        tools.get_baseball_schema(_service_obj=svc)
        tools.execute_baseball_select('SELECT * FROM "TEAM"', max_rows=999, _service_obj=svc)
        self.assertEqual(svc.sql[-1][2], tools.MAX_SQL_ROWS)

    def test_tool_schemas_hide_test_hooks(self):
        for t in tools.build_tools():
            self.assertFalse([k for k in t.args if k.startswith("_")], t.name)

    def test_agent_gets_all_registered_and_agent_specific_tools_without_duplicates(self):
        names = [tool.name for tool in tools.build_tools()]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            set(names),
            set(DOMAIN_TOOL_NAMES) | {
                "get_baseball_schema", "execute_baseball_select", "search_kbo_documents",
                "search_nearby_places", "plan_course", "get_ticket_policy", "search_documents_tool",
            },
        )
        self.assertIs(tools.build_tools()[0].func, tools.get_games)

    @override_settings(BASEBALL_QUERY_MAX_ROWS=200, BASEBALL_QUERY_MAX_SQL_BYTES=32768)
    def test_fixed_sql_passes_real_validator(self):
        """고정 SQL 이 성호 검증기(읽기 전용·허용 테이블·허용 함수)를 통과하는지."""
        from baseball.query_service import BaseballQueryService
        repo = mock.Mock()
        repo.models.return_value = [SimpleNamespace(_meta=SimpleNamespace(db_table=t)) for t in ALL_TABLES]
        repo.execute_readonly.return_value = {"columns": [], "rows": [], "truncated": False}
        svc = BaseballQueryService(repo)
        for out in (tools.get_games(team="KIA", stadium="잠실", status="finished", _service_obj=svc),
                    tools.get_standings(_service_obj=svc),
                    tools.get_ticket_prices(team="LG", zone_keyword="테이블", _service_obj=svc),
                    tools.get_ticket_policy(team="롯데", _service_obj=svc)):
            self.assertNotIn("조회 오류", out)
        self.assertEqual(repo.execute_readonly.call_count, 4)


class OtherToolTest(unittest.TestCase):
    def setUp(self):
        tools.new_state("JAMSIL", question="친구랑 잠실 경기 전후 코스 짜줘", history=[{"role": "user", "content": "안녕"}])

    def test_nearby_uses_screen_stadium(self):
        places = [{"name": "잠실 게스트하우스", "detail": "여행 > 숙박 > 게스트하우스", "distance": 800, "address": "x",
                   "placeId": "1", "kind": "stay", "lat": 0, "lng": 0}]
        out = json.loads(tools.search_nearby_places("stay", _nearby=lambda code, kind: places if code == "JAMSIL" else []))
        self.assertEqual(out[0]["walk_min"], 10)

    def test_plan_course_keeps_full_question_and_result(self):
        course = mock.Mock()
        course.answer.return_value = {"answer": "코스 글", "places": [{"name": "a"}], "sources": [{"doc_id": "d"}]}
        self.assertEqual(tools.plan_course("잠실 코스", _course=course), "코스 글")
        course.answer.assert_called_once_with("친구랑 잠실 경기 전후 코스 짜줘", history=[{"role": "user", "content": "안녕"}],
                                              hint_stadium="JAMSIL")
        self.assertEqual(tools.state()["course"]["places"], [{"name": "a"}])

    def test_nested_plan_course_restores_and_updates_parent_state(self):
        parent = tools.state()

        class Course:
            @staticmethod
            def answer(*_args, **_kwargs):
                with tools.request_state("CHILD", "child"):
                    tools.state()["schema_seen"] = True
                return {"answer": "중첩 코스", "places": [{"name": "식당"}], "sources": [{"doc_id": "nested"}]}

        self.assertEqual(tools.plan_course("잠실 코스", _course=Course), "중첩 코스")
        self.assertIs(tools.state(), parent)
        self.assertEqual(parent["course"]["places"], [{"name": "식당"}])
        self.assertFalse(parent["schema_seen"])


class AgentFlowTest(unittest.TestCase):
    """RAG → 프롬프트 → 에이전트(도구 호출) → 파서 전체를 가짜 모델로 돌린다."""

    def test_db_question_uses_fixed_tool(self):
        svc = FakeService()
        tool_list = [tools.StructuredTool.from_function(
            lambda **kw: tools.get_games(_service_obj=svc, **kw), name="get_games", args_schema=tools.GamesInput, description="games")]
        model = scripted(
            AIMessage(content="", tool_calls=[call("get_games", {"team": "KIA", "home_away": "home", "status": "upcoming"}, 1)]),
            AIMessage(content="KIA 홈경기는 1경기 남았어요."),
        )
        r = pipeline.answer("KIA 홈경기 몇 경기 남았어?", _chain_obj=pipeline.build_chain(model=model, tool_list=tool_list, retriever=no_docs))
        self.assertEqual(r["answer"], "KIA 홈경기는 1경기 남았어요.")
        self.assertEqual(r["route"], "agent:rag,games")
        self.assertEqual(r["sources"][0]["doc_id"], "baseball_db:games")
        self.assertNotIn("places", r)

    def test_course_result_is_passed_to_map(self):
        course = mock.Mock()
        course.answer.return_value = {"answer": "16:00 식당 → 구장", "places": [{"name": "식당"}], "coursePayload": {"title": "t"},
                                      "stadiumCode": "JAMSIL", "travel": {"mode": "walk"}, "sources": []}
        tool_list = [tools.StructuredTool.from_function(
            lambda request: tools.plan_course(request, _course=course), name="plan_course", args_schema=tools.CourseInput, description="c")]
        model = scripted(
            AIMessage(content="", tool_calls=[call("plan_course", {"request": "잠실 코스 짜줘"}, 1)]),
            AIMessage(content="요약해서 바꾼 코스 설명"),
        )
        r = pipeline.answer("잠실 코스 짜줘", hint_stadium="JAMSIL",
                            _chain_obj=pipeline.build_chain(model=model, tool_list=tool_list, retriever=no_docs))
        self.assertEqual(r["answer"], "16:00 식당 → 구장")          # 지도와 글이 어긋나지 않게 코스 결과 그대로
        self.assertEqual(r["places"], [{"name": "식당"}])
        self.assertEqual(r["stadiumCode"], "JAMSIL")

    def test_empty_answer_raises_for_fallback(self):
        chain = pipeline.build_chain(model=scripted(AIMessage(content="")), tool_list=[], retriever=no_docs)
        with self.assertRaises(ValueError):
            pipeline.answer("안녕", _chain_obj=chain)

    def test_stream_does_not_repeat_the_same_tool_call(self):
        invoked = []

        def get_games(team: str):
            """Test games."""
            invoked.append(team)
            return {"count": 1}

        tool = tools.StructuredTool.from_function(get_games)
        repeated = call("get_games", {"team": "LG"}, 1)
        model = scripted(
            AIMessage(content="", tool_calls=[repeated]),
            AIMessage(content="", tool_calls=[{**repeated, "id": "call_2"}]),
            AIMessage(content="최종 답변"),
        )
        self.assertEqual(
            "".join(pipeline.stream_answer(
                "LG 일정", model=model, tool_list=[tool], retriever=no_docs,
            )),
            "최종 답변",
        )
        self.assertEqual(invoked, ["LG"])

    def test_stream_keeps_new_tool_calls_beside_a_duplicate(self):
        invoked = []

        def get_games(team: str):
            """Test games."""
            invoked.append(("games", team))
            return {"count": 1}

        def get_standings(as_of: str):
            """Test standings."""
            invoked.append(("standings", as_of))
            return {"rank": 1}

        game = call("get_games", {"team": "LG"}, 1)
        model = scripted(
            AIMessage(content="", tool_calls=[game]),
            AIMessage(content="", tool_calls=[
                {**game, "id": "call_2"},
                call("get_standings", {"as_of": "2026-09-16"}, 3),
            ]),
            AIMessage(content="READY"),
            AIMessage(content="최종 답변"),
        )
        self.assertEqual("".join(pipeline.stream_answer(
            "LG 정보", model=model,
            tool_list=[tools.StructuredTool.from_function(get_games), tools.StructuredTool.from_function(get_standings)],
            retriever=no_docs,
        )), "최종 답변")
        self.assertEqual(invoked, [("games", "LG"), ("standings", "2026-09-16")])


class DispatcherTest(unittest.TestCase):
    def test_every_baseball_question_goes_to_assistant(self):
        from .. import dispatcher
        with mock.patch.object(dispatcher.assistant, "answer", return_value={"answer": "답", "sources": [], "route": "agent:rag"}) as a:
            for q in ("LG 몇 위야?", "잠실 코스 짜줘", "광주 숙소 추천", "보조배터리 반입 돼?"):
                self.assertEqual(dispatcher.answer(q)["route"], "agent:rag")
        self.assertEqual(a.call_count, 4)

    def test_off_topic_is_answered_without_agent(self):
        from .. import dispatcher
        with mock.patch.object(dispatcher.assistant, "answer") as a:
            self.assertEqual(dispatcher.answer("오늘 주식 뭐 사?")["route"], "dispatcher:scope")
        a.assert_not_called()

    def test_hard_off_topic_is_blocked_even_with_baseball_words(self):
        """야구 단어를 섞어도 코딩·자동차 구매 같은 무관 주제는 에이전트로 새지 않는다 (2026-09-16 회귀)."""
        from .. import dispatcher
        with mock.patch.object(dispatcher.assistant, "answer") as a:
            for q in ("야구 좋아하는데 파이썬 코딩 알려줘", "야구장 갈 때 탈 자동차 추천해줘 아반떼 어때",
                      "LG 팬인데 코딩 공부법 알려줘", "잠실 직관 가는데 주식 뭐 살까"):
                self.assertEqual(dispatcher.answer(q)["route"], "dispatcher:scope", q)
        a.assert_not_called()
        # 야구 질문은 계속 통과한다
        self.assertNotEqual(dispatcher.route("잠실 주차 요금 얼마야?"), "scope")
        self.assertNotEqual(dispatcher.route("자동차 가지고 잠실 가는데 주차 어디 해?"), "scope")

    def test_agent_failure_falls_back_to_domain(self):
        from .. import dispatcher
        with mock.patch.object(dispatcher.assistant, "answer", side_effect=RuntimeError("boom")), \
             mock.patch.object(dispatcher, "_domain_answer", return_value={"answer": "도메인 답", "sources": [], "route": "club>x"}):
            r = dispatcher.answer("LG 몇 위야?")
        self.assertEqual(r["answer"], "도메인 답")
        self.assertTrue(r["route"].startswith("agent:error>"))

    def test_final_answer_uses_provider_chunks_before_provider_finishes(self):
        release = Event()
        streamed = Event()

        class DelayedModel:
            def bind_tools(self, _tools):
                return self

            def invoke(self, _messages, config=None):
                return AIMessage(content="READY")

            def stream(self, _messages, config=None):
                streamed.set()
                yield AIMessageChunk(content="첫 청크")
                if not release.wait(2):
                    raise AssertionError("consumer did not receive the first provider chunk")
                yield AIMessageChunk(content="와 끝")

        chunks = pipeline.stream_answer(
            "질문", model=DelayedModel(), tool_list=[], retriever=no_docs,
        )
        self.assertEqual(next(chunks), "첫 청크")
        self.assertTrue(streamed.is_set())
        release.set()
        self.assertEqual(list(chunks), ["와 끝"])

    def test_stream_failure_after_public_delta_never_appends_fallback_answer(self):
        from .. import dispatcher

        def partial(*_args, **_kwargs):
            yield "부분 답변"
            raise RuntimeError("provider disconnected")

        with (
            mock.patch.object(dispatcher.assistant, "stream_answer", side_effect=partial),
            mock.patch.object(dispatcher, "_domain_answer") as fallback,
        ):
            stream = dispatcher.stream("LG 일정 알려줘")
            self.assertEqual(next(stream), "부분 답변")
            with self.assertRaises(RuntimeError):
                next(stream)
        fallback.assert_not_called()

    def test_stream_strips_selected_stadium_prefix_like_invoke(self):
        from .. import pipeline as rag_pipeline
        args = rag_pipeline.RagChatChain._args({
            "question": "[선택한 구장: 잠실야구장]\n가는 법은?",
            "chat_history": [HumanMessage(content="잠실 알려줘"), AIMessage(content="잠실 답변")],
            "intent": "stadium",
        })
        self.assertEqual(args, {
            "question": "가는 법은?",
            "history": [
                {"role": "user", "content": "잠실 알려줘"},
                {"role": "assistant", "content": "잠실 답변"},
            ],
            "stadium_name": "잠실야구장",
            "intent": "stadium",
        })


if __name__ == "__main__":
    unittest.main()
