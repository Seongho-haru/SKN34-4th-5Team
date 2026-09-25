import asyncio
from unittest.mock import Mock

from django.test import SimpleTestCase, override_settings

from baseball.query_repository import BaseballQueryExecutionError, BaseballQueryRepository
from baseball.query_service import BaseballQueryService
from llm.v1.tools import create_baseball_tools


@override_settings(BASEBALL_QUERY_MAX_ROWS=200, BASEBALL_QUERY_MAX_SQL_BYTES=32768)
class BaseballToolsTest(SimpleTestCase):
    def setUp(self):
        self.repository = Mock(spec=BaseballQueryRepository)
        self.repository.models.return_value = BaseballQueryRepository.models()
        self.repository.get_schema.return_value = {
            "schema": "public",
            "tables": [{"name": "TEAM", "quoted_name": '"TEAM"', "columns": []}],
        }
        self.repository.execute_readonly.return_value = {
            "columns": ["team_name_ko", "games"],
            "rows": [["홈", 1]],
            "truncated": False,
        }
        self.schema_tool, self.select_tool = create_baseball_tools(
            BaseballQueryService(self.repository)
        )

    def test_tools_expose_serializable_langchain_schemas_and_invoke_in_sequence(self):
        self.assertEqual(self.schema_tool.name, "get_baseball_schema")
        self.assertEqual(self.schema_tool.args_schema.model_json_schema()["properties"], {})
        properties = self.select_tool.args_schema.model_json_schema()["properties"]
        self.assertEqual(properties["max_rows"]["type"], "integer")
        self.assertEqual(self.schema_tool.invoke({})["tables"][0]["quoted_name"], '"TEAM"')

        result = self.select_tool.invoke(
            {
                "sql": (
                    'SELECT t.team_name_ko, COUNT(g.id) AS games FROM "TEAM" t '
                    'JOIN "GAME" g ON g.home_team_id=t.id '
                    'WHERE t.id=%(team_id)s GROUP BY t.id'
                ),
                "params": {"team_id": 1},
                "max_rows": 10,
            }
        )
        self.assertEqual(result["rows"], [["홈", 1]])
        _, params, max_rows = self.repository.execute_readonly.call_args.args
        self.assertEqual(params, {"team_id": 1})
        self.assertEqual(max_rows, 10)

    def test_real_service_validation_and_tool_errors_are_safe(self):
        for sql in ('DELETE FROM "TEAM"', "SELECT * FROM auth_user"):
            with self.subTest(sql=sql):
                message = self.select_tool.invoke({"sql": sql, "max_rows": 10})
                self.assertNotIn("Traceback", message)
                self.assertFalse(self.repository.execute_readonly.called)

        self.assertEqual(
            self.select_tool.invoke({'sql': 'SELECT * FROM "TEAM"', "max_rows": True}),
            "도구 입력 형식이 올바르지 않습니다. 스키마와 인자 설명을 확인하세요.",
        )
        self.assertEqual(
            self.select_tool.invoke({'sql': 'SELECT * FROM "TEAM"', "max_rows": "10"}),
            "도구 입력 형식이 올바르지 않습니다. 스키마와 인자 설명을 확인하세요.",
        )

        self.repository.execute_readonly.side_effect = BaseballQueryExecutionError(
            "야구 조회를 실행하지 못했습니다."
        )
        self.assertEqual(
            self.select_tool.invoke({'sql': 'SELECT * FROM "TEAM"'}),
            "야구 조회를 실행하지 못했습니다.",
        )
        self.repository.execute_readonly.side_effect = RuntimeError("unexpected detail")
        with self.assertRaisesRegex(RuntimeError, "unexpected detail"):
            self.select_tool.invoke({'sql': 'SELECT * FROM "TEAM"'})

    def test_sync_tools_are_available_through_ainvoke(self):
        result = asyncio.run(self.select_tool.ainvoke({'sql': 'SELECT * FROM "TEAM"'}))
        self.assertEqual(result["columns"], ["team_name_ko", "games"])
