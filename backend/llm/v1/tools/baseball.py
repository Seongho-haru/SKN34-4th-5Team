from typing import Any

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field, StrictInt

from baseball.query_repository import BaseballQueryError
from baseball.query_service import BaseballQueryService, BaseballQueryValidationError


class ExecuteBaseballSelectInput(BaseModel):
    sql: str = Field(
        description=(
            "get_baseball_schema로 확인한 public 야구 테이블의 quoted_name만 사용하는 "
            "단일 PostgreSQL SELECT. JOIN, 집계, 서브쿼리, 비재귀 CTE를 포함할 수 있다."
        )
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description="SQL의 %(name)s named parameter 값. 문자열 보간 대신 사용한다.",
    )
    max_rows: StrictInt = Field(
        default=100,
        description="반환할 최대 행 수. 서비스 설정 범위 안의 정수여야 한다.",
    )


def create_baseball_tools(service: BaseballQueryService | None = None):
    """기존 BaseballQueryService를 호출하는 LangChain 도구 두 개를 만든다."""
    service = service or BaseballQueryService()

    def schema() -> dict:
        """SQL 작성 전에 public 야구 테이블, quoted_name, 컬럼, FK 관계를 조회한다."""
        return service.get_baseball_schema()

    def select(sql: str, params: dict[str, Any] | None = None, max_rows: int = 100) -> dict:
        """스키마에 있는 quoted 테이블만 대상으로 단일 읽기 SELECT를 실행한다."""
        try:
            return service.execute_baseball_select(sql, params, max_rows)
        except (BaseballQueryValidationError, BaseballQueryError) as exc:
            raise ToolException(str(exc)) from None

    invalid_input = "도구 입력 형식이 올바르지 않습니다. 스키마와 인자 설명을 확인하세요."
    return (
        StructuredTool.from_function(
            schema,
            name="get_baseball_schema",
            description=(
                "야구 SQL 작성의 첫 단계로 호출한다. public 스키마의 허용 테이블 quoted_name, "
                "컬럼 타입, PK/null 여부, FK 관계를 반환한다."
            ),
        ),
        StructuredTool.from_function(
            select,
            name="execute_baseball_select",
            description=(
                "get_baseball_schema 결과를 바탕으로 PostgreSQL 단일 SELECT를 실행한다. "
                "quoted 테이블 JOIN, 집계, 서브쿼리, 비재귀 CTE와 %(name)s named params를 "
                "지원하며 결과 행 수는 max_rows로 제한한다."
            ),
            args_schema=ExecuteBaseballSelectInput,
            handle_tool_error=True,
            handle_validation_error=invalid_input,
        ),
    )


get_baseball_schema, execute_baseball_select = create_baseball_tools()
