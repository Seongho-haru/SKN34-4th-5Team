"""Explicit public UI tool; never render model HTML or navigate to a model URL."""
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator


class PlanningQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=160)
    choices: list[str] = Field(max_length=10, description="Empty for free text; otherwise 2-10 distinct short choices.")

    @field_validator("choices")
    @classmethod
    def validate_choices(cls, choices):
        if len(choices) == 1 or any(not c.strip() or len(c) > 80 for c in choices) or len(set(choices)) != len(choices):
            raise ValueError("Use empty choices for free text or 2-10 distinct nonempty choices")
        return choices


def public_ui(value):
    if not isinstance(value, dict) or not isinstance(value.get("offer_writer"), bool):
        return None
    questions = value.get("questions")
    if not isinstance(questions, list) or len(questions) > 4 or (not questions and not value["offer_writer"]):
        return None
    cleaned = []
    for item in questions:
        if not isinstance(item, dict):
            return None
        question, choices = item.get("question"), item.get("choices")
        if not isinstance(question, str) or not question.strip() or len(question) > 160:
            return None
        if not isinstance(choices, list) or len(choices) == 1 or len(choices) > 10:
            return None
        if any(not isinstance(c, str) or not c.strip() or len(c) > 80 for c in choices) or len(set(choices)) != len(choices):
            return None
        cleaned.append({"question": question, "choices": list(choices)})
    return {"offer_writer": value["offer_writer"], "questions": cleaned}


@tool(response_format="content_and_artifact")
def present_planning_questions(offer_writer: bool, questions: list[PlanningQuestion]):
    """Use for essential supervisor questions only; answer directly when context/results suffice and reuse prior candidates/results without repeated lookup or questions. Before asking for an enumerable team/stadium/player/place, obtain verified candidates via exposed direct tools or a fixed specialist's narrow lookup (ask_baseball for teams/stadiums); main asks after the result, never invent options. Put 2-10 verified candidates in choices, not question prose; existing free input stays available. Show all candidates when at most 10; otherwise disclose a partial list. Empty choices only for genuinely open-ended input or disclosed unavailable facts/failed lookup. Ask 1-4 essential unknown questions. Unspecified schedule can query the whole schedule without requiring team/date. Non-course: offer_writer=False. Course: first condition questions may offer the writer, follow-ups False; a writer-only offer may have no questions. With a known course stadium, call ask_course first without requiring date/game/time."""
    payload = public_ui({"offer_writer": offer_writer, "questions": [q.model_dump() for q in questions]})
    if payload is None:
        raise ValueError("Invalid public planning questions")
    return "계획 조건을 선택하거나 채팅으로 답할 수 있어요.", payload
