from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda, Runnable
from typing import TypedDict

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


class QuestionGenerator:
    """LLM-기반 객관식 문제 생성기"""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo-0125",
        temperature: float = 0.0,
    ) -> None:
        # 1️⃣ 출력 스키마
        schemas = [
            ResponseSchema(name="question", description="한글 문제"),
            ResponseSchema(name="choice1", description="보기 5개 중 1번"),
            ResponseSchema(name="choice2", description="보기 5개 중 2번"),
            ResponseSchema(name="choice3", description="보기 5개 중 3번"),
            ResponseSchema(name="choice4", description="보기 5개 중 4번"),
            ResponseSchema(name="choice5", description="보기 5개 중 5번"),
        ]
        parser = StructuredOutputParser.from_response_schemas(schemas)
        fmt_instructions = parser.get_format_instructions()

        # 2️⃣ 프롬프트 템플릿
        template = """
            너는 고등학교 미적분 교재 집필진이다.

            ◆ 주제
            {topics}

            ◆ 범위
            {range}

            ◆ 범위 내용 요약 설명
            {summarized}

            ◆ 난이도
            {difficulty}

            ◆ 문제 예시
            {quiz_examples}

            ---
            아래 지시를 **반드시** 지켜라:
            {format_instructions}
        """
        self.prompt = PromptTemplate.from_template(
            template,
            partial_variables={"format_instructions": fmt_instructions},
        )

        # 3️⃣ LLM 세팅
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            response_format={"type": "json_object"},  # JSON 응답 강제
        )

        # 4️⃣ 체인
        self.chain = self.prompt | self.llm | parser

    def generate_question(
        self,
        *,
        topics: str,
        range_: str,
        summarized: str,
        difficulty: str,
        quiz_examples: str,
    ) -> dict:
        """
        문제를 생성하고 JSON(dict)으로 반환한다.
        """
        return self.chain.invoke(
            {
                "topics": topics,
                "range": range_,
                "summarized": summarized,
                "difficulty": difficulty,
                "quiz_examples": quiz_examples,
            }
        )