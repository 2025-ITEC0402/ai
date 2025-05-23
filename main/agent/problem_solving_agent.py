from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

class ProblemSolvingAgent:
    """
    공학수학 문제의 단계별 풀이를 제공하는 에이전트
    """
    
    def __init__(self):
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True,
            temperature=0.1
        )
        #더미 툴
        @tool
        def solve_math_problem(problem: str) -> str:
            """수학 문제를 분석하고 풀이하는 도구"""
            return f"문제 분석 완료: {problem}"
        self.tools = [solve_math_problem]

        self.solving_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 공학수학 문제 풀이 에이전트입니다. 사용자가 제시한 수학 문제에 대해 명확하고 단계적인 풀이를 제공해야 합니다.
            
            당신은 직접 사용자에게 응답하지 않습니다. 대신 TaskManager 에이전트에게 정보를 제공하는 역할을 합니다.
            TaskManager는 당신이 제공한 정보를 바탕으로 최종 응답을 생성할 것입니다.
            
            풀이 작성 시 다음 사항을 준수하세요:
            1. 문제를 정확히 이해하고 분석합니다.
            2. 풀이 접근 방법에 대한 설명을 제공합니다.
            3. 모든 풀이 단계를 논리적으로 나열하고 각 단계마다 수행하는 연산과 그 이유를 설명합니다.
            4. 수식은 LaTeX 형식으로 명확하게 표현합니다.
            5. 중간 계산 과정을 모두 보여주고 계산 실수가 없도록 주의합니다.
            6. 최종 답안을 명확히 표시하고 필요시 답안의 의미를 설명합니다.
            7. 가능한 경우, 답안 검증 방법이나 다른 풀이 접근법도 간략히 언급합니다.
            
            답변 형식:
            - 정보 유형: "문제 풀이"
            - 문제 요약: [문제의 핵심 내용을 간략히 요약]
            - 접근 방법: [문제 해결을 위한 접근 방법 설명]
            - 단계별 풀이: [상세한 풀이 과정]
            - 최종 답안: [명확한 최종 답변]
            - 추가 설명: [필요시 추가 설명이나 대안적 접근법]"""),
        ])

        self.agent = create_react_agent(self.llm, self.tools, state_modifier = self.solving_prompt)