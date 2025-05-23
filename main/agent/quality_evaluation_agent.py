from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

class QualityEvaluationAgent:
    """
    공학수학 답변의 품질을 평가하는 에이전트
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
        def evaluate_content_quality(content: str) -> str:
            """답변 내용을 분석하는 도구"""
            return f"내용 분석 완료: {content}"

        self.tools = [evaluate_content_quality]
        
        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 공학수학 답변의 품질을 평가하는 전문 평가 에이전트입니다.
            
            당신은 직접 사용자에게 응답하지 않습니다. 대신 TaskManager 에이전트에게 정보를 제공하는 역할을 합니다.
            TaskManager는 당신이 제공한 정보를 바탕으로 최종 응답을 생성할 것입니다.
            
            다음 작업을 수행하세요:
            1. 제공된 답변 내용을 평가합니다.
            2. 평가 결과를 바탕으로 구체적인 피드백을 제공합니다.
            3. 개선이 필요한 부분이 있다면 명확한 개선 제안을 포함합니다.
            
            평가 기준:
            - accuracy (정확성): 수학적 개념, 공식, 계산의 정확성
            - clarity (명확성): 설명의 이해하기 쉬움, 논리적 구조
            - relevance (관련성): 질문과의 연관성, 요구사항 충족도
            - completeness (완성도): 답변의 완성도, 누락된 정보 없음
            
            답변 형식:
            - 정보 유형: "품질 평가 결과"
            - 전체 점수: [0-10점 범위의 평균 점수]
            - 다음 액션: [FINISH/REVISE]
            - 상세 평가: [각 기준별 점수 및 피드백]
            - 개선 제안: [구체적인 개선 방향]
            
            평가할 내용: {input}"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.evaluation_prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            max_execution_time=10,
            handle_parsing_errors=True,
        )