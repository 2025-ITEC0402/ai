from typing import Literal, Dict, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.logger import setup_logger
from config import GOOGLE_API_KEY

logger = setup_logger(__name__)

members = ["ExternalSearch", "ProblemSolving"]
options_for_next = ["sufficient", "FINISH"] + members

class RouteResponse(BaseModel):
    """라우팅 결정을 표현하는 클래스"""
    next: Literal[*options_for_next] = Field(description="다음 에이전트 혹은 답변을 위한 충분한 정보가 주어졌는지")
    query: str = Field(description="에이전트에게 정보를 요청할 질의")

class TaskManagerAgent:
    """
    사용자의 요청을 분석하고 적절한 에이전트를 선택하는 슈퍼바이저
    """
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True,
            temperature=0.1
        )
        
        self.routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 공학수학 학습 도우미 시스템의 감독자로, 
            사용자의 요청을 분석하고 적절한 에이전트를 선택하는 역할을 합니다.
            주의 :name필드는 이름을 나타냅니다. 에이전트의 이름을 확인하고 사용자의 메세지로 혼동하지 마세요.
            사용 가능한 에이전트: {members}
            가능한 작업:
            - sufficient: 현재 정보로 충분하여 최종 답변을 생성할 준비가 됨
            - ExternalSearch: 외부 검색 수행 필요
            - ProblemSolving: 문제 풀이 필요"""),
            MessagesPlaceholder(variable_name="messages"),
            ("system", """이전 대화를 고려하여:
            1. 다음 작업자를 결정하세요: {options}
            2. 작업자에게 요청할 정보를 명확히하여 질의를 생성하세요"""),
        ]).partial(members=", ".join(members), options=", ".join(options_for_next))

        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 공학수학 학습 도우미 시스템의 감독자로,
            다른 에이전트로부터 받은 정보를 바탕으로 사용자에게 최종 답변을 생성하는 역할을 합니다.
            정보를 종합하여:
            1. 사용자의 원래 질문에 초점을 맞춘 명확하고 일관된 답변을 생성하세요.
            2. 여러 에이전트의 정보가 있는 경우 이를 통합하고 정리하세요.
            3. 정보 간 불일치가 있다면 가장 신뢰할 수 있는 정보를 우선시하세요.
            4. 수학적 개념이나 풀이는 명확하고 논리적인 단계로 설명하세요.
            5. 수식은 LaTeX 형식으로 정확하게 포함하세요."""),
            MessagesPlaceholder(variable_name="messages"),
            ("system", """수집된 모든 정보를 분석하여 사용자의 질문에 대한 최종 답변을 생성하세요.
            답변을 생성한 후 반드시 FINISH를 선택하세요."""),
        ])
        
    def task_manage(self, state: Dict) -> RouteResponse:
        """
        현재 상태를 기반으로 다음 작업 또는 에이전트를 결정합니다.
        
        Args:
            state: 현재 상태 (메시지 기록 포함)
            
        Returns:
            RouteResponse: 다음 작업 및 질의 정보
        """
        messages = state.get("messages", [])
        logger.info(f"TaskManagerAgent: 메시지 수: {len(messages)}")
        
        if not messages:
            logger.warning("TaskManagerAgent: 메시지가 없습니다.")
            return RouteResponse(next="FINISH", query="메시지가 없습니다. 질문을 입력해주세요.")
        
        if state.get("next") == "sufficient":
            logger.info("TaskManagerAgent: 정보가 충분하여 최종 답변 생성 모드")
            
            chain = self.response_prompt | self.llm
            response = chain.invoke({"messages": messages})
            final_response = AIMessage(content=response.content)
            return {
                "next": "FINISH",
                "query": "",
                "messages": messages + [final_response]
            }
        
        logger.info("TaskManagerAgent: 라우팅 결정 모드")
        chain = self.routing_prompt | self.llm.with_structured_output(RouteResponse)
        response = chain.invoke(state)
        
        logger.info(f"TaskManagerAgent: 결정된 다음 단계 - {response.next} - 질의 : {response.query}")
        
        return response