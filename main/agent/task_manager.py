from typing import Literal, Dict, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

members = ["ExternalSearch", "ProblemSolving", "ProblemGeneration", "GeneratingResponse", "ExplainTheoryAgent"]

class RouteResponse(BaseModel):
    next: Literal[*members]
class TaskManager:
    """
    사용자의 요청을 분석하고 적절한 에이전트를 선택하는 슈퍼바이저
    """
    def __init__(self):
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True,
            temperature=0.1,
        )
        
        self.routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 공학수학 학습 도우미 시스템의 감독자로, 
            사용자의 요청을 분석하고 적절한 에이전트를 선택하는 역할을 합니다.
            
            주의: name필드는 이름을 나타냅니다. 에이전트의 이름을 확인하고 사용자의 메세지로 혼동하지 마세요.
            주의: 사용자가 요구하는 양을 명확히 이해하고 횟수를 명시하지 않는 이상 일반적인 경우 에이전트를 한 번씩만 호출하세요
             
            사용 가능한 에이전트: {members}
            
            에이전트 선택 기준:
            - 문제를 생성하기 위해 정보를 수집하려는 경우 ProblemGeneration
            - 문제에 대한 풀이 정보가 필요한 경우 ProblemSolving
            - 특정 개념이나 정보에 대한 외부 검색이 필요한 경우 ExternalSearch
            - 이론 설명을 위해 이론에 대한 의미 기반 검색을 수행하려는 경우 ExplainTheoryAgent
            - 모든 정보가 충족되고 사용자의 요청에 대한 최종 응답을 생성하려는 경우 GeneratingResponse (FINISH, END와 같은 역할을 합니다)
            
            
            """),
            MessagesPlaceholder(variable_name="messages"),
            ("system", """메시지 히스토리를 분석하여:
            1. 사용자의 원래 요청이 무엇인지 파악
            2. 이미 해당 작업을 수행한 에이전트가 있는지 확인(message의 name 필드 확인)
            3. 에이전트 라우팅 결정
            4. 해당 에이전트 실행 (next 필드에 에이전트 이름 설정)
            5. 에이전트 결과 수집 후 최종 응답 생성 (next를 "GeneratingResponse"로 설정)
            6. 작업이 미완료라면 적절한 에이전트 선택
            .""")
        ]).partial(members=", ".join(members))
        self.agent = self.routing_prompt | self.llm.with_structured_output(RouteResponse)