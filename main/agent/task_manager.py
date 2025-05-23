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

members = ["ExternalSearch", "ProblemSolving", "ProblemGeneration", "GeneratingResponse"]

class RouteResponse(BaseModel):
    next: Literal[*members]
class TaskManager:
    """
    사용자의 요청을 분석하고 적절한 에이전트를 선택하는 슈퍼바이저
    """
    def __init__(self):
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True,
            temperature=0.1,
        )
        
        self.routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 공학수학 학습 도우미 시스템의 감독자로, 
            사용자의 요청을 분석하고 적절한 에이전트를 선택하는 역할을 합니다.
            
            주의: name필드는 이름을 나타냅니다. 에이전트의 이름을 확인하고 사용자의 메세지로 혼동하지 마세요.
            
            사용 가능한 에이전트: {members}
            
            에이전트 선택 기준:
            - 문제 생성에 대한 정보를 얻으려는 경우 ProblemGeneration
            - 문제 풀이에 대한 정보가 필요한 경우 ProblemSolving
            - 특정 개념이나 정보에 대한 외부 검색이 필요한 경우 ExternalSearch
            - 모든 정보가 충족되고 사용자의 요청에 대한 최종 응답을 생성하려는 경우 GeneratingResponse (FINISH, END와 같은 역할을 합니다)
            
            사용자가 요구하는 양을 명확히 이해하고 횟수를 명시하지 않는 이상 일반적인 경우 에이전트를 한 번씩만 호출하세요
            """),
            MessagesPlaceholder(variable_name="messages"),
            ("system", """이전 대화들을 고려하여:
            1. 사용자 요청 분석 및 절차 파악(문제생성 후 해당 문제 풀이가 필요)
            2. 사용자가 원하는 양, 횟수 등을 정확히 파악하여 에이전트를 필요한 만큼만 호출
            3. 에이전트 라우팅 결정
            4. 해당 에이전트 실행 (next 필드에 에이전트 이름 설정)
            5. 에이전트 결과 수집 후 최종 응답 생성 (next를 "GeneratingResponse"로 설정)
            .""")
        ]).partial(members=", ".join(members))
        self.agent = self.routing_prompt | self.llm.with_structured_output(RouteResponse)