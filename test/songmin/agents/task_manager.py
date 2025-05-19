from typing import Literal, Dict, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.logger import setup_logger
from config import GOOGLE_API_KEY
import json
import re

logger = setup_logger(__name__)

members = ["ExternalSearch", "ProblemSolving", "ProblemGeneration"]
options_for_next = ["sufficient", "FINISH"] + members

class RouteResponse(BaseModel):
    """라우팅 결정을 표현하는 클래스"""
    next: Literal[*options_for_next] = Field(description="다음 에이전트 혹은 답변을 위한 충분한 정보가 주어졌는지")
    query: str = Field(description="에이전트에 정보를 요청할 입력")

class TaskManagerAgent:
    """
    사용자의 요청을 분석하고 적절한 에이전트를 선택하는 슈퍼바이저
    """
    
    def __init__(self):
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
            
            문제 생성 및 풀이 워크플로우:
            1. 사용자가 문제 생성을 요청하면 ProblemGeneration 에이전트로 라우팅
            2. 문제가 생성되면 해당 문제에 대한 풀이를 위해 ProblemSolving 에이전트로 라우팅
             - query에는 생성된 문제를 넣으세요
            3. 문제와 풀이가 모두 완료되면 'sufficient'로 결정하여 최종 답변 생성
            
            에이전트 선택 기준:
            - 문제 관련 정보나 문제 생성 요청이 있는 경우 ProblemGeneration
            - 이미 문제가 생성되고 풀이가 필요한 경우 ProblemSolving
            - 특정 개념이나 정보 검색이 필요한 경우 ExternalSearch
            
            """),
            MessagesPlaceholder(variable_name="messages"),
            ("system", """이전 대화를 고려하여:
            1. 메시지 기록을 확인하고 현재까지 어떤 에이전트가 이미 작업했는지 파악하세요.
            3. 사용자가 요청하는 문제의 개수와 주제, 범위를 잘 파악하세요.
            4. 다음 작업자를 결정하세요: {options}
               - 문제 생성만 완료된 상태라면 ProblemSolving을 선택하세요.
               - 문제 생성과 풀이가 모두 완료되었다면 'sufficient'를 선택하세요.
            5. 작업자에게 요청할 정보를 명확히하여 질의를 생성하세요."""),
        ]).partial(members=", ".join(members), options=", ".join(options_for_next))

        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 공학수학 학습 도우미 시스템의 감독자로,
            다른 에이전트로부터 받은 정보를 바탕으로 사용자에게 최종 답변을 생성하는 역할을 합니다.
            
            수집된 모든 정보를 분석하여 반드시 아래 JSON 형식으로 응답하세요:
            
            {{
                "problems": [
                    {{
                        "problem_number": "1",
                        "problem": "여기에 첫 번째 문제 내용 작성",
                        "choices": [
                            "1) 첫 번째 선택지",
                            "2) 두 번째 선택지",
                            "3) 세 번째 선택지",
                            "4) 네 번째 선택지",
                            "5) 다섯 번째 선택지"
                        ],
                        "correct_answer": "정답 번호(1~5 중 하나)",
                        "solution": "여기에 풀이 과정 작성"
                    }},
                    {{
                        "problem_number": "2",
                        "problem": "여기에 두 번째 문제 내용 작성",
                        "choices": [
                            "1) 첫 번째 선택지",
                            "2) 두 번째 선택지",
                            "3) 세 번째 선택지",
                            "4) 네 번째 선택지",
                            "5) 다섯 번째 선택지"
                        ],
                        "correct_answer": "정답 번호(1~5 중 하나)",
                        "solution": "여기에 풀이 과정 작성"
                    }}
                    // 요청된 문제 수에 맞게 추가
                ]
            }}
            
            각 필드에 대한 설명:
            - problems: 모든 문제들을 포함하는 배열
            - problem_number: 문제 번호
            - problem: 문제 전체 내용(수식은 LaTeX 형식으로 포함)
            - choices: 5개의 선택지 배열
            - correct_answer: 정답 번호 또는 답안
            - solution: 단계별 풀이 과정(수식은 LaTeX 형식으로 포함)
            
            반드시 위 JSON 형식을 정확히 따라야 합니다. 키 이름과 구조를 정확히 유지하세요.
            사용자가 요청한 문제 개수만큼 problems 배열에 문제를 포함시키세요."""),
            MessagesPlaceholder(variable_name="messages"),
            ("system", """생성된 모든 문제와 풀이를 정확히 JSON 형식으로 변환하여 제공하세요. 
            추가 설명이나 주석은 포함하지 마세요. 정확한 JSON 형식만 응답하세요."""),
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
            content = response.content
            if content.startswith("```") and content.endswith("```"):
                if "```json" in content[:10]:
                    json_str = content[7:-3].strip()
                else:
                    json_str = content[3:-3].strip()
            else:
                json_str = content
            final_response = AIMessage(content=json_str)
            return {
                "next": "QualityEvaluation",
                "query": final_response,
                "messages": messages + [final_response]
            }

        if state.get("next") == "FINISH":
            logger.info("TaskManagerAgent: 작업 완료 모드")
            return {"next": "FINISH", "query": ""}
            
        logger.info("TaskManagerAgent: 라우팅 결정 모드")
        
        chain = self.routing_prompt | self.llm.with_structured_output(RouteResponse)
        response = chain.invoke({"messages": messages})
        
        logger.info(f"TaskManagerAgent: 결정된 다음 단계 - {response.next} - 입력 : {response.query}")
        
        return response