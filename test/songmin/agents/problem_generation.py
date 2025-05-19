from typing import Dict, List, Any
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.logger import setup_logger
from config import GOOGLE_API_KEY

logger = setup_logger(__name__)

class ProblemGenerationAgent:
    """
    공학수학 문제를 생성하는 에이전트
    """
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True,
            temperature=0.2
        )
        self.problem_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 공학수학 문제 생성 에이전트입니다. 사용자의 요청에 따라 적절한 문제를 생성해야 합니다.
            
            당신은 직접 사용자에게 응답하지 않습니다. 대신 TaskManager 에이전트에게 정보를 제공하는 역할을 합니다.
            TaskManager는 당신이 제공한 정보를 바탕으로 최종 응답을 생성할 것입니다.
            
            너는 고등학교 미적분 교재 집필진이다.
            공학수학 문제를 생성하여 Taskmanager에게 제공해라

            """),
            ("human", "위 지침에 따라 문제를 생성해주세요.")
        ])
    
    def generate_problem(self, query: str) -> str:
        """
        주어진 입력에 따라 문제를 생성합니다.
        
        Args:
            query (str): 문제 생성을 위한 입력 정보
            
        Returns:
            str: 생성된 문제 정보 (question과 choice 포함)
        """
        logger.info("ProblemGenerationAgent: 문제 생성 중")
        chain = self.problem_generation_prompt | self.llm | StrOutputParser()
        
        result = chain.invoke({
            "query": query
        })
        
        logger.info(f"ProblemGenerationAgent: 문제 생성 완료")
        return result