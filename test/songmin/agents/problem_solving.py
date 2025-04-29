from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from utils.logger import setup_logger
from config import GOOGLE_API_KEY

# 로거 설정
logger = setup_logger(__name__)

class ProblemSolvingAgent:
    """
    공학수학 문제의 단계별 풀이를 제공하는 에이전트
    """
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True,
            temperature=0.1
        )
        
        # 풀이 생성 프롬프트 템플릿
        self.solving_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 공학수학 전문가입니다. 사용자가 제시한 수학 문제에 대해 명확하고 단계적인 풀이를 제공해야 합니다.

            풀이 작성 시 다음 사항을 준수하세요:
            1. 먼저 문제를 정확히 이해합니다.
            2. 풀이 접근 방법에 대한 간략한 설명을 제공합니다.
            3. 모든 풀이 단계를 논리적으로 나열하고 각 단계마다 수행하는 연산과 그 이유를 설명합니다.
            4. 수식은 LaTeX 형식으로 명확하게 표현합니다.
            5. 중간 계산 과정을 모두 보여주고 계산 실수가 없도록 주의합니다.
            6. 최종 답안을 명확히 표시하고 필요시 답안의 의미를 설명합니다.
            7. 가능한 경우, 답안 검증 방법이나 다른 풀이 접근법도 간략히 언급합니다.

            주제: {topic}
            문제: {query}

            응답은 학부 수준 공학수학 학생이 이해할 수 있는 수준으로 작성하되, 수학적 정확성을 반드시 유지하세요."""),
            ("human", "{query}")
        ])
    
    def solve_problem(self, query: str, topic: str = None) -> str:
        """
        주어진 문제에 대한 단계별 풀이를 제공합니다.
        
        Args:
            query (str): 사용자가 제시한 문제
            topic (str, optional): 문제 주제. 기본값은 None.
            
        Returns:
            str: 문제 풀이 텍스트
        """
        logger.info(f"ProblemSolvingAgent: 문제 풀이 생성 중 - 주제: '{topic}', 문제: '{query}'")
        
        chain = self.solving_prompt | self.llm
        result = chain.invoke({
            "topic": topic,
            "query": query
        })
        
        solution = result.content
        logger.info(f"ProblemSolvingAgent: 풀이 생성 완료 (길이: {len(solution)}자)")
        
        return solution