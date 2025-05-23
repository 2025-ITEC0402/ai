from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

class ResponseGenerationAgent:
    """
    다른 에이전트들의 결과를 종합하여 최종 사용자 응답을 생성하는 에이전트
    """
    def __init__(self):
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True,
            temperature=0.2
        )
        
        # 더미 툴
        @tool
        def generate_final_response(content: str) -> str:
            """더미 툴로, 아무 기능이 없습니다"""
            return f"최종 응답 생성 완료: {content}"

        self.tools = [generate_final_response]
        
        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 공학수학 학습 시스템의 최종 응답 생성 에이전트입니다.
            
            다른 에이전트들(ExternalSearch, ProblemSolving, ProblemGeneration)이 제공한 정보를 바탕으로 
            사용자에게 최적화된 최종 응답을 생성하는 역할을 합니다.
            
            응답 생성 시 다음 품질 기준을 반영해야 합니다:
            
            1. **정확성 (Accuracy)**:
               - 수학적 개념, 공식, 계산이 정확한지 확인
               - 오류가 있다면 수정하여 제공
               - 불확실한 내용은 명시적으로 표시
            
            2. **명확성 (Clarity)**:
               - 이해하기 쉬운 언어로 설명
               - 논리적이고 체계적인 구조
               - 적절한 예시와 비유 활용
               - 복잡한 개념은 단계별로 분해
            
            3. **관련성 (Relevance)**:
               - 사용자 질문에 직접적으로 답변
               - 불필요한 정보는 제거
               - 사용자의 학습 수준에 맞는 내용 제공
            
            4. **완성도 (Completeness)**:
               - 질문에 대한 완전한 답변 제공
               - 누락된 중요한 정보가 없는지 확인
               - 필요시 추가 학습 방향 제시
            
            응답 형식 가이드라인:
            - 친근하고 교육적인 톤 사용
            - 수학 표기는 LaTeX 형식으로 명확하게 표현
            - 중요한 내용은 **굵게** 강조
            - 단계별 설명이 필요한 경우 번호나 단계 표시
            - 시각적 구분을 위해 마크다운 헤더 활용
            - 마지막에 학습 팁이나 추가 도움말 제공
            
            입력 정보 유형별 처리:
            - **외부 검색 결과**: 개념 설명을 이해하기 쉽게 재구성
            - **문제 풀이**: 단계별 풀이 과정을 명확하게 정리
            - **문제 생성**: 문제와 해설을 교육적으로 제시"""),
        ])
        
        self.agent = create_react_agent(self.llm, self.tools, state_modifier = self.response_prompt)
    