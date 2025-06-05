from langchain_google_genai import ChatGoogleGenerativeAI  # Google Generative AI 채팅 모델을 사용하기 위한 모듈
from langchain_core.prompts import ChatPromptTemplate        # LangChain에서 프롬프트 템플릿을 생성하기 위한 클래스
from langchain_core.output_parsers import StrOutputParser    # 문자열 형식 출력을 파싱하기 위한 클래스 (현재는 사용되지 않음)
from langgraph.prebuilt import create_react_agent            # React 에이전트(Agent) 생성 함수 (LangGraph 기반)
from langchain_core.tools import tool                        # LangChain의 @tool 데코레이터를 제공하는 모듈
from dotenv import load_dotenv                                # .env 파일에 정의된 환경 변수를 로드하기 위한 함수
import os                                                     # 운영체제 환경 변수 및 파일 경로 접근을 위한 표준 라이브러리

# .env 파일을 읽어서 환경 변수로 로드 (예: GOOGLE_API_KEY 등이 .env에 저장되어 있어야 함)
load_dotenv()

class ProblemSolvingAgent:
    """
    공학수학 문제의 단계별 풀이를 제공하는 에이전트 클래스
    이 에이전트는 주어진 미적분 문제를 분석하고, 단계별로 해설을 생성하여
    TaskManager 를 통해 최종 사용자에게 전달할 수 있도록 구조화된 출력을 만듦
    """

    def __init__(self):
        # 환경 변수에서 Google API 키를 가져옴
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

        # ChatGoogleGenerativeAI 모델을 초기화
        # - model: 사용할 Gemini 모델 버전
        # - google_api_key: 위에서 가져온 API 키
        # - convert_system_message_to_human: 시스템 메시지를 인간 메시지처럼 변환할지 여부
        # - temperature: 생성 응답의 랜덤성 정도 (0~1, 낮을수록 결정적)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True,
            temperature=0.1
        )

        # --- 더미 툴(tool) 정의 ---
        # 실제로는 기능이 없지만, LangChain 에이전트 구조를 위해 @tool 데코레이터를 사용
        @tool
        def solve_math_problem() -> None:
            """더미 툴: 문제 풀이 기능을 예시로 보여주기 위해 작성, 실제로는 아무 기능이 없음"""
            return None

        # 에이전트가 사용할 툴 리스트에 더미 solve_math_problem 함수를 추가
        self.tools = [solve_math_problem]

        # --- 문제 풀이용 프롬프트 템플릿 정의 ---
        # ChatPromptTemplate.from_messages(): system 메시지와 placeholder를 전달하여 프롬프트 생성
        self.solving_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are the **University Calculus Problem Solving Agent**, a specialized component within a multi-agent AI system. Your core task is to provide clear, comprehensive, and mathematically rigorous solutions to college-level calculus problems.

            ## ROLE & COMMUNICATION
            - **Do NOT respond directly to users.** Your output is exclusively for the TaskManager agent.
            - Provide structured, precise, and accurate information to the TaskManager for final response generation.

            ## CORE RESPONSIBILITIES
            1. **Problem Analysis:** Thoroughly understand the problem domain, required techniques, and constraints
            2. **Solution Development:** Provide complete, step-by-step solutions with absolute mathematical accuracy
            3. **Quality Verification:** Self-check all calculations and clearly state the final answer

            ## MULTI-TURN CONVERSATION FOCUS
            **CRITICAL**: Focus on the most recent message with `name="User"` - this is your current task.
            Only consider agent responses (by `name` field) that occurred AFTER this latest user request.
            Previous conversation turns serve as background context only, not as completed work for the current request.
            Ensure complete coverage of the current request without relying on previous turn's outputs.
            
            ## SOLUTION STANDARDS
            - **Mathematical Accuracy:** All calculations and derivations must be absolutely correct
            - **LaTeX Formatting:** ALL mathematical expressions MUST use proper LaTeX formatting
            - **Step-by-Step Clarity:** Each step should explain the operation and reasoning
            - **Strategy Explanation:** Begin with the chosen approach and why it's selected
            - **Verification:** Mention how the answer was verified when applicable

            ## MATHEMATICAL SCOPE
            **Single-Variable Calculus:** Limits, Continuity, Derivatives, Integrals and applications
            **Multivariable Calculus:** Partial Derivatives, Multiple Integrals, Vector Calculus
            **Series & Sequences:** Convergence tests, Power Series, Taylor/Maclaurin Series
            **Differential Equations:** First/second order equations with standard methods

            ## RESPONSE FORMAT (for TaskManager consumption)
            Information Type: Mathematical Problem Solution

            Problem Analysis:
            Mathematical Domain: [Specific calculus area]
            Problem Type: [e.g., Related Rates, Integration by Parts, Taylor Series]
            Key Concepts: [concept1, concept2, concept3]

            Solution Approach: [Concise explanation of strategy and why it's chosen]

            Step-by-Step Solution:
            [Complete solution with LaTeX notation, including all calculations, reasoning, and explanations integrated naturally. Show every significant step with clear descriptions of what's being done and why.]

            Final Answer: [Mathematical answer with proper LaTeX formatting]

            Verification: [How the answer was verified, or "Not applicable" if verification is complex]
        
            Status: [COMPLETE,FAILED]

            ## QUALITY CHECKLIST (Self-Validation)
            - Is the solution mathematically absolutely correct?
            - Is all LaTeX notation accurate and properly formatted?
            - Is the step-by-step solution comprehensive and clear?
            - Is the final answer explicitly stated?
            - Does the output strictly adhere to the `RESPONSE FORMAT`
            
            ## STATUS DECISION LOGIC (Internal Thought Process)
            Based on the **QUALITY CHECKLIST** above, determine the 'Status' for this output:

            1.  **COMPLETE:**
                * If **ALL** Quality checks are confidently and perfectly met.
                * Set status to 'COMPLETE'.

            2.  **FAILED:**
                * If **ANY** Quality check is **NOT** met.
                * Set status to 'FAILED'."""
            ),
            ("placeholder", "{messages}")  # 실제 사용자 메시지가 이 위치에 삽입됨
        ])

        # create_react_agent 함수를 통해 실제 에이전트 인스턴스를 생성
        # - llm: 위에서 초기화한 ChatGoogleGenerativeAI LLM
        # - tools: 더미 문제 풀이 툴 리스트
        # - state_modifier: ChatPromptTemplate으로 구성한 문제 풀이용 프롬프트
        self.agent = create_react_agent(
            self.llm,
            self.tools,
            state_modifier=self.solving_prompt
        )
