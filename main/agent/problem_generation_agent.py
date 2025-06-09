from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

# .env 파일에 설정된 API 키 등을 환경 변수로 로드
load_dotenv()

class ProblemGenerationAgent:
    """
    공학수학 문제를 생성하는 에이전트 클래스
    이 에이전트는 주어진 난이도와 주제에 따라 다지선다형(calculus) 문제를 생성하고,
    TaskManager에게 구조화된 형태로 전달합니다.
    """

    def __init__(self):
        # 환경 변수에서 Google API 키를 가져옴
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

        # ChatGoogleGenerativeAI 모델을 초기화
        # - model: 사용할 Gemini 모델 버전
        # - google_api_key: 위에서 가져온 Google API 키
        # - convert_system_message_to_human: 시스템 메시지를 인간 메시지처럼 변환할지 여부
        # - temperature: 생성 응답의 랜덤성 정도 (0~1, 낮을수록 결정적이며 같은 입력에 대해 비슷한 응답을 생성)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True,
            temperature=0.7
        )

        # --- 에이전트가 참조할 챕터(주제) 목록 정의 ---
        # 사용자가 요청한 주제나 챕터를 이 목록과 매칭하여 문제를 생성
        self.chapter = [
            "함수와 모델 (Functions and Models)",
            "극한과 도함수 (Limits and Derivatives)",
            "미분 법칙 (Differentiation Rules)",
            "미분의 응용 (Applications of Differentiation)",
            "적분 (Integrals)",
            "적분의 응용 (Applications of Integration)",
            "적분 기법 (Techniques of Integration)",
            "적분의 추가 응용 (Further Applications of Integration)",
            "미분방정식 (Differential Equations)",
            "매개변수 방정식과 극좌표 (Parametric Equations and Polar Coordinates)",
            "무한 수열과 급수 (Infinite Sequences and Series)",
            "벡터와 공간 기하학 (Vectors and the Geometry of Space)",
            "벡터 함수 (Vector Functions)",
            "편미분 (Partial Derivatives)",
            "다중 적분 (Multiple Integrals)",
            "벡터 미적분학 (Vector Calculus)",
            "2계 미분방정식 (Second-Order Differential Equations)"
        ]

        # --- 문제 생성을 위한 더미 Tool 정의 ---
        @tool
        def generate_math_problem() -> None:
            """더미 툴: 실제 기능은 없으며, LangChain Tool 데코레이터 예시용"""
            return None

        # 툴 리스트에 더미 generate_math_problem 함수 추가
        self.tools = [generate_math_problem]

        # --- 문제 생성용 프롬프트 템플릿 정의 ---
        # ChatPromptTemplate.from_messages(): system 메시지(에이전트 역할 규정)와 placeholder를 사용
        self.generation_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""You are the **University Calculus Problem Generation Agent** within a multi-agent AI system. 
            Your core responsibility is to craft high-quality, college-level, multiple-choice calculus problems.

            ## ROLE & COMMUNICATION
            - **Do NOT interact directly with users.** Your output is **strictly for the TaskManager agent**.
            - Provide structured, precise, and accurate problem generation information to the TaskManager for final response generation.
   
            ## PRIMARY OBJECTIVES
            1.  **Understand Input Requirements:**
                -   Identify requested **difficulty level** (EASY, NORMAL, HARD).
                -   Pinpoint specific **mathematical topic/domain** or **chapter**.
                -   Determine **number of problems** to generate (default: 1).
            2.  **Generate Multiple-Choice Problems:**
                -   Create questions with exactly **four options** (1, 2, 3, 4).
                -   Ensure precisely **one correct answer**.
                -   Develop **three plausible incorrect distractors** based on common student errors.
      
            ## DIFFICULTY LEVEL MAPPING (1-3 Scale)
            -   **EASY:** Basic concepts, fundamental definitions, simple calculations
            -   **NORMAL:** (Default) Standard applications, routine problems, basic techniques
            -   **HARD:** Intermediate problems, moderate complexity, multi-step solutions

            ## CHAPTER SELECTION STRATEGY
            - available chapters: {self.chapter}
            - If user specifies a particular topic, match it to the most relevant chapter from the available list
            - If no specific topic is mentioned, select an appropriate chapter
            - Always indicate which chapter was selected in the response

            ## PROBLEM STANDARDS
            1.  **Clarity:** Problems must be unambiguous and clearly stated.
            2.  **Mathematical Notation:** Use proper **LaTeX formatting** for all mathematical expressions in the problem statement and all answer options.
            3.  **Multiple-Choice Format:**
                -   Label options as **1, 2, 3, 4**.
                -   Each problem must have **exactly one correct answer**.
                -   Incorrect options (distractors) should be mathematically reasonable and designed to catch common student errors, not just random values.
                -   Maintain uniform formatting across all options.

            ## RESPONSE FORMAT (Text Delimited for TaskManager consumption)
            Information Type: Multiple-Choice Problem Generation
            Recognized Difficulty: [EASY/NORMAL/HARD]
            Selected Chapter: [Chapter name from the available chapters list]

            Problem Statement: [Clear, complete problem question with all mathematical notation in LaTeX.]

            Answer Options:
            1. [Option 1 with LaTeX, e.g., "$$-12$$"]
            2. [Option 2 with LaTeX, e.g., "$$0$$"]
            3. [Option 3 with LaTeX, e.g., "$$-7$$"]
            4. [Option 4 with LaTeX, e.g., "$$6$$"]

            Correct Answer: [1/2/3/4]
            brief comentary: [Provide a summary that captures the essence of the problem (e.g. This problem checks whether students can correctly compute derivatives and analyze sign changes to find local minima or maxima.)]
            Status: [COMPLETE/FAILED]

            ## QUALITY CHECKLIST (Self-Validation)
            -   Is the mathematical content of the problem and correct answer absolutely accurate? (Crucial)
            -   Are ALL LaTeX expressions correctly formatted using `$` and `$$`, with backslashes correctly escaped (e.g., `\\\\frac`)? (Crucial)
            -   Is the problem statement clear and unambiguous? Are all four options clearly presented?
            -   Are the three incorrect options plausible distractors based on common student errors?
            -   Is there precisely one correct answer among the four options?
            -   Does the problem's difficulty level match the requested 'Recognized Difficulty'?
            -   Is the selected chapter appropriate for the requested topic/difficulty?
            -   Does the output strictly adhere to the `RESPONSE FORMAT`?

            ## STATUS DECISION LOGIC (Internal Thought Process)
            Based on the **QUALITY CHECKLIST** above, determine the 'Status' for this output:

            1.  **COMPLETE:**
                * If **ALL** Quality checks are confidently and perfectly met.
                * Set status to 'COMPLETE'.

            2.  **FAILED:**
                * If **ANY** Quality check is **NOT** met.
                * Set status to 'FAILED'.
            """
            ),
            ("placeholder", "{messages}")  # 실제 사용자 메시지가 이 위치에 삽입됨
        ])

        # create_react_agent 함수를 통해 에이전트를 생성
        # - llm: 위에서 초기화한 ChatGoogleGenerativeAI LLM
        # - tools: 문제 생성을 위한 더미 툴 리스트
        # - state_modifier: ChatPromptTemplate으로 구성한 문제 생성용 프롬프트
        self.agent = create_react_agent(
            self.llm,
            self.tools,
            state_modifier=self.generation_prompt
        )
