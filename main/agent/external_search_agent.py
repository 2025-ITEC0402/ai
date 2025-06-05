from langchain_google_genai import ChatGoogleGenerativeAI  # Google Generative AI 기반 채팅 LLM을 사용하기 위한 모듈
from langchain_core.prompts import ChatPromptTemplate       # LangChain에서 프롬프트 템플릿을 생성하기 위한 클래스
from langchain_community.tools.tavily_search import TavilySearchResults  # Tavily 검색 도구 (외부 웹 검색용)
from dotenv import load_dotenv                             # .env 파일에 정의된 환경 변수를 로드하기 위한 함수
from langgraph.prebuilt import create_react_agent          # React 에이전트(Agent) 생성 함수 (LangGraph 사용)
import os                                                    # 운영체제 환경 변수 및 파일 경로 접근을 위한 표준 라이브러리

# .env 파일에 설정된 API 키 등을 환경 변수로 로드
load_dotenv()

class ExternalSearchAgent:
    """
    LangChain 도구를 사용하여 외부 정보를 검색하고 요약하는 에이전트 클래스
    이 에이전트는 대학교 미적분 관련 질문에 대해 외부 검색을 수행하고
    검색 결과를 TaskManager에게 전달할 형식으로 요약한다.
    """

    def __init__(self):
        # 환경 변수에서 Google API 키와 Tavily API 키를 가져옴
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")    # Google LLM 호출을 위한 API 키
        TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")    # Tavily 검색 도구 사용을 위한 API 키

        # ChatGoogleGenerativeAI 모델을 초기화
        # - model: 사용할 Gemini 모델 버전
        # - google_api_key: 위에서 가져온 Google API 키
        # - convert_system_message_to_human: 시스템 메시지를 인간 메시지처럼 변환 여부
        # - temperature: 생성 응답의 랜덤성 정도 (0~1, 낮을수록 결정적)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True,
            temperature=0.2
        )

        # TavilySearchResults 도구를 초기화하여 외부 웹 검색 기능 활성화
        # - max_results: 검색 결과 최대 반환 개수
        # - api_key: Tavily API 키
        # - search_depth: 검색 깊이 옵션 ("advanced" 등)
        self.search_tool = TavilySearchResults(
            max_results=3,
            api_key=TAVILY_API_KEY,
            search_depth="advanced"
        )

        # 검색 도구 리스트에 추가 (후에 에이전트에 전달)
        self.tools = [self.search_tool]

        # 검색 에이전트가 사용할 프롬프트 템플릿을 정의
        # ChatPromptTemplate.from_messages()에 system 메시지와 placeholder를 인자로 전달
        # - system: 에이전트의 역할, 책임, 응답 형식을 상세히 정의
        # - placeholder: 나중에 들어올 실제 사용자 메시지를 삽입할 부분
        self.search_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are the **External Information Search Agent** for University Calculus. Your primary role is to find and summarize relevant mathematical information from external sources using your search tools.

                ## ROLE & COMMUNICATION
                - **Do NOT respond directly to users.** Your output is **exclusively for the TaskManager agent**.
                - Provide highly structured, precise, and accurate information to the TaskManager in the specified text format. The TaskManager will use your output to help form the final user response.

                ## CORE RESPONSIBILITIES & STANDARDS
                1.  **Search Execution:** Formulate effective search queries based on the input and execute searches using your `TavilySearchResults` tool.
                2.  **Information Extraction:** Identify and extract key mathematical concepts, definitions, formulas, theorems, and explanations from the search results.
                3.  **Source Prioritization:** Prioritize information from academic, educational (.edu, .org, reputable university sites), and authoritative mathematical sources.
                4.  **Concise Summarization:** Synthesize the findings into a clear, concise summary, focusing only on information directly relevant to the query.
                5.  **LaTeX Formatting:** Ensure **ALL mathematical expressions and formulas use correct LaTeX formatting**.

                ## MULTI-TURN CONVERSATION FOCUS
                **CRITICAL**: Focus on the most recent message with `name="User"` - this is your current task.
                Only consider agent responses (by `name` field) that occurred AFTER this latest user request.
                Previous conversation turns serve as background context only, not as completed work for the current request.
                Ensure complete coverage of the current request without relying on previous turn's outputs.
                
                ## RESPONSE FORMAT (Text Delimited for TaskManager)
                Information Type: External Search Results
                Search Query: [The exact query you executed using TavilySearchResults]
                Key Concepts Found: [Comma-separated list of 3-5 main mathematical concepts identified, e.g., 'Integration by Parts', 'Chain Rule', 'Limit Definition']
                Important Formulas:
                [List any key formulas or theorems found, each on a new line, using LaTeX formatting (e.g., "$$\\\\int u \\\\,dv = uv - \\\\int v \\\\,du$$").]
                Main Findings Summary:
                [A comprehensive, well-structured summary (2-4 paragraphs) of the relevant information from search results. Integrate definitions, explanations, and context. Ensure all mathematical expressions are in LaTeX.]
                Source Quality Assessment: [e.g., 'High (Academic/Educational)', 'Medium (General Reference)', 'Mixed']
                Limitations/Gaps: [Briefly note any information that was hard to find, conflicting, or potentially missing for a complete understanding.]
            
                Status: [COMPLETE,FAILED]

                ## QUALITY ASSURANCE CHECKLIST (Self-Validation)
                -   Is the search query precise and effective?
                -   Are the extracted concepts and formulas accurate and relevant?
                -   Is the main findings summary comprehensive, concise, and easy to understand?
                -   Is all LaTeX notation accurate and correctly escaped (e.g., `\\\\frac` for `\\frac`)?
                -   Is the information authoritative and reliable?
                -   Does the output strictly adhere to the `RESPONSE FORMAT`?
                
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

        # create_react_agent 함수를 통해 에이전트 인스턴스를 생성
        # - llm: 위에서 초기화한 ChatGoogleGenerativeAI 모델
        # - tools: Tavily 검색 도구 리스트
        # - state_modifier: ChatPromptTemplate으로 구성한 프롬프트 (search_prompt)
        self.agent = create_react_agent(
            self.llm,
            self.tools,
            state_modifier=self.search_prompt
        )
