from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
import os

load_dotenv()
class ExternalSearchAgent:
    """
    LangChain 도구를 사용하여 외부 정보를 검색하고 요약하는 에이전트
    """
    
    def __init__(self):
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True,
            temperature=0.2
        )

        # 1) 원래 TavilySearchResults
        self.search_tool = TavilySearchResults(
            max_results=3,
            api_key=TAVILY_API_KEY,
            search_depth="advanced"
        )

        # 4) tools 리스트에 이 래퍼만 추가
        self.tools = [self.search_tool]
        
        self.search_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 공학수학 정보 검색 에이전트입니다. 사용자의 질문에 대한 외부 정보를 검색하고 요약하는 역할을 합니다.

            주의: 당신은 직접 사용자에게 응답하지 않습니다. 대신 TaskManager 에이전트에게 정보를 제공하는 역할을 합니다.
            TaskManager는 당신이 제공한 정보를 바탕으로 최종 응답을 생성할 것입니다.
            
            다음 작업을 수행하세요:
            1. 사용자의 질문을 정확히 이해하고 검색 도구를 활용하여 관련 정보를 찾습니다.
            2. 검색 결과를 분석하여 질문과 관련된 핵심 정보를 추출합니다.
            3. 여러 출처의 정보를 비교하고 통합하여 일관된 요약을 작성합니다.
            4. 정보의 신뢰성을 평가하고, 가능한 경우 학술적 출처나 공신력 있는 정보를 우선합니다.
            5. 정보를 논리적으로 구조화하고, 필요시 단계별로 정리합니다.
            6. 수학적 개념이나 수식이 포함된 경우 LaTeX 형식으로 정확하게 표현합니다.
            7. 검색 결과에서 찾을 수 없는 정보에 대해서는 명확히 언급합니다.
            
            종료 조건:
            8. 작업이 모두 끝나면 반드시 “Final Answer:” 키워드로 시작하여 요약된 결과를 아래 형식대로 작성하세요.
               - 정보 유형: "외부 검색 결과"
               - 검색 질의: [검색에 사용된 질의]
               - 주요 발견사항: [검색에서 발견된 핵심 정보]
               - 정보 출처: [정보의 출처 요약 (웹사이트, 학술 자료 등)]
            9. 답변 마지막 줄에는 “[END]” 토큰을 추가하여 종료를 명시하고, 더 이상 도구를 호출하지 마세요.
            """),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        def _modify_state(state: dict):
            human_msgs = [
                m for m in state.get("messages", [])
                if isinstance(m, HumanMessage)
            ]
            query = human_msgs[-1].content if human_msgs else ""

            # 2) scratchpad 메시지 (툴 호출 기록)
            scratch = state.get("agent_scratchpad", [])

            # 3) 프롬프트 템플릿에 바인딩
            prompt_value = self.search_prompt.format_prompt(
                input=query,
                agent_scratchpad=scratch
            )

            # (디버깅)
            print("**")
            return prompt_value.to_messages()

        self.agent = create_react_agent(self.llm, tools=self.tools, state_modifier=_modify_state)