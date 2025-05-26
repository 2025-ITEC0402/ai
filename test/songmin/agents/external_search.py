from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from config import GOOGLE_API_KEY, TAVILY_API_KEY
from utils.logger import setup_logger
from langgraph.prebuilt import create_react_agent
class ExternalSearchAgent:
    """
    LangChain 도구를 사용하여 외부 정보를 검색하고 요약하는 에이전트
    """
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True,
            temperature=0.2
        )
        
        self.search_tool = TavilySearchResults(
            max_results=3,
            api_key=TAVILY_API_KEY,
            search_depth="advanced"
        )
        
        self.tools = [self.search_tool]
        
        self.search_agent = create_react_agent(self.llm,tools = self.tools)
        
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
            
            답변 형식:
            - 정보 유형: "외부 검색 결과"
            - 검색 질의: [검색에 사용된 질의]
            - 주요 발견사항: [검색에서 발견된 핵심 정보]
            - 정보 출처: [정보의 출처 요약 (웹사이트, 학술 자료 등)]
            
            질문: {query}
            검색 결과: {search_results}"""),
            ("human", "{query}")
        ])
        
        self.search_chain = self.search_prompt | self.llm_with_tools | StrOutputParser()
    
    def search_and_summarize(self, query: str) -> str:
        """
        주어진 질의에 대해 외부 검색을 수행하고 결과를 요약합니다.
        
        Args:
            query (str): 사용자 질의
            
        Returns:
            str: 검색 결과 요약 텍스트
        """
        self.logger.info(f"ExternalSearchAgent: 외부 검색 수행 중 - 질의: '{query}'")
        search_results = self.search_tool.invoke(query)
        
        if not search_results:
            return "정보 유형: \"외부 검색 결과\"\n검색 질의: \"" + query + "\"\n주요 발견사항: 검색 결과가 없습니다.\n신뢰도 평가: 해당 없음"
        
        result = self.search_chain.invoke({
            "query": query,
            "search_results": search_results
        })
        
        return result