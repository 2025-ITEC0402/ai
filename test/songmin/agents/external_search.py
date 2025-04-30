from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from config import GOOGLE_API_KEY, TAVILY_API_KEY
from utils.logger import setup_logger

class ExternalSearchAgent:
    """
    LangChain 도구를 사용하여 외부 정보를 검색하고 요약하는 에이전트
    """
    
    def __init__(self):
        self.logger = setup_logger("external_search")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True,
            temperature=0.2
        )
        
        self.search_tool = TavilySearchResults(
            max_results=5,
            api_key=TAVILY_API_KEY,
            search_depth="advanced"
        )
        
        self.tools = [self.search_tool]
        
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        self.search_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 공학수학 정보 검색 전문가입니다. 사용자의 질문에 대한 정보를 찾고 포괄적이고 정확한 답변을 제공해야 합니다.

            다음 작업을 수행하세요:
            1. 사용자의 질문을 정확히 이해하고 필요한 정보를 찾기 위해 검색 도구를 활용합니다.
            2. 검색 결과를 종합적으로 분석하여 관련된 핵심 정보를 추출합니다.
            3. 여러 출처의 정보를 비교하고 통합하여 일관된 답변을 작성합니다.
            4. 정보의 신뢰성을 평가하고, 가능한 경우 학술적 출처나 공신력 있는 정보를 우선합니다.
            5. 답변은 논리적으로 구조화하고, 필요시 단계별로 설명합니다.
            6. 수학적 개념이나 수식은 LaTeX 형식으로 정확하게 표현합니다.
            7. 검색 결과에서 찾을 수 없는 정보에 대해서는 명확히 언급합니다.

            주제: {topic}
            질문: {query}
            검색 결과: {search_results}

            응답은 정확성과 전문성을 유지하면서 학부 수준의 공학수학 학생이 이해할 수 있는 수준으로 작성하세요.
            사용자에게 직접 말하는 형식으로 답변하며, 적절한 수식과 설명을 포함하세요."""),
            ("human", "{query}")
        ])
        
        self.search_chain = self.search_prompt | self.llm_with_tools | StrOutputParser()
    
    def search_and_summarize(self, query: str, topic: Optional[str] = None) -> str:
        """
        주어진 질의에 대해 외부 검색을 수행하고 결과를 요약합니다.
        
        Args:
            query (str): 사용자 질의
            topic (str, optional): 질의 주제. 기본값은 None.
            
        Returns:
            str: 검색 결과 요약 텍스트
        """
        self.logger.info(f"ExternalSearchAgent: 외부 검색 수행 중 - 주제: '{topic}', 질의: '{query}'")
        search_results = self.search_tool.invoke(query)
        
        if not search_results:
            return "검색 결과가 없습니다. 다른 키워드로 검색해보세요."
        
        result = self.search_chain.invoke({
            "topic": topic,
            "query": query,
            "search_results": search_results
        })
        
        return result