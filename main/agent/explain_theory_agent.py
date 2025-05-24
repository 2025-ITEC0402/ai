from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
import os

load_dotenv()
class ExplainTheoryAgent:
    """
    LangChain 도구를 사용하여 외부 정보를 검색하고 요약하는 에이전트
    """

    def __init__(self):
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True,
            temperature=0.2
        )

        base_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-exp-03-07"
        )

        #calculus rag 툴
        cal_vectorstore = FAISS.load_local("vectorstore", base_embeddings, allow_dangerous_deserialization=True)
        cal_retriever = cal_vectorstore.as_retriever()
        def calculus_search_fn(query: str) -> list[dict]:
            docs = cal_retriever.get_relevant_documents(query)
            return [
                {
                    "text": doc.page_content,
                    "chapter": doc.metadata.get("Header 1"),
                    "section": doc.metadata.get("Header 2")
                }
                for doc in docs
            ]
        self.cal_tool = Tool.from_function(
            calculus_search_fn,
            name="calculus_search",
            description = (
                "이 도구는 학술적이고 전문적인 수준의 문의에 최적화된 미적분 교재에 대한 의미 기반 검색을 수행합니다. "
                "사용자가 미적분 개념(예: 정의, 정리, 증명, 예제, 공식, 응용)에 대해 자연어로 질문할 때, "
                "가장 관련성 높은 교재 섹션과 발췌문을 챕터 및 섹션 제목 등의 메타데이터와 함께 반환하여, "
                "에이전트가 정확하고 맥락에 부합하며 권위 있는 답변을 제공할 수 있도록 돕습니다."
            )
        )

        #md파일 rag 툴
        md_vectorstore = FAISS.load_local("md_vectorstore", base_embeddings, allow_dangerous_deserialization=True)
        md_retriever = md_vectorstore.as_retriever()
        def md_search_fn(query: str) -> list[dict]:
            docs = md_retriever.get_relevant_documents(query)
            return [
                {
                    "text": doc.page_content,
                    "chapter": doc.metadata.get("Header 1"),
                    "section": doc.metadata.get("Header 2"),
                    "url": doc.metadata["url"],
                }
                for doc in docs
            ]

        self.md_tool = Tool.from_function(
            md_search_fn,
            name="md_search",
            description = (
                "이 도구는 사용자 친화적인 마크다운 형식의 미적분 학습 가이드를 의미 기반으로 검색합니다. "
                "미적분 개념에 대한 자연어 질문이 주어지면, 가장 관련성 높은 마크다운 요약 섹션과 해당 정적 페이지 URL을 함께 반환하여, "
                "에이전트가 “자세한 내용은 이 페이지를 확인하세요.”라고 제안할 수 있도록 합니다."
            )
        )

        # 4) tools 리스트에 이 래퍼만 추가
        self.tools = [self.cal_tool, self.md_tool]

        self.theory_explanation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 ‘이론 설명 에이전트’로, 미적분 개념에 대한 질문에 전문적으로 답변하고 설명합니다. 사용자가 미적분 이론 관련 질문을 하면 다음을 수행해야 합니다:\n"
                    "1. `calculus_search` 도구를 사용해 미적분 교재에서 권위 있는 문단을 의미 기반으로 검색합니다.\n"
                    "2. `md_search` 도구를 사용해 관련된 마크다운 형식 학습 가이드 섹션과 그 정적 페이지 URL을 가져옵니다.\n"
                    "3. 한국인이 이해하기 쉽도록 친절하고 명확하게 답변하세요.\n"
                    "설명 중에는 검색된 문맥을 반드시 인용하고, 두 도구 모두 유효한 결과를 찾지 못하면 내부 지식을 활용하세요. "
                    "응답은 학술적이고 체계적인 구조로 작성하되, 필요할 때 정적 페이지 URL을 함께 제안해야 합니다."
                ),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )


        def _modify_state(state: dict):
            human_msgs = [
                m for m in state.get("messages", [])
                if isinstance(m, HumanMessage)
            ]
            query = human_msgs[-1].content if human_msgs else ""

            # 2) scratchpad 메시지 (툴 호출 기록)
            scratch = state.get("agent_scratchpad", [])

            # 3) 프롬프트 템플릿에 바인딩
            prompt_value = self.theory_explanation_prompt.format_prompt(
                input=query,
                agent_scratchpad=scratch
            )

            # (디버깅)
            print("**")
            return prompt_value.to_messages()

        self.agent = create_react_agent(self.llm, tools=self.tools, state_modifier=_modify_state)