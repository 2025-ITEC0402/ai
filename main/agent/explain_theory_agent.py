from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
import os

load_dotenv()

class ExplainTheoryAgent:
    """
    LangChain 도구를 사용하여 외부 자료를 검색하고 요약하는 에이전트 클래스.
    이 에이전트는 미적분 이론 설명을 위한 전용 프롬프트와 검색 도구들을 구성한다.
    """

    def __init__(self):
        # 환경 변수에서 Google API 키를 가져온다.
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

        # Google Generative AI Chat 모델을 초기화
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro-preview-06-05",      # 사용할 LLM 모델 이름
            google_api_key=GOOGLE_API_KEY,                # 인증을 위한 API 키
            convert_system_message_to_human=True,          # 시스템 메시지를 인간 메시지처럼 변환
            temperature=0.2                                # 응답 랜덤성 정도 (0 ~ 1)
        )

        # Google Generative AI 임베딩 모델을 초기화
        base_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-exp-03-07"      # 사용할 임베딩 모델 경로/이름
        )

        # --- 계산학(calcculus) 관련 RAG 도구 구성 ---
        # 로컬에 저장된 FAISS 벡터스토어를 로드 (벡터 디시리얼라이제이션 허용)
        cal_vectorstore = FAISS.load_local(
            "vectorstore",                                # 벡터스토어 디렉터리 이름
            base_embeddings,                              # 위에서 초기화한 임베딩 객체
            allow_dangerous_deserialization=True           # 위험할 수 있는 직렬화 해제 허용 여부
        )
        # 벡터스토어를 검색기(retriever) 형태로 변환
        cal_retriever = cal_vectorstore.as_retriever(search_kwargs={"k": 2})

        # 실제 검색을 수행하는 함수 정의 (질문을 입력하면 관련 문서 리스트 반환)
        def calculus_search_fn(query: str) -> list[dict]:
            # FAISS retriever를 통해 쿼리와 관련된 문서들을 가져옴
            docs = cal_retriever.get_relevant_documents(query)
            # 각 문서에서 필요한 정보만 추출하여 리스트 형태로 반환
            return [
                {
                    "text": doc.page_content,                 # 문서 내용(텍스트)
                    "chapter": doc.metadata.get("Header 1"),   # 문서 메타데이터에서 "Header 1" 값
                    "section": doc.metadata.get("Header 2")    # 문서 메타데이터에서 "Header 2" 값
                }
                for doc in docs
            ]

        # LangChain Tool 형태로 래핑: 이름, 설명을 포함
        self.cal_tool = Tool.from_function(
            calculus_search_fn,
            name="calculus_search",
            description=(
                "Search academic calculus textbooks (ENGLISH CONTENT) for authoritative mathematical content. "
                "This database contains English textbooks - always use ENGLISH queries for best results. "
                "Use this tool to find rigorous definitions, theorems, proofs, examples, formulas, and applications. "
                "Returns relevant textbook sections with chapter and section metadata for accurate, scholarly explanations."
            )
        )

        # --- Markdown(md) 파일 기반 RAG 도구 구성 (한글 학습 가이드용) ---
        # 로컬에 저장된 md_vectorstore 벡터스토어를 로드
        md_vectorstore = FAISS.load_local(
            "md_vectorstore",                             # md 벡터스토어 디렉터리 이름
            base_embeddings,                              # 동일한 임베딩 객체 사용
            allow_dangerous_deserialization=True           # 위험할 수 있는 직렬화 해제 허용 여부
        )
        # 벡터스토어를 검색기로 변환
        md_retriever = md_vectorstore.as_retriever(search_kwargs={"k": 2})

        # Markdown 검색 함수 정의 (한글 쿼리 입력 시 관련 문서 반환)
        def md_search_fn(query: str) -> list[dict]:
            # FAISS retriever를 통해 문서 검색
            docs = md_retriever.get_relevant_documents(query)
            # 문서 리스트에서 필요한 정보만 추출
            return [
                {
                    "text": doc.page_content,                 # md 파일의 내용
                    "chapter": doc.metadata.get("Header 1"),   # 메타데이터의 "Header 1"
                    "section": doc.metadata.get("Header 2"),   # 메타데이터의 "Header 2"
                    "url": doc.metadata["url"],                # 문서가 위치한 URL
                }
                for doc in docs
            ]

        # LangChain Tool 형태로 래핑: 이름, 설명 포함
        self.md_tool = Tool.from_function(
            md_search_fn,
            name="md_search",
            description=(
                "Search user-friendly markdown calculus learning guides (KOREAN CONTENT) for accessible explanations. "
                "This database contains Korean markdown guides - always use KOREAN queries for best results. "
                "Use this tool to find simplified summaries, study guides, and educational content with URLs. "
                "Returns markdown sections with static page URLs for additional learning resources."
            )
        )

        # 생성된 두 개의 Tool을 리스트로 묶어둠 (나중에 에이전트에 전달)
        self.tools = [self.cal_tool, self.md_tool]

        # --- 에이전트가 사용할 프롬프트 템플릿 정의 ---
        # TaskManager가 처리할 이론 설명용 프롬프트 레이아웃을 선언
        self.theory_explanation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are the **Calculus Theory Explanation Agent**, a specialized component within a multi-agent AI system. 
                    Your expertise lies in providing comprehensive, authoritative, and accessible explanations of calculus concepts.

                    ## ROLE & COMMUNICATION
                    - **Do NOT respond directly to users.** Your output is **exclusively for the TaskManager agent**.
                    - Provide highly structured, well-researched theoretical explanations in the specified text format.

                    ## CORE RESPONSIBILITIES & STANDARDS
                    1.  **Research Strategy:** Use both `calculus_search` (for academic rigor) and `md_search` (for accessible explanations/resources) tools to gather comprehensive information based on the user's query.
                    2.  **Content Synthesis:** Combine retrieved academic rigor with accessible, easy-to-understand explanations. Prioritize accuracy and clarity.
                    3.  **Source Integration:** Clearly incorporate relevant information from search results into your explanation.
                    4.  **Educational Value:** Provide clear, systematic explanations suitable for college-level students, starting from basic concepts and progressing to applications.
                    5.  **LaTeX Formatting:** Ensure **ALL mathematical expressions and formulas use correct LaTeX formatting**. Use `$` for inline math and `$$` for display math.

                    ## MULTI-TURN CONVERSATION FOCUS
                    **CRITICAL**: Focus on the most recent message with `name="User"` - this is your current task.
                    Only consider agent responses (by `name` field) that occurred AFTER this latest user request.
                    Previous conversation turns serve as background context only, not as completed work for the current request.
                    Ensure complete coverage of the current request without relying on previous turn's outputs.
                    
                    ## AVAILABLE TOOLS
                    - **calculus_search:** Academic textbooks with formal definitions, theorems, and rigorous mathematical content. **(Use ENGLISH queries)**
                    - **md_search:** User-friendly markdown learning guides with accessible explanations, examples, and additional resource URLs. **(Use KOREAN queries)**

                    Choose appropriate tools based on the type of information needed and the nature of the inquiry. You may use one or both tools depending on the complexity and desired depth of the explanation.

                    ---
                    ## SEARCH STRATEGY (Internal Thought Process)
                    - **Tool Selection:** Choose tools based on the type of information needed.
                        - `calculus_search`: For formal mathematical definitions, theorems, and academic rigor (always use English queries for this tool).
                        - `md_search`: For practical explanations, examples, and learning resources (always use Korean queries for this tool).
                    - **Query Approach:** Use clear mathematical terminology that precisely describes the concept you're searching for. Ensure the query language matches the tool's requirement.
                    - **Information Synthesis:** When using multiple sources, combine information coherently and logically to provide a comprehensive understanding. Prioritize the most relevant and authoritative information.
                    ---

                    ## RESPONSE FORMAT (Text Delimited for TaskManager consumption)
                    Information Type: Calculus Theory Explanation
                    Concept Query: [The specific mathematical concept requested by the user]
                    Explanation Language: [English/Korean, based on the primary language of the explanation generated]

                    Concept Overview:
                    [A concise, high-level summary of the mathematical concept. Start broad, then narrow down.]

                    Mathematical Content:
                    [Detailed definitions, theorems, fundamental formulas, and rigorous explanations. Use precise mathematical language and include ALL equations/expressions in LaTeX.]

                    Practical Examples & Applications:
                    [One or more clear, step-by-step examples or real-world applications that illustrate the concept. Integrate LaTeX as needed. If no specific examples are found, state 'Not applicable'.]

                    Additional Resources:
                    [List any relevant URLs found using the md_search tool, e.g., "- [Link Title](URL)". If no URLs are found, state 'None found'.]
                    
                    Status: [COMPLETE, FAILED]

                    ## QUALITY ASSURANCE CHECKLIST (Self-Validation)
                    -   Is the explanation mathematically absolutely accurate and authoritative?
                    -   Is all LaTeX notation correct and properly formatted (using `$` and `$$` delimiters)?
                    -   Is the explanation clear, systematic, and appropriate for a college student?
                    -   Are both academic (calculus_search) and accessible (md_search) perspectives integrated where appropriate?
                    -   Are any relevant URLs from `md_search` included in 'Additional Resources'?
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
                # 플레이스홀더: 실제 사용자 메시지들은 여기 "{messages}"에 삽입되어 연결됨
                ("placeholder", "{messages}"),
            ]
        )

        # create_react_agent를 사용하여 실제 에이전트 인스턴스를 생성
        # - self.llm: 위에서 설정한 ChatGoogleGenerativeAI 모델
        # - tools: 계산 학습 자료 검색 도구(calculus_search, md_search)
        # - state_modifier: 프롬프트 템플릿 (theory_explanation_prompt)
        self.agent = create_react_agent(
            self.llm,
            tools=self.tools,
            state_modifier=self.theory_explanation_prompt
        )
