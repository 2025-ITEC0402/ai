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
            model="gemini-2.5-flash-preview-05-20",
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
                "Search academic calculus textbooks (ENGLISH CONTENT) for authoritative mathematical content. "
                "This database contains English textbooks - always use ENGLISH queries for best results. "
                "Use this tool to find rigorous definitions, theorems, proofs, examples, formulas, and applications. "
                "Returns relevant textbook sections with chapter and section metadata for accurate, scholarly explanations."
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
                "Search user-friendly markdown calculus learning guides (KOREAN CONTENT) for accessible explanations. "
                "This database contains Korean markdown guides - always use KOREAN queries for best results. "
                "Use this tool to find simplified summaries, study guides, and educational content with URLs. "
                "Returns markdown sections with static page URLs for additional learning resources."
            )
        )

        # 4) tools 리스트에 이 래퍼만 추가
        self.tools = [self.cal_tool, self.md_tool]

        self.theory_explanation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", """You are the **Calculus Theory Explanation Agent**, a specialized component within a multi-agent AI system. 
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
                    [Detailed definitions, theorems, fundamental formulas, and rigorous explanations. Use precise mathematical language and include ALL equations/expressions in LaTeX.

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
                        * Set status to 'FAILED'."""),
                ("placeholder", "{messages}"),
            ]
        )

        self.agent = create_react_agent(self.llm, tools=self.tools, state_modifier=self.theory_explanation_prompt)