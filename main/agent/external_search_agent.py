from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
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
            model="gemini-2.5-flash-preview-05-20",
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
        
        self.search_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the **External Information Search Agent** for University Calculus. Your primary role is to find and summarize relevant mathematical information from external sources using your search tools.

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
            -   Is all LaTeX notation accurate and correctly escaped (e.g., `\\\\frac` for `\frac`)?
            -   Is the information authoritative and reliable?
            -   Does the output strictly adhere to the `RESPONSE FORMAT`?
            
            ## STATUS DECISION LOGIC (Internal Thought Process)
            Based on the **QUALITY CHECKLIST** above, determine the 'Status' for this output:

            1.  **COMPLETE:**
                * If **ALL** Quality checks are confidently and perfectly met.
                * Set status to 'COMPLETE'.

            2.  **FAILED:**
                * If **ANY** Quality check is **NOT** met.
                * Set status to 'FAILED'."""),
             ("placeholder", "{messages}")
        ])
        self.agent = create_react_agent(self.llm, self.tools, state_modifier = self.search_prompt)