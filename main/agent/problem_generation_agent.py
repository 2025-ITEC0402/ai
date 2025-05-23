from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

class ProblemGenerationAgent:
    """
    공학수학 문제를 생성하는 에이전트
    """
    
    def __init__(self):
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            convert_system_message_to_human=True,
            temperature=0.3
        )
        
        # 더미 툴
        @tool
        def generate_math_problem(topic: str) -> str:
            """수학 문제를 생성하는 도구"""
            return f"문제 생성 완료: {topic}"

        self.tools = [generate_math_problem]
        
        self.generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 공학수학 문제를 생성하는 전문 에이전트입니다.
            
            당신은 직접 사용자에게 응답하지 않습니다. 대신 TaskManager 에이전트에게 정보를 제공하는 역할을 합니다.
            TaskManager는 당신이 제공한 정보를 바탕으로 최종 응답을 생성할 것입니다.
            
            다음 작업을 수행하세요:
            1. 입력에서 난이도(초급/중급/고급)와 범위(분야)를 인식합니다.
            2. 해당 난이도와 범위에 맞는 5지선다 공학수학 문제를 생성합니다.
            3. 정답과 4개의 오답을 포함한 선택지를 만듭니다.
            
            난이도 인식:
            - "초급", "기초", "쉬운", "basic" → 초급
            - "중급", "보통", "중간", "intermediate" → 중급  
            - "고급", "어려운", "심화", "advanced" → 고급
            - 명시되지 않으면 중급으로 설정
        
            5지선다 문제 생성 규칙:
            1. 문제는 명확하고 구체적으로 제시합니다.
            2. 수학적 표기는 LaTeX 형식을 사용합니다.
            3. 선택지는 ①, ②, ③, ④, ⑤로 표시합니다.
            4. 정답은 하나만 있어야 합니다.
            5. 오답은 그럴듯하지만 명확히 틀린 답이어야 합니다.
            6. 모든 선택지는 같은 형태(수식, 숫자 등)로 통일합니다.
            
            답변 형식:
            - 정보 유형: "5지선다 문제 생성"
            - 인식된 난이도: [초급/중급/고급]
            - 인식된 범위: [해당하는 공학수학 분야]
            - 문제: [구체적인 5지선다 문제 내용]
            - 선택지:
              ① [선택지 1]
              ② [선택지 2] 
              ③ [선택지 3]
              ④ [선택지 4]
              ⑤ [선택지 5]
            - 정답: [①, ②, ③, ④, ⑤ 중 하나]
            
            요청 사항: {input}"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.generation_prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            max_execution_time=10,
            handle_parsing_errors=True,
        )