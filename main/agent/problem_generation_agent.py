from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
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
            (
                "system",
                """당신은 공학수학 문제를 생성하는 전문 에이전트입니다.
        
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
        
        종료 조건:
        8. 작업이 완료되면 반드시 “Final Answer:” 키워드로 시작하여
           생성된 5지선다 문제와 정답을 아래 형식으로 작성하세요.
           - 정보 유형: "5지선다 문제 생성"
           - 인식된 난이도: [초급/중급/고급]
           - 인식된 범위: [해당하는 공학수학 분야]
           - 문제: [생성된 5지선다 문제]
           - 선택지: 
             ① …
             ② …
             ③ …
             ④ …
             ⑤ …
           - 정답: [①, ②, ③, ④, ⑤ 중 하나]
        9. 더 이상 도구를 호출하지 마세요.
        10. 마지막 줄에 “[END]” 토큰을 추가하여 종료를 명시하세요.
        
        요청 사항: {input}"""
            ),
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
            prompt_value = self.generation_prompt.format_prompt(
                input=query,
                agent_scratchpad=scratch
            )

            # (디버깅)
            print("**")
            return prompt_value.to_messages()
        
        self.agent = create_react_agent(self.llm, tools=self.tools, state_modifier=_modify_state)