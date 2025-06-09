"""
EMA (Engineering Mathematics Assistant) 메인 실행 파일
"""
import os
import warnings
import re, json
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from workflow import graph
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import traceback 
warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!")
config = RunnableConfig(recursion_limit=10)

def process_query(query: str) -> str:
    """
    사용자 질의를 처리하고 응답을 반환합니다.

    Args:
        query (str): 사용자의 질의

    Returns:
        str: 시스템의 응답
    """
    print(f"사용자 질의 처리 시작: {query}")

    state = {"messages": [HumanMessage(content=query, name="User")]}
    
    try:
        final_state = graph.invoke(state, config=config)
        messages = final_state['messages']
        final_message = messages[-1].content if messages else "응답을 생성할 수 없습니다."

        return final_message

    except Exception as e:
        print(f"오류 발생: {e}")
        print(f"오류 타입: {type(e)}")
        print(f"오류 상세 정보 (Traceback): \n{traceback.format_exc()}")
        return

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    convert_system_message_to_human=True,
    temperature=0.2
)
app = FastAPI(
    title="Calc-Question Generator API",
    description="미적분 객관식 문제를 생성하는 REST API",
    version="1.0.0",
)
@app.get("/")
async def root():
    return {"message": "EMA Backend API"}

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# new 문제생성 api
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
class NewQuestionRequest(BaseModel):
    topics: str           = Field(..., example="함수와 모델 (Functions and Models)")
    range_: str           = Field(..., alias="range", example="1")
    summarized: str       = Field(..., example="공학수학의 기초 개념인 함수, 극한, 미분을 먼저 학습하여 기본기를 다져야 합니다.")
    difficulty: str       = Field(..., example="Normal")
    quiz_examples: str    = Field(..., example="(예시 문제)")

class NewQuestionResponse(BaseModel):
    chapter : str
    question: str
    choice1: str
    choice2: str
    choice3: str
    choice4: str
    answer: int
    solution: str
    difficulty: str
    ai_summary: str

@app.post(
    "/newquestions",
    response_model=NewQuestionResponse,
    summary="객관식 문제 생성",
    status_code=status.HTTP_201_CREATED,
)
async def create_question(payload: NewQuestionRequest):
    """
    LangChain + OpenAI 모델을 사용해 객관식 문제를 생성하여 JSON 으로 반환합니다.
    """
    try:
        # 1) payload → JSON string
        payload_str = payload.model_dump_json(by_alias=True)
        # 2) 1차 LLM 요청: “문제 생성”
        query = (
            "다음 JSON을 기반으로 객관식 문제를 만들어줘. 그리고 그 문제를 풀어줘.\n"
            "summarized에서 사용자를 위한 요구 목표를 반영해줘"
            f"{payload_str}"
        )
        raw_result: str = process_query(query)
        print("***** raw_result = ", raw_result)
        # 3) 2차 LLM 요청: “JSON으로 변환”
        json_prompt = f"""
        당신은 교육용 문제 데이터를 JSON으로 포맷팅하는 전문가입니다.
        아래 입력(input) 문자열에 포함된 모든 정보를 사용하여, 반드시 다음 스키마에 맞는 JSON을 생성하세요.
        
        --- 스키마 설명 (NewQuestionResponse) ---
        - chapter    : 문자열 (예: "함수와 모델 (Functions and Models)", "적분론" 등)
        - question   : 문자열 (문제 지문)
        - choice1    : 문자열 (선택지 1)
        - choice2    : 문자열 (선택지 2)
        - choice3    : 문자열 (선택지 3)
        - choice4    : 문자열 (선택지 4)
        - answer     : 정수 (1~4 중 하나; 정답이 몇 번 선택지인지)
        - solution   : 문자열 (문제 풀이 과정이나 해설)
        - difficulty : 문자열(EASY,NORMAL,HARD 중 하나)
        - ai_summary : 문자열(문제 한줄평)

        ※ 주의사항
        1. **answer** 필드는 반드시 1, 2, 3, 4 중 하나여야 합니다.
        2. **difficulty** 필드는 반드시 1, 2, 3 중 하나여야 합니다.
        3. 출력은 JSON 형식으로만 이루어져야 하고, 다른 텍스트나 주석을 포함하면 안 됩니다.
        4. 하나의 JSON 객체만 출력하세요.
        
        --- input 문자열 ---
        {raw_result}
        
        --- JSON 예시 ---
        ```json
        {{
          "chapter": "함수와 모델 (Functions and Models)",
          "question": "함수 f(x)=x^2+2x+1의 극값을 구하시오.",
          "choice1": "x=-1",
          "choice2": "x=0",
          "choice3": "x=1",
          "choice4": "x=2",
          "answer": 1,
          "solution": "도함수 f'(x)=2x+2. f'(x)=0 ⇒ x=-1이 극값이고, f''(x)=2>0이므로 최솟값이다.",
          "difficulty": "Normal"
          "ai_summary": "기본적인 도함수 계산과 극값 판별을 묻는 문제입니다."
          ""
        }}
        ```
        위 예시에서는
        
        "answer": 1 (선택지 1이 정답)
        
        "difficulty": "Normal" (보통 난이도)
        
        위 input 문자열 텍스트를 바탕으로, 동일한 형식의 JSON 객체를 출력해 주세요. 출력 시 따옴표(“)와 이스케이프 문자(\)를 정확히 지켜야 하며, 불필요한 설명은 포함하지 말고 오로지 JSON 객체만 반환하시기 바랍니다.
        """
        structured_llm = llm.with_structured_output(NewQuestionResponse)
        response = await structured_llm.ainvoke([HumanMessage(content=json_prompt)])
        return response

    except Exception as e:
        print(f"오류 발생: {e}")
        print(f"오류 타입: {type(e)}")
        print(f"오류 상세 정보 (Traceback): \n{traceback.format_exc()}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# 질의응답 api
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
class QARequest(BaseModel):
    query: str = Field(..., example="함수의 극한이란 무엇인가요?")

class QAResponse(BaseModel):
    answer: str

@app.post(
    "/qna",
    response_model=QAResponse,
    summary="사용자 질의응답",
    status_code=status.HTTP_200_OK,
)
async def answer_query(payload: QARequest):
    """
    사용자로부터 받은 질의를 AI 그래프에 전달해 답변을 생성합니다.
    """
    try:
        result = process_query(payload.query)
        return QAResponse(answer=result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# 질의응답 + 제목 api
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
class QATResponse(BaseModel):
    answer: str
    title: str

@app.post(
    "/qnantitle",
    response_model=QATResponse,
    summary="사용자 질의응답",
    status_code=status.HTTP_200_OK,
)
async def answer_query(payload: QARequest):
    """
    사용자로부터 받은 질의를 AI 그래프에 전달해 답변을 생성합니다.
    """
    try:
        result = process_query(payload.query)
        prompt = f"""System:
        당신은 ‘채팅방 제목 생성기(Chat Title Generator)’입니다.
        사용자가 보낸 메시지를 입력으로 받아, 그 메시지의 핵심 주제를 3~6개의 단어로 요약한 짧고 명확한 제목을 출력하세요.
        • 제목에는 불필요한 조사나 접속사를 쓰지 마세요.
        • 구체적인 키워드를 포함해 대화 내용을 한눈에 알 수 있게 작성하세요.
        • 출력 형식은 제목 텍스트만, 따옴표나 볼드체 등의 스타일, 추가 설명 없이 제공해야 합니다.

        User:
        {payload.query}"""


        response = await llm.agenerate([[HumanMessage(content=prompt)]])
        tit = response.generations[0][0].text.strip()
        print("***** tit = ", tit)

        return QATResponse(answer=result, title=tit)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# 질의응답 이미지 + 문자 api
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
class QAImageRequest(BaseModel):
    query: str = Field(..., example="함수의 극한이란 무엇인가요?")
    image_base64: str = Field(
        ...,
        description="data:image/png;base64, 로 시작하는 Base64 인코딩 이미지 문자열",
        example="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
    )

@app.post(
    "/qnaimg",
    response_model=QAResponse,
    summary="사용자 질의응답",
    status_code=status.HTTP_200_OK,
)
async def answer_query(payload: QAImageRequest):
    try:


        # 3) 질의 + 이미지 데이터를 그래프 상태에 삽입
        state = {
            "messages": [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": ""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": payload.image_base64
                            }
                        }
                    ],
                    name="User"
                )
            ]
        }

        final_state = graph.invoke(state, config=config)
        messages = final_state['messages']
        final_message = messages[-1].content if messages else "응답을 생성할 수 없습니다."
        return QAResponse(answer=final_message)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app",host="0.0.0.0", port=8000, reload=True, log_level="debug")
