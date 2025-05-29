"""
EMA (Engineering Mathematics Assistant) 메인 실행 파일
"""
import os
import uuid
import warnings
import re, json
import base64
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from workflow import graph
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!")
config = RunnableConfig(recursion_limit=10, configurable={"thread_id": str(uuid.uuid4())})

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
        for step in graph.stream(state, config=config):
            if step:
                node_name = list(step.keys())[0]
                print(f"🔄 실행 중: {node_name}")

        final_state = graph.get_state(config=config)
        messages = final_state.values['messages']

        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        if ai_messages:
            final_message = ai_messages[-1].content
        else:
            final_message = messages[-1].content if messages else "응답을 생성할 수 없습니다."

        return final_message

    except Exception as e:
        return f"오류 발생: {str(e)}"

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

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# 문제생성 api
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
class QuestionRequest(BaseModel):
    topics: str           = Field(..., example="함수의 극한 확인문제")
    range_: str           = Field(..., alias="range", example="2.2 The Limit of Functions")
    summarized: str       = Field(..., example="극한의 정의, 한쪽·무한 극한, 수직 점근선")
    difficulty: str       = Field(..., example="풀이 5줄 이내")
    quiz_examples: str    = Field(..., example="(예시 문제)")

class QuestionResponse(BaseModel):
    question: str
    choice1: str
    choice2: str
    choice3: str
    choice4: str
    choice5: str
    answer: int
    solution: str

@app.post(
    "/questions",
    response_model=QuestionResponse,
    summary="객관식 문제 생성",
    status_code=status.HTTP_201_CREATED,
)
async def create_question(payload: QuestionRequest):
    """
    LangChain + OpenAI 모델을 사용해 객관식 문제를 생성하여 JSON 으로 반환합니다.
    """
    try:
        # 1) payload → JSON string
        payload_str = payload.model_dump_json(by_alias=True)
        # 2) 1차 LLM 요청: “문제 생성”
        query = (
            "다음 JSON을 기반으로, 객관식 문제를 ‘사람이 읽기 편한’ 형식으로 만들어줘.\n"
            f"{payload_str}"
        )
        raw_result: str = process_query(query)
        # 3) 2차 LLM 요청: “JSON으로 변환”
        json_prompt = (
            "위에서 생성된 문제를, 아래 스키마에 맞춰 **순수 JSON**으로만 변환해줘.\n"
            "키: question, choice1~choice5, answer(정답 번호, 1~5), solution\n\n"
            f"{raw_result}"
        )
        response = await llm.agenerate([[HumanMessage(content=json_prompt)]])
        json_str = response.generations[0][0].text.strip()
        # 3-4) JSON → Pydantic 모델
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", json_str, re.S)
        clean_json = match.group(1) if match else json_str
        data = json.loads(clean_json)
        return QuestionResponse(**data)

    except Exception as e:
        # 필요 시 세분화된 예외 처리
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
        • 출력 형식은 **제목** 텍스트만, 따옴표나 추가 설명 없이 제공해야 합니다.

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

        for step in graph.stream(state, config=config):
            if step:
                node_name = list(step.keys())[0]
                print(f"🔄 실행 중: {node_name}")

        final_state = graph.get_state(config=config)
        messages = final_state.values['messages']

        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        if ai_messages:
            final_message = ai_messages[-1].content
        else:
            final_message = messages[-1].content if messages else "응답을 생성할 수 없습니다."

        return QAResponse(answer=final_message)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="debug")
