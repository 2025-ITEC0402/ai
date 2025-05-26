"""
EMA (Engineering Mathematics Assistant) 메인 실행 파일
"""
import os
import uuid
import warnings
import re, json
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from workflow import graph
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!")
warnings.filterwarnings("ignore", message=".*get_relevant_documents.*")
# 설정
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
        print("***** payload_str = ", payload_str)
        # 2) 1차 LLM 요청: “문제 생성”
        query = (
            "다음 JSON을 기반으로, 객관식 문제를 ‘사람이 읽기 편한’ 형식으로 만들어줘.\n"
            f"{payload_str}"
        )
        print("***** query = ", query)
        raw_result: str = process_query(query)
        print("***** raw_result = ", raw_result)
        # 3) 2차 LLM 요청: “JSON으로 변환”
        json_prompt = (
            "위에서 생성된 문제를, 아래 스키마에 맞춰 **순수 JSON**으로만 변환해줘.\n"
            "키: question, choice1~choice5, answer(정답 번호, 1~5), solution\n\n"
            f"{raw_result}"
        )
        response = await llm.agenerate([[HumanMessage(content=json_prompt)]])
        print("***** response = ", response)
        json_str = response.generations[0][0].text.strip()
        print("***** json_str = ", json_str)
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



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="debug")
