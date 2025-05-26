from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from question_generator import QuestionGenerator
from dotenv import load_dotenv
import os

# ───────────────────────────────
# 준비
# ───────────────────────────────
load_dotenv()                            # .env 에서 OPENAI_API_KEY 로딩
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY 가 설정되지 않았습니다.")

app = FastAPI(
    title="Calc-Question Generator API",
    description="미적분 객관식 문제를 생성하는 REST API",
    version="1.0.0",
)

generator = QuestionGenerator()          # 싱글턴으로 재사용

# ───────────────────────────────
# Pydantic 모델
# ───────────────────────────────
class QuestionRequest(BaseModel):
    topics: str           = Field(..., example="함수의 극한 확인문제")
    range_: str           = Field(..., alias="range", example="2.2 The Limit of Functions")
    summarized: str       = Field(..., example="극한의 정의, 한쪽·무한 극한, 수직 점근선")
    difficulty: str       = Field(..., example="풀이 5줄 이내")
    quiz_examples: str    = Field(..., example="(예시 문제)")

    class Config:
        allow_population_by_field_name = True   # range_ ↔ range 매핑 허용


class QuestionResponse(BaseModel):
    question: str
    choice1: str
    choice2: str
    choice3: str
    choice4: str
    choice5: str


# ───────────────────────────────
# 엔드포인트
# ───────────────────────────────
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
        result = generator.generate_question(
            topics=payload.topics,
            range_=payload.range_,
            summarized=payload.summarized,
            difficulty=payload.difficulty,
            quiz_examples=payload.quiz_examples,
        )
        return result
    except Exception as e:
        # 필요 시 세분화된 예외 처리
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get("/health", summary="헬스 체크")
async def health_check():
    return {"status": "ok"}
