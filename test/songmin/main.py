"""
EMA (Engineering Mathematics Assistant) 메인 실행 파일
"""
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from graph.workflow import graph
from utils.logger import setup_logger
from langchain_core.runnables import RunnableConfig
import uuid
import warnings
import uvicorn
warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!")

logger = setup_logger("main")
app = FastAPI(title="EMA API", description="Engineering Mathematics Assistant API")

class Query(BaseModel):
    text: str

class Response(BaseModel):
    result: str

def process_query(query: str) -> str:
    """
    사용자 질의를 처리하고 응답을 반환합니다.
    
    Args:
        query (str): 사용자의 질의
        
    Returns:
        str: 시스템의 응답
    """
    logger.info(f"사용자 질의 처리 시작: {query[:50]}...")
    
    config = RunnableConfig(recursion_limit=10, configurable={"thread_id": str(uuid.uuid4())})
    state = {"messages": [HumanMessage(content=query, name="User")]}
    
    try:
        for step in graph.stream(state, config=config):
            logger.debug(f"실행 중인 노드: {step}")
    except Exception as e:
        logger.error(f"워크플로우 실행 중 오류 발생: {str(e)}")
        return f"오류 발생: {str(e)}"
    
    final_state = graph.get_state(config=config)
    messages = final_state.values['messages']
    ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
    
    if not ai_messages:
        return "응답을 생성하지 못했습니다."
    
    final_message = ai_messages[-1].content
    return final_message

@app.get("/")
async def root():
    return {"status": "EMA API is running"}

@app.post("/query", response_model=Response)
async def handle_query(query: Query, background_tasks: BackgroundTasks):
    """
    사용자 질의를 처리하고 응답을 반환합니다.
    """
    if not query.text:
        raise HTTPException(status_code=400, detail="질의 내용이 비어있습니다.")
    
    result = process_query(query.text)
    return Response(result=result)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)