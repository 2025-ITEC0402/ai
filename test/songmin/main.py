"""
EMA (Engineering Mathematics Assistant) 메인 실행 파일
"""
import os
import sys
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
from graph.workflow import graph
from utils.logger import setup_logger
from langchain_core.runnables import RunnableConfig


# 로거 설정
logger = setup_logger("main")
config = RunnableConfig(recursion_limit=10, configurable={"thread_id": "abc123"})

def process_query(query: str) -> str:
    """
    사용자 질의를 처리하고 응답을 반환합니다.
    
    Args:
        query (str): 사용자의 질의
        
    Returns:
        str: 시스템의 응답
    """
    logger.info(f"사용자 질의 처리 시작: {query[:50]}...")
    
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
    final_message = ai_messages[-1].content

    return final_message
def main():
    query = "1+2는?"
    response = process_query(query)
    print("\n" + "=" * 50)
    print("EMA의 응답:")
    print(response)
    print("=" * 50)

if __name__ == "__main__":
    main()


