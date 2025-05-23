"""
EMA (Engineering Mathematics Assistant) 메인 실행 파일
"""
import os
import sys
import uuid
import warnings
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from workflow import graph
from langchain_core.runnables import RunnableConfig

warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!")

# 설정
config = RunnableConfig(
    recursion_limit=10, 
    configurable={"thread_id": str(uuid.uuid4())}
)

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

def main():
    """메인 실행 함수"""
    
    # 테스트 케이스들
    query = "편미분 문제 하나 만들어주세요"
    response = process_query(query)
    
    print("\n" + "=" * 50)
    print("EMA의 응답:")
    print("=" * 50)
    print(response)
    print("=" * 50)

if __name__ == "__main__":
    main()