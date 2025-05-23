from agent.external_search_agent import ExternalSearchAgent
from agent.problem_solving_agent import ProblemSolvingAgent
from agent.quality_evaluation_agent import QualityEvaluationAgent
from agent.problem_generation_agent import ProblemGenerationAgent
import warnings
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
import uuid

warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!")
config = RunnableConfig(recursion_limit=10, configurable={"thread_id": str(uuid.uuid4())})
def main(choice: int):
    """개별 에이전트 테스트"""
    try:
        if choice == 1:
            agent = ExternalSearchAgent()
            input_text = "미분의 기본 정리에 대해 설명해주세요"
            print(f"입력: {input_text}")
            
            messages = [HumanMessage(content=input_text)]
            result = agent.agent.invoke({"messages": messages}, config=config)
            
        elif choice == 2:
            agent = ProblemSolvingAgent()
            input_text = "x + y의 x에 대한 편미분"
            print(f"입력: {input_text}")
            
            messages = [HumanMessage(content=input_text)]
            result = agent.agent.invoke({"messages": messages}, config=config)
            
        elif choice == 3:
            agent = QualityEvaluationAgent()
            input_text = """{'query': 'x + y의 x에 대한 편미분', 'output': '```json\n{\n"정보 유형": "문제 풀이",\n"문제 요약": "주어진 함수 \\(f(x, y) = x + y\\)에 대해 x에 대한 편미분을 구하는
            문제입니다.",\n"접근 방법": "편미분의 정의에 따라, y를 상수로 취급하고 x에 대해 미분합니다.",\n"단계별 풀이": [\n{\n"단계": 1,\n"설명": "함수 \\(f(x, y) = x + y\\)를
            x에 대해 편미분합니다. 여기서 y는 상수항으로 취급됩니다.",\n"수식": "\\(\\frac{\\partial}{\\partial x}(x + y)\\)"\n},{\n"단계": 2,\n"설명": "x에 대한 미분과 y에 대한
            미분을 분리합니다.",\n"수식": "\\(\\frac{\\partial x}{\\partial x} + \\frac{\\partial y}{\\partial x}\\)"\n},\n{\n"단계": 3,\n"설명": "\\(\\frac{\\partial x}{\\partial x} = 1\\) 이고, y는 상수이므로 \\(\\frac{\\partial y}{\\partial x} = 0\\) 입니다.",\n"수식": "\\(1 + 0\\)"\n},\n{\n"단계": 4,\n"설명": "결과를 계산합니다.",\n"수식
            ": "\\(1\\)"\n}\n],\n"최종 답안": "x + y의 x에 대한 편미분은 1입니다.",\n"추가 설명": "편미분의 기본적인 정의에 따라 간단하게 풀 수 있는 문제입니다. 다른 접근 방법은 
            필요하지 않습니다."\n}\n```'}"""
            print(f"입력: {input_text[:100]}...")
            
            messages = [HumanMessage(content=input_text)]
            result = agent.agent.invoke({"messages": messages}, config=config)
            
        elif choice == 4:
            agent = ProblemGenerationAgent()
            input_text = "중급 편미분 문제"
            print(f"입력: {input_text}")
            
            messages = [HumanMessage(content=input_text)]
            result = agent.agent.invoke({"messages": messages}, config=config)
        print("\n결과:")
        print("-" * 60)

        if isinstance(result, dict) and 'messages' in result:
            # 메시지 결과에서 내용 추출
            if result['messages']:
                last_message = result['messages'][-1]
                print(last_message.content)
            else:
                print("응답 메시지가 없습니다.")
        else:
            print(result)
            
    except Exception as e:
        print(f"❌ 에이전트 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main(2)
    