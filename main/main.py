from agent.external_search_agent import ExternalSearchAgent
from agent.problem_solving_agent import ProblemSolvingAgent
from agent.quality_evaluation_agent import QualityEvaluationAgent
from agent.problem_generation_agent import ProblemGenerationAgent
from agent.explain_theory_agent import ExplainTheoryAgent
import warnings
warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!")

def main(choice):
    
    if choice == 1:
        # 외부 검색 에이전트
        agent = ExternalSearchAgent().agent
        input = "미분의 기본 정리에 대해 설명해주세요"
        result = agent.invoke({"messages": input})
    elif choice == 2:
        # 문제 해결 에이전트
        agent = ProblemSolvingAgent().agent
        input = "x + y의 x에 대한 편미분"
        result = agent.invoke({"messages": input})
        
    elif choice == 3:
        # 품질 평가 에이전트
        agent = QualityEvaluationAgent().agent
        input = """{'query': 'x + y의 x에 대한 편미분', 'output': '```json\n{\n"정보 유형": "문제 풀이",\n"문제 요약": "주어진 함수 \\(f(x, y) = x + y\\)에 대해 x에 대한 편미분을 구하는
            문제입니다.",\n"접근 방법": "편미분의 정의에 따라, y를 상수로 취급하고 x에 대해 미분합니다.",\n"단계별 풀이": [\n{\n"단계": 1,\n"설명": "함수 \\(f(x, y) = x + y\\)를
            x에 대해 편미분합니다. 여기서 y는 상수항으로 취급됩니다.",\n"수식": "\\(\\frac{\\partial}{\\partial x}(x + y)\\)"\n},{\n"단계": 2,\n"설명": "x에 대한 미분과 y에 대한
            미분을 분리합니다.",\n"수식": "\\(\\frac{\\partial x}{\\partial x} + \\frac{\\partial y}{\\partial x}\\)"\n},\n{\n"단계": 3,\n"설명": "\\(\\frac{\\partial x}{\\partial x} = 1\\) 이고, y는 상수이므로 \\(\\frac{\\partial y}{\\partial x} = 0\\) 입니다.",\n"수식": "\\(1 + 0\\)"\n},\n{\n"단계": 4,\n"설명": "결과를 계산합니다.",\n"수식
            ": "\\(1\\)"\n}\n],\n"최종 답안": "x + y의 x에 대한 편미분은 1입니다.",\n"추가 설명": "편미분의 기본적인 정의에 따라 간단하게 풀 수 있는 문제입니다. 다른 접근 방법은 
            필요하지 않습니다."\n}\n```'}"""
        result = agent.invoke({"messages": input})

    elif choice == 4:
        # 문제 생성 에이전트
        agent = ProblemGenerationAgent().agent
        input = "중급 편미분 문제"
        result = agent.invoke({"messages": input})

    elif choice == 5:
        # 이론 설명 에이전트
        agent = ExplainTheoryAgent().agent
        input = "편미분 하는 방법을 모르겠어"
        result = agent.invoke({"messages": input})


    print("결과:")
    print(result)

if __name__ == "__main__":
    main(5)