from agents.problem_solving import ProblemSolvingAgent
from utils.logger import setup_logger

# 로거 설정
logger = setup_logger("main")

def main():
    """
    메인 함수: 하드코딩된 문제와 주제를 사용하여 문제 풀이 에이전트를 실행합니다.
    """
    problem_solver = ProblemSolvingAgent()
    
    query = "미분방정식 dy/dx + 2y = e^x를 풀어라"
    topic = "미분방정식"
    
    print(f"문제: {query}")
    print(f"주제: {topic}")
    
    try:
        logger.info(f"문제 풀이 요청: '{query}', 주제: '{topic}'")
        solution = problem_solver.solve_problem(query, topic)
        print("\n===== 문제 풀이 =====")
        print(solution)
        print("====================\n")
    except Exception as e:
        logger.error(f"문제 풀이 중 오류 발생: {str(e)}")
        print(f"오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main()