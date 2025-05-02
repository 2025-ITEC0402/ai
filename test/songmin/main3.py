from agents.problem_solving import ProblemSolvingAgent
from agents.external_search import ExternalSearchAgent
from utils.logger import setup_logger


logger = setup_logger("main")

def main():
    
    '''problem_solver = ProblemSolvingAgent()
    
    solving_query = "미분방정식 dy/dx + 2y = e^x를 풀어라"
    solving_topic = "미분방정식"
    
    print("\n===== 문제 풀이 테스트 =====")
    print(f"문제: {solving_query}")
    print(f"주제: {solving_topic}")
    
    try:
        logger.info(f"문제 풀이 요청: '{solving_query}', 주제: '{solving_topic}'")
        solution = problem_solver.solve_problem(solving_query, solving_topic)
        print("\n----- 문제 풀이 결과 -----")
        print(solution)
        print("-------------------------\n")
    except Exception as e:
        logger.error(f"문제 풀이 중 오류 발생: {str(e)}")
        print(f"오류가 발생했습니다: {str(e)}")'''
    
    external_searcher = ExternalSearchAgent()
    
    search_query = "편미분에 대해서 알려줘"
    search_topic = "편미분"
    
    print("\n===== 외부 검색 테스트 =====")
    print(f"검색 질의: {search_query}")
    print(f"주제: {search_topic}")
    
    try:
        logger.info(f"외부 검색 요청: '{search_query}', 주제: '{search_topic}'")
        search_result = external_searcher.search_and_summarize(search_query, search_topic)
        print("\n----- 검색 결과 요약 -----")
        print(search_result)
        print("-------------------------\n")
    except Exception as e:
        logger.error(f"외부 검색 중 오류 발생: {str(e)}")
        print(f"오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main()