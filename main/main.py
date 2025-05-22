from agent.external_search_agent import ExternalSearchAgent
import warnings
warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!")
def main():
    agent = ExternalSearchAgent()
    query = "미분의 기본 정리에 대해 설명해주세요"
    result = agent.agent_executor.invoke({"query": query})
    print("결과:")
    print(result)

if __name__ == "__main__":
    main()