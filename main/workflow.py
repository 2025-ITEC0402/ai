from typing import TypedDict, Sequence, Annotated
import operator
import time
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from agent.external_search_agent import ExternalSearchAgent
from agent.problem_solving_agent import ProblemSolvingAgent
from agent.problem_generation_agent import ProblemGenerationAgent
from agent.response_generation_agent import ResponseGenerationAgent
from agent.explain_theory_agent import ExplainTheoryAgent
from agent.task_manager import TaskManager
from functools import partial

search_agent = ExternalSearchAgent()
solving_agent = ProblemSolvingAgent()
generating_agent = ProblemGenerationAgent()
response_agent = ResponseGenerationAgent()
explain_theory_agent = ExplainTheoryAgent()
Task_Manager = TaskManager()

members = ["ExternalSearch", "ProblemSolving", "ProblemGeneration", "ExplainTheoryAgent"]

def supervisor_agent(state):
    start_time = time.time()
    result = Task_Manager.agent.invoke(state)
    end_time = time.time()
    
    print(f"TaskManager: {end_time - start_time:.3f}초 → {result.next} 선택")
    return result

def agent_node(state, agent, name):
    start_time = time.time()
    agent_response = agent.agent.invoke(state)
    end_time = time.time()
    print(f"{name}: {end_time - start_time:.3f}초")
    
    msg = HumanMessage(content=agent_response["messages"][-1].content, name=name)
    return {"messages": [msg]}

search_node = partial(agent_node, agent=search_agent, name="ExternalSearch")
solving_node = partial(agent_node, agent=solving_agent, name="ProblemSolving")
generating_node = partial(agent_node, agent=generating_agent, name="ProblemGeneration")
explain_node = partial(agent_node, agent=explain_theory_agent, name="ExplainTheoryAgent")
response_node = partial(agent_node, agent=response_agent, name="GeneratingResponse")

class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage], operator.add]
    next: str

workflow = StateGraph(AgentState)

workflow.add_node("ExternalSearch", search_node)
workflow.add_node("ProblemSolving", solving_node)
workflow.add_node("ProblemGeneration", generating_node)
workflow.add_node("GeneratingResponse", response_node)
workflow.add_node("ExplainTheoryAgent", explain_node)
workflow.add_node("TaskManager", supervisor_agent)

for m in members:
    workflow.add_edge(m, "TaskManager")

members.append("GeneratingResponse")
conditional_map = {name: name for name in members}

def get_next(state):
    return state["next"]

workflow.add_conditional_edges("TaskManager", get_next, conditional_map)
workflow.add_edge(START, "TaskManager")
workflow.add_edge("GeneratingResponse", END)

original_graph = workflow.compile()

class TimedGraph:
    def __init__(self, graph):
        self.graph = graph
    
    def invoke(self, state, config=None):
        print(f"\n 처리 시작: {state['messages'][0].content[:50]}...")
        print("="*60)
        
        start_time = time.time()
        result = self.graph.invoke(state, config)
        end_time = time.time()
        
        print("="*60)
        print(f" 총 처리시간: {end_time - start_time:.3f}초")
        print("="*60)
        
        return result

graph = TimedGraph(original_graph)