from typing import TypedDict, Sequence, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from agents.task_manager import TaskManagerAgent, members
from agents.external_search import ExternalSearchAgent
from agents.problem_solving import ProblemSolvingAgent
from agents.problem_generation import ProblemGenerationAgent
from agents.quality_evaluation import QualityEvaluationAgent
from functools import partial

search_agent = ExternalSearchAgent()
solving_agent = ProblemSolvingAgent()
generating_agent = ProblemGenerationAgent()
quality_agent = QualityEvaluationAgent()
task_manager = TaskManagerAgent()

def task_manager_node(state):
    return task_manager.task_manage(state)

def external_search_node(state):
    query = state.get("query", "")
    if not query and state.get("messages"):
        for msg in reversed(state.get("messages", [])):
            if not hasattr(msg, "name") or msg.name != "ExternalSearch":
                query = msg.content
                break
    
    answer = search_agent.search_and_summarize(query)
    msg = HumanMessage(content=answer, name="ExternalSearch")
    return {"messages": [msg]}

def problem_solving_node(state):
    query = state.get("query", "")
    if not query and state.get("messages"):
        for msg in reversed(state.get("messages", [])):
            if not hasattr(msg, "name") or msg.name != "ProblemSolving":
                query = msg.content
                break
    
    answer = solving_agent.solve_problem(query)
    msg = HumanMessage(content=answer, name="ProblemSolving")
    return {"messages": [msg]}

def problem_generation_node(state):
    query = state.get("query", "")
    if not query and state.get("messages"):
        for msg in reversed(state.get("messages", [])):
            if not hasattr(msg, "name") or msg.name != "ProblemGeneration":
                query = msg.content
                break
    
    problem = generating_agent.generate_problem(query)
    msg = HumanMessage(content=problem, name="ProblemGeneration")
    return {"messages": [msg]}

def quality_agent_node(state):
    query = state.get("query", "")
    if not query and state.get("messages"):
        for msg in reversed(state.get("messages", [])):
            if not hasattr(msg, "name") or msg.name != "QualityEvaluation":
                query = msg.content
                break
    
    evaluation_result = quality_agent.evaluate_quality(query)
    msg = HumanMessage(content=str(evaluation_result), name="QualityEvaluation")
    return {"messages": [msg], "next": evaluation_result.get("next")}


class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage], operator.add]
    next: str
    query: str

workflow = StateGraph(AgentState)

workflow.add_node("TaskManager", task_manager_node)
workflow.add_node("ExternalSearch", external_search_node)
workflow.add_node("ProblemSolving", problem_solving_node)
workflow.add_node("ProblemGeneration", problem_generation_node)
workflow.add_node("QualityEvaluation", quality_agent_node)

members.append("QualityEvaluation")
for m in members:
    workflow.add_edge(m, "TaskManager")

workflow.add_edge(START, "TaskManager")

conditional_map = {name: name for name in members}
conditional_map["sufficient"] = "TaskManager"
conditional_map["FINISH"] = END

def get_next(state):
    return state["next"]

workflow.add_conditional_edges("TaskManager", get_next, conditional_map)

graph = workflow.compile(checkpointer=MemorySaver())


