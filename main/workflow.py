from typing import TypedDict, Sequence, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from agent.external_search_agent import ExternalSearchAgent
from agent.problem_solving_agent import ProblemSolvingAgent
from agent.problem_generation_agent import ProblemGenerationAgent
from agent.quality_evaluation_agent import QualityEvaluationAgent
from agent.task_manager import TaskManager
from functools import partial

search_agent = ExternalSearchAgent()
solving_agent = ProblemSolvingAgent()
generating_agent = ProblemGenerationAgent()
quality_agent = QualityEvaluationAgent()
TaskManager = TaskManager()
members = ["ExternalSearch", "ProblemSolving", "ProblemGeneration", "QualityEvaluation"]

def agent_node(state, agent, name):
    agent_response = agent.invoke(state)
    msg = HumanMessage(content=agent_response["messages"][-1], name=name)
    return {"messages": [msg]}

search_node = partial(agent_node, agent=search_agent, name="ExternalSearch")
solving_node = partial(agent_node, agent=solving_agent, name="ProblemSolving")
generating_node = partial(agent_node, agent=generating_agent, name="ProblemGeneration")
quality_evaluation_node = partial(agent_node, agent=quality_agent, name="QualityEvaluation")
TaskManager_node = partial(agent_node, agent=TaskManager, name="TaskManager")
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage], operator.add]
    next: str

workflow = StateGraph(AgentState)

workflow.add_node("ExternalSearch", search_node)
workflow.add_node("ProblemSolving", solving_node)
workflow.add_node("ProblemGeneration", generating_node)
workflow.add_node("QualityEvaluation", quality_evaluation_node)

for m in members:
    workflow.add_edge(m, "TaskManager")

workflow.add_edge(START, "TaskManager")

conditional_map = {name: name for name in members}
conditional_map["FINISH"] = END

def get_next(state):
    
    return state["next"]

workflow.add_conditional_edges("TaskManager", get_next, conditional_map)
workflow.add_edge(START, "TaskManager")
graph = workflow.compile(checkpointer=MemorySaver())


