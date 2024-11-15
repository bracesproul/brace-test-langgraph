from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage

from langgraph.graph import StateGraph, add_messages

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

sub_subgraph_builder = StateGraph(AgentState)

def call_model_sub_subgraph(state):
    return {
        "messages": [{ "role": "user", "content": "output from call_model_subgraph" }]
    }

def sub_random_passthrough_node(state):
    return {
        "messages": [{ "role": "user", "content": "output from random_passthrough_node" }]
    }

sub_subgraph_builder.add_node("sub_subgraph_call_model", call_model_sub_subgraph)
sub_subgraph_builder.add_node("sub_subgraph_passthrough", sub_random_passthrough_node)

sub_subgraph_builder.set_entry_point("sub_subgraph_call_model")
sub_subgraph_builder.add_edge("sub_subgraph_call_model", "sub_subgraph_passthrough")
sub_subgraph_builder.set_finish_point("sub_subgraph_passthrough")

sub_subgraph = sub_subgraph_builder.compile()


subgraph_builder = StateGraph(AgentState)

def call_model_subgraph(state):
    return {
        "messages": [{ "role": "user", "content": "output from call_model_subgraph" }]
    }


subgraph_builder.add_node("subgraph_call_model", call_model_subgraph)
subgraph_builder.add_node("invoke_sub_subgraph", sub_subgraph)

subgraph_builder.set_entry_point("subgraph_call_model")
subgraph_builder.add_edge("subgraph_call_model", "invoke_sub_subgraph")
subgraph_builder.set_finish_point("invoke_sub_subgraph")

subgraph = subgraph_builder.compile()

def call_model(state):
    return {
        "messages": [{ "role": "user", "content": "output from call_model" }]
    }


# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent_node", call_model)
workflow.add_node("invoke_subgraph", subgraph)

workflow.add_edge("agent_node", "invoke_subgraph")

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent_node")

workflow.set_finish_point("invoke_subgraph")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile()
