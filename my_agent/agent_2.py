from typing import TypedDict, Literal, Any

from langgraph.graph import StateGraph, END, START
from my_agent.utils.nodes import call_model
from my_agent.utils.state import AgentState
from my_agent.utils.tools import tools
from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0, model_name="gpt-4o")
model = model.bind_tools(tools)

class SubgraphOutputSchema(TypedDict):
    subgraph_output_field: str
    messages: Any


def call_model_subgraph(state):
    response = model.invoke(state['messages'])

    return { "subgraph_output_field": str(response) }

def random_passthrough_node(state):
    return {
        "messages": state["messages"]
    }

subgraph_builder = StateGraph(AgentState, output=SubgraphOutputSchema)

subgraph_builder.add_node("subgraph_call_model", call_model_subgraph)
subgraph_builder.add_node("subgraph_passthrough", random_passthrough_node)

subgraph_builder.set_entry_point("subgraph_call_model")
subgraph_builder.add_edge("subgraph_call_model", "subgraph_passthrough")
subgraph_builder.set_finish_point("subgraph_passthrough")

subgraph = subgraph_builder.compile()

class OutputSchema(TypedDict):
    messages_output_field: str

# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]

def call_model(state):
    """Invoke the model and return the output message as a string in an output field."""
    response = model.invoke(state['messages'])

    return { "messages_output_field": str(response) }


# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig, output=OutputSchema)

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
