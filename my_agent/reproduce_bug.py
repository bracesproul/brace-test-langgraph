import operator
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from typing_extensions import Annotated


# States
class InputState(BaseModel):
    """Input state for the graph"""
    items: Annotated[List[str], operator.add] = Field(default_factory=list, description="List of items to process")

class OutputState(BaseModel):
    """Output state for the graph"""
    processed_items: Annotated[List[str], operator.add] = Field(default_factory=list, description="List of processed items")

class WorkingState(InputState, OutputState):
    """Working state that combines input and output"""
    current_item: str = Field(default="", description="Currently processing item")

class ConfigSchema(BaseModel):
    """Configuration schema"""
    streaming: bool = Field(default=True, description="Whether to stream results")

# Node functions
def find_items(state: WorkingState) -> WorkingState:
    """Find items to process"""
    return WorkingState(
        items=["item1", "item2", "item3"],
        processed_items=state.processed_items
    )

def map_items_to_process(state: WorkingState) -> List[Send]:
    """Map each item to an individual process operation"""
    return [
        Send("Process Item", WorkingState(
            items=[item],
            processed_items=[]
        ))
        for item in state.items
    ]

def prepare_item(state: WorkingState) -> Dict[str, Any]:
    """Prepare a single item for processing"""
    if not state.items:
        print("No items to process")
        return {"current_item": ""}

    item = state.items[0]
    print(f"Preparing item: {item}")
    return {"current_item": item}

def process_single_item(state: WorkingState) -> Dict[str, Any]:
    """Process a single item in the subgraph"""
    if not state.current_item:
        return {"processed_items": []}

    processed = f"Processed {state.current_item}"
    return {"processed_items": [processed]}

def create_process_item_subgraph() -> CompiledStateGraph:
    """Creates a subgraph for processing a single item"""
    workflow = StateGraph(
        WorkingState,
        input=WorkingState,
        # Using OutputState here triggers the null bug
        output=OutputState,
        # output=WorkingState,
        config_schema=ConfigSchema
    )

    # Add nodes
    workflow.add_node("Prepare Item", prepare_item)
    workflow.add_node("Process Single Item", process_single_item)

    # Connect nodes
    workflow.add_edge(START, "Prepare Item")
    workflow.add_edge("Prepare Item", "Process Single Item")
    workflow.add_edge("Process Single Item", END)

    return workflow.compile()

def create_main_graph() -> StateGraph:
    """Creates the main graph"""
    workflow = StateGraph(
        WorkingState,
        input=InputState,
        output=OutputState,
        config_schema=ConfigSchema
    )

    # Add nodes
    workflow.add_node("Find Items", find_items)
    workflow.add_node("Process Item", create_process_item_subgraph())

    # Connect nodes with fan-out pattern
    workflow.add_edge(START, "Find Items")
    workflow.add_conditional_edges(
        "Find Items",
        map_items_to_process,
        ["Process Item"]
    )
    workflow.add_edge("Process Item", END)

    return workflow

# Create the compiled graph instance
graph = create_main_graph()