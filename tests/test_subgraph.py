from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.graph import MermaidDrawMethod
from datetime import datetime
import os


# 1. 定义状态
class SubgraphState(TypedDict):
    # 子图状态
    value: int


def sub_node_1(state: SubgraphState):
    print("---sub_node_1---")
    state["value"] += 1
    return state


def sub_node_2(state: SubgraphState):
    print("---sub_node_2---")
    state["value"] += 1
    return state


# 2. 定义子图
subgraph = StateGraph(SubgraphState)
subgraph.add_node("sub_node_1", sub_node_1)
subgraph.add_node("sub_node_2", sub_node_2)
subgraph.add_edge(START, "sub_node_1")
subgraph.add_edge("sub_node_1", "sub_node_2")
subgraph.add_edge("sub_node_2", END)
# 编译子图
subgraph_runnable = subgraph.compile()


# 3. 定义主图状态
class MainGraphState(TypedDict):
    # 主图状态
    value: int


def main_node_1(state: MainGraphState):
    print("---main_node_1---")
    state["value"] += 1
    return state


def main_node_2(state: MainGraphState):
    print("---main_node_2---")
    state["value"] += 1
    return state


# 4. 定义主图
workflow = StateGraph(MainGraphState)
workflow.add_node("main_node_1", main_node_1)
# 将子图添加为节点
workflow.add_node("subgraph", subgraph_runnable)
workflow.add_node("main_node_2", main_node_2)

workflow.add_edge(START, "main_node_1")
workflow.add_edge("main_node_1", "subgraph")
workflow.add_edge("subgraph", "main_node_2")
workflow.add_edge("main_node_2", END)

# 5. 编译并运行
app = workflow.compile()

img = app.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.API,
)
dir_name = "./.graphs"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
graph_file_name = os.path.join(
    dir_name,
    f"wiki_agent_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png",
)
with open(graph_file_name, "wb") as f:
    f.write(img)

# 运行并打印结果
initial_state = {"value": 0}
final_state = app.invoke(initial_state)
print(final_state)
