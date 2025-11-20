from pyvis.network import Network
import json

def visualize_call_tree(json_data, output_html="call_graph.html"):
    net = Network(height="750px", width="100%", directed=True)
    net.force_atlas_2based()

    def add_nodes_recursively(node, parent=None):
        if isinstance(node, str):
            net.add_node(node, label=node, title=node)
            if parent:
                net.add_edge(parent, node)
            return

        if not isinstance(node, dict) or "name" not in node:
            return

        name = node["name"]
        net.add_node(name, label=name, title=name)
        if parent:
            net.add_edge(parent, name)

        for child in node.get("calls", []):
            add_nodes_recursively(child, name)

    if isinstance(json_data, dict):
        json_data = [json_data]  # 统一为 list 处理

    for file_entry in json_data:
        tree = file_entry.get("tree", file_entry)
        add_nodes_recursively(tree)

    net.show(output_html, notebook=False)
    print(f"✅ 函数调用图已生成：{output_html}")

json_path="multi_lang_tree/call_tree.json"

with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        visualize_call_tree(data)