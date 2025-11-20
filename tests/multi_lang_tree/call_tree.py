function_index = {}  # 全局统一索引

def build_call_tree(func_name, visited=None):
    if visited is None:
        visited = set()
    if func_name in visited:
        return {"name": func_name, "calls": []}  # 避免无限递归
    visited.add(func_name)
    node = {"name": func_name, "calls": []}
    for call in function_index.get(func_name, {}).get("calls", []):
        node["calls"].append(build_call_tree(call, visited))
    return node