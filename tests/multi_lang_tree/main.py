import os
import ast
import json
import re
from multi_lang_tree.simplify_tree import main

# === 查找主函数文件 ===
def find_main_files(root="."):
    """自动搜索 main.py / main.c / main.cpp 文件"""
    candidates = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower() in ("main.py", "main.c", "main.cpp"):
                candidates.append(os.path.join(dirpath, f))
    return candidates

# === Python 调用图解析 ===
class PyCallGraphBuilder(ast.NodeVisitor):
    def __init__(self):
        self.functions = {}  # {func_name: {"calls": []}}
        self.current_func = None

    def visit_FunctionDef(self, node):
        func_name = node.name
        self.functions.setdefault(func_name, {"calls": []})
        prev_func = self.current_func
        self.current_func = func_name
        self.generic_visit(node)
        self.current_func = prev_func

    def visit_Call(self, node):
        if self.current_func is None:
            return
        func_id = None
        if isinstance(node.func, ast.Name):
            func_id = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_id = node.func.attr  # obj.method()
        if func_id:
            self.functions[self.current_func]["calls"].append(func_id)
        self.generic_visit(node)

def build_call_tree(entry_func, call_dict, visited=None):
    if visited is None:
        visited = set()
    if entry_func in visited:
        return {"name": entry_func, "calls": []}  # 遇到循环就停止
    visited.add(entry_func)
    node_info = call_dict.get(entry_func, {"calls": []})
    return {
        "name": entry_func,
        "calls": [build_call_tree(c, call_dict, visited.copy()) for c in node_info.get("calls", [])]
    }

def parse_python(file_path):
    """解析 Python 文件并生成调用树"""
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=file_path)

    builder = PyCallGraphBuilder()
    builder.visit(tree)

    # 显式 main
    if "main" in builder.functions:
        entry_func = "main"
    else:
        # 隐式 __main__ 块
        main_body_nodes = []
        for node in tree.body:
            if isinstance(node, ast.If):
                test = node.test
                if (isinstance(test, ast.Compare) and
                    isinstance(test.left, ast.Name) and
                    test.left.id == "__name__"):
                    main_body_nodes = node.body
                    break

        if not main_body_nodes:
            print(f"⚠️ No 'main' function or __main__ block found in {file_path}")
            return {}

        builder.functions["__main__"] = {"calls": []}
        builder.current_func = "__main__"
        for stmt in main_body_nodes:
            builder.visit(stmt)
        entry_func = "__main__"

    return build_call_tree(entry_func, builder.functions)

# === C / C++ 简单调用图（正则提取）===
def parse_c_cpp(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    func_pattern = re.compile(r"\b([a-zA-Z_]\w*)\s*\([^)]*\)\s*\{")
    call_pattern = re.compile(r"\b([a-zA-Z_]\w*)\s*\(")

    functions = {}
    for match in func_pattern.finditer(code):
        func_name = match.group(1)
        body_start = match.end()
        # 简单找到对应的右括号
        brace_count = 1
        i = body_start
        while i < len(code) and brace_count > 0:
            if code[i] == "{":
                brace_count += 1
            elif code[i] == "}":
                brace_count -= 1
            i += 1
        body = code[body_start:i]
        called_funcs = [m.group(1) for m in call_pattern.finditer(body) if m.group(1) != func_name]
        functions[func_name] = {"calls": called_funcs}

    if "main" not in functions:
        print(f"⚠️ No 'main' function found in {file_path}")
        return {}
    return build_call_tree("main", functions)

# === 总入口 ===
def analyze_project(root="."):
    results = []
    
    # 核心修改：将传入的 root 路径作为唯一的待分析文件
    if not os.path.isfile(root):
        print(f"❌ Error: The provided path '{root}' is not a valid file.")
        return None
        
    main_files = [root] # 将单个文件路径放入列表中供循环使用
    
    # 原代码中的查找逻辑已被移除或注释
    # main_files = find_main_files(root)
    # if not main_files:
    #     print("❌ No main.py / main.c / main.cpp found.")
    #     return None

    for file_path in main_files:
        # 确保文件存在且是文件
        if not os.path.isfile(file_path):
            print(f"⚠️ Skipping '{file_path}' as it is not a valid file.")
            continue
            
        ext = os.path.splitext(file_path)[1].lower()
        print(f"🔍 Analyzing {file_path} ...")
        
        if ext == ".py":
            tree = parse_python(file_path)
        elif ext in (".c", ".cpp"):
            tree = parse_c_cpp(file_path)
        else:
            print(f"⚠️ Skipping file with unsupported extension: {ext}")
            continue
            
        results.append({"file": file_path, "tree": tree})

    return results

def main_process(inputpath):
    result = analyze_project(inputpath)
    
    with open("call_tree.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("✅ 已保存到 call_tree.json")

    result=main("call_tree.json")
    print("✅ 已化简")
    with open("call_tree_simplified.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result

# === 执行分析并保存 ===
# if __name__ == "__main__":
    # result = analyze_project("repo-agent")
    
    # with open("call_tree.json", "w", encoding="utf-8") as f:
    #     json.dump(result, f, indent=2, ensure_ascii=False)

    # print("✅ 已保存到 call_tree.json")