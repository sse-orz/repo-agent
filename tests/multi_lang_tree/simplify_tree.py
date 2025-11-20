import json
import sys
import os
from collections import defaultdict

def load_json(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def merge_sibling_calls(calls_list):
    """
    核心化简逻辑：
    1. 将同级的函数调用按“函数名”分组。
    2. 统计调用次数。
    3. 将所有同名函数的子调用列表(calls)合并，以便递归展示所有可能的子路径。
    """
    if not calls_list:
        return []

    # 使用字典聚合： key=函数名, value={ 'count': 0, 'all_sub_calls': [] }
    grouped = defaultdict(lambda: {'count': 0, 'all_sub_calls': []})
    
    # 保持顺序（首次出现的顺序）
    order = [] 

    for call in calls_list:
        name = call.get('name', 'unknown')
        if name not in grouped:
            order.append(name)
        
        grouped[name]['count'] += 1
        # 收集该次调用的所有子调用
        if 'calls' in call and call['calls']:
            grouped[name]['all_sub_calls'].extend(call['calls'])

    # 构建合并后的结果列表
    merged_results = []
    for name in order:
        info = grouped[name]
        merged_results.append({
            'name': name,
            'count': info['count'],
            'calls': info['all_sub_calls'] # 这里包含了所有该同名函数下的子调用
        })
    
    return merged_results

def build_tree_text(node, depth=0, is_root=False):
    """递归构建文本树"""
    lines = []
    
    # 1. 处理当前节点显示的名称
    name = node.get('name', 'unknown')
    
    # 如果是根节点，直接显示文件名或Main
    if is_root:
        display_str = f"root: {name}"
    else:
        # 处理计数显示
        count = node.get('count', 1)
        count_str = f" [x{count}]" if count > 1 else ""
        
        # 缩进 (使用2个空格节省Token)
        indent = "  " * depth
        display_str = f"{indent}- {name}{count_str}"

    lines.append(display_str)

    # 2. 处理子节点
    # 注意：这里的 'calls' 已经是通过 merge_sibling_calls 合并过的列表（如果是递归调用）
    # 但我们在递归前需要先对当前层级的 children 进行合并
    raw_children = node.get('calls', [])
    if raw_children:
        # 对子节点进行“同名聚合”
        merged_children = merge_sibling_calls(raw_children)
        
        for child in merged_children:
            # 递归生成子树
            lines.extend(build_tree_text(child, depth + 1, is_root=False))
            
    return lines

def generate_header(file_path, tree_index=None):
    """生成给Agent看的说明头，现在可以标记是第几个树"""
    index_str = f" [Tree {tree_index + 1}]" if tree_index is not None else ""
    return [
        "=== COMPRESSED FUNCTION CALL TREE ===",
        f"Source: {file_path}{index_str}",
        "Format: Indentation represents call depth.",
        "Notation: 'Function [xN]' means the function is called N times at this level.",
        "Logic: Children nodes are a UNION of all calls from those N instances (showing all possible paths).",
        "-------------------------------------"
    ]

def main(input_path):
    # 读取数据
    data = load_json(input_path)
    
    # --- 【关键修改：处理列表和字典】 ---
    
    trees_to_process = []
    
    if isinstance(data, list):
        # 场景 1: JSON 根是列表（包含多个树对象）
        trees_to_process = data
    elif isinstance(data, dict):
        # 场景 2: JSON 根是字典
        if 'tree' in data or 'name' in data:
             # 如果包含 'tree' key 或者它本身就是根节点，将其封装在列表中
            trees_to_process = [data]
        else:
            print("Error: Dictionary root structure is missing expected keys ('tree' or 'name').")
            sys.exit(1)
    else:
        print(f"Error: JSON root is neither a dictionary nor a list. Type found: {type(data)}")
        sys.exit(1)

    # 循环处理所有检测到的树结构
    all_output_lines = []
    
    for index, tree_data in enumerate(trees_to_process):
        
        # 从当前字典中安全地提取路径和根节点
        file_path = tree_data.get('file', f'Unknown File (Tree {index + 1})')
        
        # 兼容两种格式：直接是 tree 对象，或者包含 "tree" key
        # 使用 .get() 方法，现在 tree_data 保证是 dict，不会报错
        root_node = tree_data.get('tree', tree_data) 
        
        # 生成头部和树状文本
        header_lines = generate_header(file_path, index if len(trees_to_process) > 1 else None)
        tree_lines = build_tree_text(root_node, depth=0, is_root=True)
        
        # 分隔多个树的输出
        if index > 0:
            all_output_lines.append("\n\n#####################################\n\n")

        all_output_lines.extend(header_lines)
        all_output_lines.extend(tree_lines)


    # 输出文件
    output_path = input_path + ".txt"
    content = "\n".join(all_output_lines)
    
    # ... (文件写入逻辑保持不变)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Success! Compressed tree saved to:\n{output_path}")
        print(f"Original size: {os.path.getsize(input_path)} bytes")
        print(f"Compressed size: {len(content.encode('utf-8'))} bytes")
    except Exception as e:
        print(f"Error writing file: {e}")