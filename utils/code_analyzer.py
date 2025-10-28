import os
from typing import Any, Dict, List, Optional, Tuple
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_go as tsgo
import tree_sitter_cpp as tscpp
import tree_sitter_java as tsjava
import tree_sitter_rust as tsrust

# Dictionary to map file extensions to tree-sitter languages
LANGUAGES = {
    ".py": Language(tspython.language()),
    ".js": Language(tsjavascript.language()),
    ".go": Language(tsgo.language()),
    ".cpp": Language(tscpp.language()),
    ".c": Language(tscpp.language()),
    ".h": Language(tscpp.language()),
    ".hpp": Language(tscpp.language()),
    ".java": Language(tsjava.language()),
    ".rs": Language(tsrust.language()),
}


def get_language(file_path: str) -> Optional[Language]:
    """Get the tree-sitter Language for a given file path based on its extension."""
    _, ext = os.path.splitext(file_path)
    return LANGUAGES.get(ext)


class LanguageAnalyzer:
    """
    Base class for language-specific code analysis.
    """

    def __init__(self, code: str, language: Language):
        self.code = code
        self.language = language
        self.parser = Parser(language)
        self.tree = self.parser.parse(bytes(self.code, "utf8"))
        self.stats: Dict[str, Any] = {"classes": [], "functions": []}

    def analyze(self):
        """Starts the analysis of the code."""
        self._traverse(self.tree.root_node)
        return self.stats

    def _traverse(self, node):
        """Traverses the AST and extracts information. To be implemented by subclasses."""
        raise NotImplementedError

    def get_docstring(self, node):
        """Extracts the docstring from a given node, if available. To be implemented by subclasses."""
        raise NotImplementedError


class PythonAnalyzer(LanguageAnalyzer):
    """
    Analyzer for Python code.
    """

    def _traverse(self, node):
        if node.type == "class_definition":
            self._extract_class_info(node)
        elif node.type == "function_definition":
            # Avoid capturing methods inside class traversal
            if not self._is_in_class(node):
                self._extract_function_info(node)

        for child in node.children:
            self._traverse(child)

    def _is_in_class(self, node):
        parent = node.parent
        while parent:
            if parent.type == "class_definition":
                return True
            parent = parent.parent
        return False

    def get_docstring(self, node):
        """Extracts the docstring from a given node, if available."""
        try:
            body_node = node.child_by_field_name("body")
            if body_node and body_node.children:
                first_statement = body_node.children[0]
                if (
                    first_statement.type == "expression_statement"
                    and first_statement.children[0].type == "string"
                ):
                    string_node = first_statement.children[0]
                    docstring_raw = string_node.text.decode("utf-8")
                    # Remove quotes, handling different quote types and prefixes (r, u, f)
                    if docstring_raw.startswith(('"""', "'''")):
                        return docstring_raw[3:-3]
                    elif docstring_raw.startswith(('r"', "r'", 'u"', "u'", 'f"', "f'")):
                        return docstring_raw[2:-1]
                    else:
                        return docstring_raw[1:-1]
        except Exception:
            return None
        return None

    def _extract_class_info(self, node):
        class_name_node = node.child_by_field_name("name")
        class_name = (
            class_name_node.text.decode("utf-8")
            if class_name_node
            else "AnonymousClass"
        )

        docstring = self.get_docstring(node)

        methods = []
        body_node = node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                if child.type == "function_definition":
                    method_info = self._extract_function_info(child, is_method=True)
                    if method_info:
                        methods.append(method_info)

        self.stats["classes"].append(
            {
                "name": class_name,
                "methods": methods,
                "docstring": docstring,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
            }
        )

    def _extract_function_info(self, node, is_method=False):
        name_node = node.child_by_field_name("name")
        name = name_node.text.decode("utf-8") if name_node else "AnonymousFunction"

        params_node = node.child_by_field_name("parameters")
        params = params_node.text.decode("utf-8") if params_node else ""

        docstring = self.get_docstring(node)

        function_info = {
            "name": name,
            "parameters": params,
            "docstring": docstring,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
        }

        if not is_method:
            self.stats["functions"].append(function_info)

        return function_info


class JavaScriptAnalyzer(LanguageAnalyzer):
    """
    Analyzer for JavaScript code.
    """

    def _traverse(self, node):
        if node.type == "class_declaration":
            self._extract_class_info(node)
        elif node.type == "function_declaration":
            # Avoid capturing methods inside class traversal
            if not self._is_in_class(node):
                self._extract_function_info(node)

        for child in node.children:
            self._traverse(child)

    def _is_in_class(self, node):
        parent = node.parent
        while parent:
            if parent.type == "class_declaration":
                return True
            parent = parent.parent
        return False

    def get_docstring(self, node):
        """Extracts the docstring from a given node, if available."""
        lines = self.code.split("\n")
        start_line = node.start_point[0]
        for i in range(start_line - 1, -1, -1):
            line = lines[i].strip()
            if line.endswith("*/"):
                # Find the start /**
                for k in range(i, -1, -1):
                    if lines[k].strip().startswith("/**"):
                        doc_lines = lines[k : i + 1]
                        docstring = "\n".join(doc_lines)
                        # Remove /** and */
                        docstring = (
                            docstring.replace("/**", "", 1).replace("*/", "", 1).strip()
                        )
                        return docstring
                break
            elif line and not line.startswith("//"):
                break  # Stop if non-comment line
        return None

    def _extract_class_info(self, node):
        class_name_node = node.child_by_field_name("name")
        class_name = (
            class_name_node.text.decode("utf-8")
            if class_name_node
            else "AnonymousClass"
        )

        methods = []
        body_node = node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                if child.type == "method_definition":
                    method_info = self._extract_function_info(child, is_method=True)
                    if method_info:
                        methods.append(method_info)

        self.stats["classes"].append(
            {
                "name": class_name,
                "methods": methods,
                "docstring": self.get_docstring(node),
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
            }
        )

    def _extract_function_info(self, node, is_method=False):
        name_node = node.child_by_field_name("name")
        name = name_node.text.decode("utf-8") if name_node else "AnonymousFunction"

        params_node = node.child_by_field_name("parameters")
        params = params_node.text.decode("utf-8") if params_node else ""

        function_info = {
            "name": name,
            "parameters": params,
            "docstring": self.get_docstring(node),
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
        }

        if not is_method:
            self.stats["functions"].append(function_info)

        return function_info


def get_analyzer(language: Language) -> type[LanguageAnalyzer]:
    """Factory function to get the appropriate analyzer class.

    Args:
        language (Language): The programming language to analyze.

    Returns:
        type[LanguageAnalyzer]: The analyzer class for the specified language.
    """
    if language == Language(tspython.language()):
        return PythonAnalyzer
    if language == Language(tsjavascript.language()):
        return JavaScriptAnalyzer
    # Return a default analyzer that does nothing if language is not supported
    return LanguageAnalyzer


def analyze_code_from_string(
    code_string: str, language: Language
) -> Optional[Dict[str, Any]]:
    """Analyzes a string of code and returns structural information.

    Args:
        code_string (str): The code to analyze.
        language (Language): The programming language of the code.

    Returns:
        Optional[Dict[str, Any]]: The analysis results or None if analysis fails.
    """
    try:
        AnalyzerClass = get_analyzer(language)
        analyzer = AnalyzerClass(code_string, language)
        return analyzer.analyze()
    except Exception:
        return None


def analyze_file_with_tree_sitter(file_path: str) -> Optional[Dict[str, Any]]:
    """Analyzes a single file using tree-sitter.

    Args:
        file_path (str): The path to the file to analyze.

    Returns:
        Optional[Dict[str, Any]]: The analysis results or None if analysis fails.
    """
    language = get_language(file_path)
    if not language:
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code_to_analyze = f.read()
        return analyze_code_from_string(code_to_analyze, language)
    except Exception:
        return None


def format_tree_sitter_analysis_results(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Formats the analysis results for a single file into a dictionary.

    Args:
        stats (Dict[str, Any]): The analysis statistics.

    Returns:
        Dict[str, Any]: The formatted analysis results.
    """
    if not stats:
        return {}
    
    summary = {
        "classes": [{"name": cls["name"], "description": cls.get("docstring", "")} for cls in stats.get("classes", [])],
        "functions": [{"name": func["name"], "description": func.get("docstring", "")} for func in stats.get("functions", [])]
    }
    
    stats["summary"] = summary
    return stats


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: uv run utils/code_analyzer.py <file_path>")
        sys.exit(1)

    path = sys.argv[1]
    if os.path.isfile(path):
        file_stats = analyze_file_with_tree_sitter(path)
        if file_stats:
            print(json.dumps(format_tree_sitter_analysis_results(file_stats), indent=2))
    else:
        print("Please provide a valid file path.")
