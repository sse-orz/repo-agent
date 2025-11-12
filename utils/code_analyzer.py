import os
import re
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
    ".py": tspython.language(),
    ".js": tsjavascript.language(),
    ".go": tsgo.language(),
    ".cpp": tscpp.language(),
    ".c": tscpp.language(),
    ".h": tscpp.language(),
    ".hpp": tscpp.language(),
    ".java": tsjava.language(),
    ".rs": tsrust.language(),
}


def get_language(file_path: str) -> Optional[Language]:
    """Gets the configured tree-sitter language for the file extension.

    Args:
        file_path (str): Absolute or relative path of the file being inspected.

    Returns:
        Optional[Language]: Matched tree-sitter language or None if unsupported.
    """
    _, ext = os.path.splitext(file_path)
    lang = LANGUAGES.get(ext)
    return Language(lang) if lang else None


class LanguageAnalyzer:
    """
    Base class for language-specific code analysis.
    """

    def __init__(self, code: str, language: Language):
        """Initializes shared analyzer state for a language.

        Args:
            code (str): Source code snippet that should be parsed.
            language (Language): Tree-sitter language used for parsing.

        Returns:
            None: The constructor configures instance attributes in place.
        """
        self.code = code
        self.language = language
        # Correct parser initialization: create parser then set language
        self.parser = Parser()
        self.parser.language = language
        self.tree = self.parser.parse(bytes(self.code, "utf8"))
        self.stats: Dict[str, Any] = {"classes": [], "functions": []}

    def analyze(self):
        """Runs the AST traversal to collect structural statistics.

        Args:
            None

        Returns:
            Dict[str, Any]: Aggregated class and function metadata for the file.
        """
        self._traverse(self.tree.root_node)
        return self.stats

    def _traverse(self, node):
        """Traverses the AST and extracts domain-specific information.

        Args:
            node (Node): Root node that should be walked recursively.

        Returns:
            None: Subclasses implement traversal and populate internal stats.
        """
        raise NotImplementedError

    def get_docstring(self, node):
        """Extracts the documentation string from the supplied syntax node.

        Args:
            node (Node): AST node whose leading documentation should be parsed.

        Returns:
            Optional[str]: Extracted docstring or None when not present.
        """
        raise NotImplementedError


class PythonAnalyzer(LanguageAnalyzer):
    """
    Analyzer for Python code.
    """

    def __init__(self, code: str, language: Language):
        """Initializes Python analyzer state and dependency tracking.

        Args:
            code (str): Python source code that should be inspected.
            language (Language): Tree-sitter language instance for Python.

        Returns:
            None: Populates the stats dictionary with Python-specific keys.
        """
        # reuse base initializer then add python-specific state
        super().__init__(code, language)
        # collect external functions imported from other modules/files
        # each item will be a dict: {"module": "<module>", "name": "<func_name>"}
        self.stats["outer_dependencies"] = []

    def _traverse(self, node):
        """Walks the Python AST to collect classes, functions, and imports.

        Args:
            node (Node): Current AST node being traversed recursively.

        Returns:
            None: Updates the stats structure in place during traversal.
        """
        # Collect from-imported functions
        if node.type in ("import_from_statement", "import_from", "import_statement"):
            self._extract_import_info(node)
            # continue traversal to capture nested constructs as well

        if node.type == "class_definition":
            self._extract_class_info(node)
        elif node.type == "function_definition":
            # Avoid capturing methods inside class traversal
            if not self._is_in_class(node):
                self._extract_function_info(node)

        for child in node.children:
            self._traverse(child)

    def _is_in_class(self, node):
        """Determines whether the supplied node resides inside a class.

        Args:
            node (Node): Candidate AST node whose ancestry is inspected.

        Returns:
            bool: True when an ancestor is a class_definition node, else False.
        """
        parent = node.parent
        while parent:
            if parent.type == "class_definition":
                return True
            parent = parent.parent
        return False

    def get_docstring(self, node):
        """Extracts the Python docstring for the provided declaration node.

        Args:
            node (Node): Class or function node whose body is inspected.

        Returns:
            Optional[str]: Unwrapped docstring value or None when absent.
        """
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
            return ""
        return ""

    def _extract_class_info(self, node):
        """Captures metadata for a Python class declaration.

        Args:
            node (Node): Class_definition node produced by tree-sitter.

        Returns:
            None: Appends class details to the stats dictionary.
        """
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
        """Captures metadata for a Python function or method declaration.

        Args:
            node (Node): Function_definition node to parse.
            is_method (bool): Indicates whether the node belongs to a class body.

        Returns:
            Dict[str, Any]: Parsed function description including signature data.
        """
        name_node = node.child_by_field_name("name")
        name = name_node.text.decode("utf-8") if name_node else "AnonymousFunction"

        params_node = node.child_by_field_name("parameters")
        parameters = self._parse_parameters(params_node)

        return_type_node = node.child_by_field_name("return_type")
        return_type = (
            return_type_node.text.decode("utf-8") if return_type_node else None
        )

        docstring = self.get_docstring(node)

        function_info = {
            "name": name,
            "parameters": parameters,
            "return_type": return_type,
            "docstring": docstring,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
        }

        if not is_method:
            self.stats["functions"].append(function_info)

        return function_info

    def _parse_parameters(self, params_node):
        """Parses a parameters node to produce name and type metadata.

        Args:
            params_node (Node|None): Parameters child extracted from the AST.

        Returns:
            List[Dict[str, Optional[str]]]: Normalized parameter descriptors.
        """
        parameters = []
        if not params_node:
            return parameters
        # Traverse children to find parameter nodes, skipping '(', ',', ')'
        for child in params_node.children:
            if child.type in (
                "identifier",
                "typed_parameter",
                "default_parameter",
                "typed_default_parameter",
            ):
                param_info = self._extract_parameter_info(child)
                if param_info:
                    parameters.append(param_info)
        return parameters

    def _extract_parameter_info(self, param_node):
        """Extracts name and optional type annotation from a parameter node.

        Args:
            param_node (Node): Parameter-related node provided by tree-sitter.

        Returns:
            Optional[Dict[str, Optional[str]]]: Parameter metadata or None.
        """
        name = None
        type_annotation = ""
        if param_node.type == "identifier":
            name = param_node.text.decode("utf-8")
        elif param_node.type == "typed_parameter":
            # children: identifier, :, type
            if len(param_node.children) >= 3:
                name = param_node.children[0].text.decode("utf-8")
                type_annotation = param_node.children[2].text.decode("utf-8")
        elif param_node.type == "default_parameter":
            # children: identifier, =, expression
            if len(param_node.children) >= 1:
                name = param_node.children[0].text.decode("utf-8")
        elif param_node.type == "typed_default_parameter":
            # children: identifier, :, type, =, expression
            if len(param_node.children) >= 3:
                name = param_node.children[0].text.decode("utf-8")
                type_annotation = param_node.children[2].text.decode("utf-8")
        return {"name": name, "type": type_annotation or ""} if name else None

    def _extract_import_info(self, node):
        """Records modules and symbols imported by the current Python node.

        Args:
            node (Node): Import-related node sourced from the AST traversal.

        Returns:
            None: Populates the outer_dependencies list with discovered imports.
        """
        try:
            node_text = node.text.decode("utf-8")
        except Exception:
            # fallback to slicing the source by byte offsets
            try:
                node_text = self.code[node.start_byte : node.end_byte]
            except Exception:
                return

        if "from" in node_text:
            # handle from import: from module import a, b as c
            m = re.match(r"from\s+([\.\w]+)\s+import\s+(.+)", node_text)
            if not m:
                return

            module = m.group(1)
            imported_part = m.group(2).strip()
            # strip surrounding parentheses if present
            if imported_part.startswith("(") and imported_part.endswith(")"):
                imported_part = imported_part[1:-1]

            # split by comma and strip aliases
            parts = [p.strip() for p in imported_part.split(",") if p.strip()]
            for part in parts:
                # handle 'name as alias'
                name = part.split(" as ")[0].strip()
                # ignore wildcard imports
                if name == "*":
                    continue
                self.stats.setdefault("outer_dependencies", []).append(
                    {"module": module, "name": name}
                )
        else:
            # handle simple import: import os, sys as s
            imported_part = node_text.replace("import", "").strip()
            parts = [p.strip() for p in imported_part.split(",") if p.strip()]
            for part in parts:
                module = part.split(" as ")[0].strip()
                self.stats["outer_dependencies"].append({"module": module, "name": ""})


class JavaScriptAnalyzer(LanguageAnalyzer):
    """
    Analyzer for JavaScript code.
    """

    def __init__(self, code: str, language: Language):
        """Initializes JavaScript analyzer state and import tracking.

        Args:
            code (str): JavaScript or TypeScript-like source being parsed.
            language (Language): Tree-sitter language corresponding to the code.

        Returns:
            None: Populates analyzer statistics and primes import extraction.
        """
        super().__init__(code, language)
        # collect imported named functions from ES modules and require() destructuring
        self.stats.setdefault("outer_dependencies", [])
        self._collect_js_imports()

    def _collect_js_imports(self):
        """Scans source text for module imports to capture dependencies.

        Args:
            None

        Returns:
            None: Updates the outer_dependencies list with import metadata.
        """
        # find ES6 named imports: import { a, b as c } from 'mod'
        for m in re.finditer(
            r"import\s+\{([^}]+)\}\s+from\s+['\"]([^'\"]+)['\"]", self.code
        ):
            names = m.group(1)
            module = m.group(2)
            parts = [p.strip() for p in names.split(",") if p.strip()]
            for part in parts:
                name = part.split(" as ")[0].strip()
                if name == "*":
                    continue
                self.stats.setdefault("outer_dependencies", []).append(
                    {"module": module, "name": name}
                )

        # default/imported default: import defName from 'mod'
        for m in re.finditer(
            r"import\s+([A-Za-z_\$][A-Za-z0-9_\$]*)\s+from\s+['\"]([^'\"]+)['\"]",
            self.code,
        ):
            name = m.group(1)
            module = m.group(2)
            self.stats.setdefault("outer_dependencies", []).append(
                {"module": module, "name": name}
            )

        # require destructuring: const {a, b: c} = require('mod')
        for m in re.finditer(
            r"(?:const|let|var)\s+\{([^}]+)\}\s*=\s*require\(['\"]([^'\"]+)['\"]\)",
            self.code,
        ):
            names = m.group(1)
            module = m.group(2)
            parts = [p.strip() for p in names.split(",") if p.strip()]
            for part in parts:
                name = part.split(":")[0].strip()
                if name == "*":
                    continue
                self.stats.setdefault("outer_dependencies", []).append(
                    {"module": module, "name": name}
                )

        # side-effect import: import 'module';
        for m in re.finditer(r"import\s+['\"]([^'\"]+)['\"]\s*;", self.code):
            module = m.group(1)
            self.stats["outer_dependencies"].append({"module": module, "name": ""})

    def _traverse(self, node):
        """Walks the JavaScript AST to harvest classes and free functions.

        Args:
            node (Node): Current AST node encountered during traversal.

        Returns:
            None: Delegates to specialized extractors for supported constructs.
        """
        if node.type == "class_declaration":
            self._extract_class_info(node)
        elif node.type == "function_declaration":
            # Avoid capturing methods inside class traversal
            if not self._is_in_class(node):
                self._extract_function_info(node)

        for child in node.children:
            self._traverse(child)

    def _is_in_class(self, node):
        """Checks whether a node is nested within a class declaration.

        Args:
            node (Node): Node whose ancestry should be evaluated.

        Returns:
            bool: True if a class_declaration ancestor exists, otherwise False.
        """
        parent = node.parent
        while parent:
            if parent.type == "class_declaration":
                return True
            parent = parent.parent
        return False

    def get_docstring(self, node):
        """Extracts the leading block comment that documents a declaration.

        Args:
            node (Node): Declaration node positioned after a potential JSDoc block.

        Returns:
            Optional[str]: Normalized documentation text or None when missing.
        """
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
        return ""

    def _extract_class_info(self, node):
        """Collects metadata for a JavaScript class declaration.

        Args:
            node (Node): Class_declaration node captured during traversal.

        Returns:
            None: Appends class structure details into the stats dictionary.
        """
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
        """Collects metadata for a JavaScript function or method declaration.

        Args:
            node (Node): Function or method node identified by tree-sitter.
            is_method (bool): Flags method definitions housed within classes.

        Returns:
            Dict[str, Any]: Function signature and documentation details.
        """
        name_node = node.child_by_field_name("name")
        name = name_node.text.decode("utf-8") if name_node else "AnonymousFunction"

        params_node = node.child_by_field_name("parameters")
        parameters = self._parse_parameters(params_node)

        function_info = {
            "name": name,
            "parameters": parameters,
            "return_type": "",
            "docstring": self.get_docstring(node),
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
        }

        if not is_method:
            self.stats["functions"].append(function_info)

        return function_info

    def _parse_parameters(self, params_node):
        """Parses formal parameters into a normalized representation.

        Args:
            params_node (Node|None): formal_parameters node provided by tree-sitter.

        Returns:
            List[Dict[str, Optional[str]]]: Extracted parameter descriptors.
        """
        parameters = []
        if not params_node:
            return parameters

        for child in params_node.children:
            info = self._extract_parameter_info(child)
            if info:
                parameters.append(info)
        return parameters

    def _extract_parameter_info(self, node):
        """Extracts identifier information from diverse parameter patterns.

        Args:
            node (Node): Node representing a parameter or pattern binding.

        Returns:
            Optional[Dict[str, Optional[str]]]: Parameter metadata when resolved.
        """
        if node.type == "identifier":
            return {"name": node.text.decode("utf-8"), "type": ""}
        if node.type == "assignment_pattern":
            target = node.child_by_field_name("left")
            if not target and node.children:
                target = node.children[0]
            if target:
                return {"name": target.text.decode("utf-8"), "type": ""}
            text = node.text.decode("utf-8")
            name = text.split("=")[0].strip()
            return {"name": name, "type": ""} if name else None
        if node.type == "rest_pattern":
            identifier = node.child_by_field_name("identifier")
            if not identifier:
                identifier = next(
                    (c for c in node.children if c.type == "identifier"), None
                )
            if identifier:
                return {"name": identifier.text.decode("utf-8"), "type": ""}
            return {"name": node.text.decode("utf-8"), "type": ""}
        if node.type in {
            "array_pattern",
            "object_pattern",
            "pair",
            "object_assignment_pattern",
        }:
            # Return the raw text so callers know the destructuring signature
            return {"name": node.text.decode("utf-8"), "type": ""}
        return None


class GoAnalyzer(LanguageAnalyzer):
    """
    Analyzer for Go code.
    """

    def __init__(self, code: str, language: Language):
        """Initializes Go analyzer state and dependency collectors.

        Args:
            code (str): Go source code slated for analysis.
            language (Language): Tree-sitter Go language instance to use.

        Returns:
            None: Populates statistics structures and kicks off import parsing.
        """
        super().__init__(code, language)
        self.stats.setdefault("outer_dependencies", [])
        self._collect_go_imports_and_usages()

    def _collect_go_imports_and_usages(self):
        """Parses import blocks and records qualified function usages.

        Args:
            None

        Returns:
            None: Updates the outer_dependencies list with modules and symbols.
        """
        # collect imported packages and aliases
        imports = {}
        # single import: import "pkg/path"
        for m in re.finditer(r'import\s+"([^"]+)"', self.code):
            path = m.group(1)
            alias = path.split("/")[-1]
            imports[alias] = path
            self.stats["outer_dependencies"].append({"module": path, "name": ""})

        # grouped imports: import ( "a" b "c" ) or with aliases: import ( name "path" )
        # simpler grouped parsing
        grouped = re.findall(r"import\s*\((.*?)\)", self.code, re.S)
        for block in grouped:
            for line in block.splitlines():
                line = line.strip()
                if not line:
                    continue
                # possible forms: "path" or alias "path"
                m2 = re.match(r'(?:([A-Za-z_][A-Za-z0-9_]*)\s+)?"([^"]+)"', line)
                if m2:
                    alias = m2.group(1) or m2.group(2).split("/")[-1]
                    path = m2.group(2)
                    imports[alias] = path
                    self.stats["outer_dependencies"].append(
                        {"module": path, "name": ""}
                    )

        # now scan for usages like alias.FuncName and record them
        for alias, path in imports.items():
            pattern = re.compile(
                r"\b" + re.escape(alias) + r"\.([A-Za-z_][A-Za-z0-9_]*)"
            )
            for m in pattern.finditer(self.code):
                func = m.group(1)
                self.stats.setdefault("outer_dependencies", []).append(
                    {"module": path, "name": func}
                )

    def _traverse(self, node):
        """Walks the Go AST to gather types, functions, and methods.

        Args:
            node (Node): Current syntax node obtained during traversal.

        Returns:
            None: Delegates to extractor helpers to populate statistics.
        """
        if node.type == "type_declaration":
            self._extract_class_info(node)
        elif node.type == "function_declaration":
            # Avoid capturing methods inside type traversal
            if not self._is_in_type(node):
                self._extract_function_info(node)
        elif node.type == "method_declaration":
            # Methods are handled separately, but for now treat as functions
            self._extract_function_info(node, is_method=True)

        for child in node.children:
            self._traverse(child)

    def _is_in_type(self, node):
        """Checks whether a node is defined inside a type declaration.

        Args:
            node (Node): Node whose ancestors should be inspected.

        Returns:
            bool: True when a type_declaration ancestor exists, else False.
        """
        parent = node.parent
        while parent:
            if parent.type == "type_declaration":
                return True
            parent = parent.parent
        return False

    def get_docstring(self, node):
        """Extracts the Go doc comment immediately preceding a node.

        Args:
            node (Node): Declaration node potentially preceded by // comments.

        Returns:
            Optional[str]: Concatenated comment text or None when absent.
        """
        lines = self.code.split("\n")
        start_line = node.start_point[0]
        doc_lines = []
        for i in range(start_line - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith("//"):
                doc_lines.insert(0, line[2:].strip())
            elif line:
                break
        if doc_lines:
            return "\n".join(doc_lines)
        return ""

    def _extract_class_info(self, node):
        """Captures struct or interface metadata from a type declaration.

        Args:
            node (Node): Type_declaration node encountered in traversal.

        Returns:
            None: Appends structural information to the analyzer stats.
        """
        type_spec = node.child_by_field_name("type")
        if type_spec and type_spec.type in ("struct_type", "interface_type"):
            name_node = node.child_by_field_name("name")
            name = name_node.text.decode("utf-8") if name_node else "AnonymousType"

            # For Go, methods are not nested in the type declaration
            # So, methods list is empty for now
            methods = []

            self.stats["classes"].append(
                {
                    "name": name,
                    "methods": methods,
                    "docstring": self.get_docstring(node),
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                }
            )

    def _extract_function_info(self, node, is_method=False):
        """Captures data for Go functions and methods.

        Args:
            node (Node): Function or method declaration node.
            is_method (bool): True when the declaration originates from a method.

        Returns:
            Dict[str, Any]: Recorded function metadata including signature data.
        """
        name_node = node.child_by_field_name("name")
        name = name_node.text.decode("utf-8") if name_node else "AnonymousFunction"

        params_node = node.child_by_field_name("parameters")
        parameters = self._parse_parameters(params_node)

        result_node = node.child_by_field_name("result")
        return_type = result_node.text.decode("utf-8") if result_node else ""

        function_info = {
            "name": name,
            "parameters": parameters,
            "return_type": return_type,
            "docstring": self.get_docstring(node),
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
        }

        if not is_method:
            self.stats["functions"].append(function_info)

        return function_info

    def _parse_parameters(self, params_node):
        """Parses Go parameter lists into structured descriptors.

        Args:
            params_node (Node|None): Parameters node extracted from tree-sitter.

        Returns:
            List[Dict[str, Optional[str]]]: Flattened representation of parameters.
        """
        parameters = []
        if not params_node:
            return parameters

        for i in range(params_node.child_count):
            child = params_node.child(i)
            entries = self._extract_parameter_info(child)
            if entries:
                parameters.extend(entries)
        return parameters

    def _extract_parameter_info(self, node):
        """Extracts parameter names and types from Go declarations.

        Args:
            node (Node): Parameter declaration node to normalize.

        Returns:
            List[Dict[str, Optional[str]]]: Zero or more parameter descriptors.
        """
        if node.type == "parameter_declaration":
            id_nodes = [
                node.child(i)
                for i in range(node.child_count)
                if node.child(i).type == "identifier"
            ]
            type_node = next(
                (
                    node.child(i)
                    for i in range(node.child_count)
                    if node.child(i).type not in {"identifier", ","}
                ),
                None,
            )
            type_text = type_node.text.decode("utf-8") if type_node else ""
            return [
                {"name": ident.text.decode("utf-8"), "type": type_text or ""}
                for ident in id_nodes
            ]

        if node.type == "variadic_parameter_declaration":
            name_node = next(
                (
                    node.child(i)
                    for i in range(node.child_count)
                    if node.child(i).type == "identifier"
                ),
                None,
            )
            type_node = next(
                (
                    node.child(i)
                    for i in range(node.child_count)
                    if node.child(i).type not in {"identifier", "..."}
                ),
                None,
            )
            if not name_node:
                return []
            type_text = type_node.text.decode("utf-8") if type_node else ""
            type_text = f"...{type_text}".strip()
            return [
                {
                    "name": name_node.text.decode("utf-8"),
                    "type": type_text or "...",
                }
            ]

        return []


class CppAnalyzer(LanguageAnalyzer):
    """
    Analyzer for C++ code.
    """

    def __init__(self, code: str, language: Language):
        """Initializes C++ analyzer state and header tracking.

        Args:
            code (str): C or C++ source code to analyze.
            language (Language): Tree-sitter C++ language instance.

        Returns:
            None: Stores references and immediately indexes include directives.
        """
        super().__init__(code, language)
        self.stats.setdefault("outer_dependencies", [])
        self._collect_cpp_includes_and_usages()

    def _collect_cpp_includes_and_usages(self):
        """Scans the source for includes and qualified symbol usages.

        Args:
            None

        Returns:
            None: Augments outer_dependencies with header and symbol references.
        """
        # collect included headers
        headers = []
        for m in re.finditer(r'#include\s+["<]([^">]+)[">]', self.code):
            headers.append(m.group(1))

        # add includes as dependencies
        for h in headers:
            self.stats["outer_dependencies"].append({"module": h, "name": ""})

        # find qualified calls like ns::func or Class::method
        for m in re.finditer(
            r"([A-Za-z_][A-Za-z0-9_:]*)::([A-Za-z_][A-Za-z0-9_]*)", self.code
        ):
            module = m.group(1)
            name = m.group(2)
            # try to map module to a header if possible (best-effort)
            mapped = ""
            # if module matches a header basename, use that
            for h in headers:
                if (
                    h.endswith(module + ".h")
                    or h.endswith(module + ".hpp")
                    or module.endswith(h.split("/")[-1].split(".")[0])
                ):
                    mapped = h
                    break
            self.stats.setdefault("outer_dependencies", []).append(
                {"module": mapped or module, "name": name}
            )

    def _find_function_declarator(self, node):
        """Recursively locates the nested function_declarator node.

        Args:
            node (Node|None): Declarator subtree yielded by tree-sitter.

        Returns:
            Optional[Node]: Matching function_declarator node when present.
        """
        if node and node.type == "function_declarator":
            return node
        if node:
            for child in node.children:
                result = self._find_function_declarator(child)
                if result:
                    return result
        return None

    def _extract_return_type(self, node):
        """Derives the return type text for a function definition.

        Args:
            node (Node): function_definition node from the C++ grammar.

        Returns:
            Optional[str]: Raw return type string or None if undetectable.
        """
        declarator = node.child_by_field_name("declarator")
        if not declarator:
            return ""

        start = node.start_point
        decl_start = declarator.start_point

        lines = self.code.split("\n")
        return_type_parts = []

        if start[0] == decl_start[0]:
            # Same line
            line = lines[start[0]]
            return_type = line[start[1] : decl_start[1]].strip()
        else:
            # Multi-line
            for i in range(start[0], decl_start[0]):
                if i == start[0]:
                    return_type_parts.append(lines[i][start[1] :])
                else:
                    return_type_parts.append(lines[i])
            return_type_parts.append(lines[decl_start[0]][: decl_start[1]])
            return_type = "\n".join(return_type_parts).strip()

        return return_type if return_type else ""

    def _traverse(self, node):
        """Walks the C++ AST to record classes and free functions.

        Args:
            node (Node): Node under consideration during traversal.

        Returns:
            None: Dispatches to helper extractors for supported constructs.
        """
        if node.type == "class_specifier":
            self._extract_class_info(node)
        elif node.type == "function_definition":
            # Avoid capturing methods inside class traversal
            if not self._is_in_class(node):
                self._extract_function_info(node)

        for child in node.children:
            self._traverse(child)

    def _is_in_class(self, node):
        """Determines whether a node belongs to a class scope.

        Args:
            node (Node): Node whose ancestor chain is reviewed.

        Returns:
            bool: True when a class_specifier ancestor exists, else False.
        """
        parent = node.parent
        while parent:
            if parent.type == "class_specifier":
                return True
            parent = parent.parent
        return False

    def get_docstring(self, node):
        """Extracts a preceding block comment that documents the node.

        Args:
            node (Node): Declaration node potentially documented with Javadoc style.

        Returns:
            Optional[str]: Cleaned comment text or None when not found.
        """
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
        """Collects metadata about a C++ class specification.

        Args:
            node (Node): class_specifier node discovered during traversal.

        Returns:
            None: Appends class data to the aggregate statistics structure.
        """
        name_node = node.child_by_field_name("name")
        class_name = name_node.text.decode("utf-8") if name_node else "AnonymousClass"

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
                "docstring": self.get_docstring(node),
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
            }
        )

    def _extract_function_info(self, node, is_method=False):
        """Collects signature data for a C++ function or method definition.

        Args:
            node (Node): function_definition node identified by tree-sitter.
            is_method (bool): Indicates whether the node originated within a class.

        Returns:
            Dict[str, Any]: Function metadata including parameters and docstring.
        """
        declarator_node = node.child_by_field_name("declarator")
        function_declarator = self._find_function_declarator(declarator_node)
        if function_declarator:
            name_node = function_declarator.child_by_field_name("declarator")
            if name_node and name_node.type == "identifier":
                name = name_node.text.decode("utf-8")
            else:
                name = "AnonymousFunction"

            params_node = function_declarator.child_by_field_name("parameters")
            parameters = self._parse_parameters(params_node)
        else:
            name = "AnonymousFunction"
            parameters = []

        return_type = self._extract_return_type(node)

        function_info = {
            "name": name,
            "parameters": parameters,
            "return_type": return_type,
            "docstring": self.get_docstring(node),
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
        }

        if not is_method:
            self.stats["functions"].append(function_info)

        return function_info

    def _parse_parameters(self, params_node):
        """Parses a parameter list into name and type components.

        Args:
            params_node (Node|None): Parameter list node from the grammar.

        Returns:
            List[Dict[str, Optional[str]]]: Collection of parsed parameter entries.
        """
        parameters = []
        if not params_node:
            return parameters

        for i in range(params_node.child_count):
            child = params_node.child(i)
            info = self._extract_parameter_info(child)
            if info:
                parameters.append(info)
        return parameters

    def _extract_parameter_info(self, param_node):
        """Extracts a parameter's identifier and associated type text.

        Args:
            param_node (Node): Parameter declaration node being analyzed.

        Returns:
            Optional[Dict[str, Optional[str]]]: Parameter descriptor or None.
        """
        if param_node.type not in {
            "parameter_declaration",
            "optional_parameter_declaration",
        }:
            return None

        name_node = self._find_identifier_in_node(param_node)
        if not name_node:
            return None

        name = name_node.text.decode("utf-8")
        param_text = param_node.text.decode("utf-8")
        idx = param_text.rfind(name)
        type_annotation = param_text[:idx].strip() if idx != -1 else ""
        return {"name": name, "type": type_annotation or ""}

    def _find_identifier_in_node(self, node):
        """Searches depth-first for an identifier node within the subtree.

        Args:
            node (Node): Parent node whose children should be scanned.

        Returns:
            Optional[Node]: Identifier node if found, otherwise None.
        """
        if node.type == "identifier":
            return node
        for i in range(node.child_count):
            child = node.child(i)
            found = self._find_identifier_in_node(child)
            if found:
                return found
        return None


class JavaAnalyzer(LanguageAnalyzer):
    """
    Analyzer for Java code.
    """

    def __init__(self, code: str, language: Language):
        """Initializes Java analyzer metadata collectors.

        Args:
            code (str): Java source code slated for analysis.
            language (Language): Tree-sitter Java language instance for parsing.

        Returns:
            None: Sets up dependency tracking and parses import directives.
        """
        super().__init__(code, language)
        self.stats.setdefault("outer_dependencies", [])
        self._collect_java_static_imports()

        # regular imports: import java.util.List;
        for m in re.finditer(r"import\s+([A-Za-z0-9_\.]+)\s*;", self.code):
            module = m.group(1)
            self.stats["outer_dependencies"].append({"module": module, "name": ""})

    def _collect_java_static_imports(self):
        """Indexes static import statements to note external dependencies.

        Args:
            None

        Returns:
            None: Updates the outer_dependencies list with static members.
        """
        # static imports bring methods into scope: import static a.b.Class.method;
        for m in re.finditer(
            r"import\s+static\s+([A-Za-z0-9_\.]+)\.([A-Za-z0-9_\*]+)\s*;", self.code
        ):
            module = m.group(1)
            name = m.group(2)
            if name == "*":
                # wildcard static import; can't enumerate, record module with '*'
                self.stats.setdefault("outer_dependencies", []).append(
                    {"module": module, "name": "*"}
                )
            else:
                self.stats.setdefault("outer_dependencies", []).append(
                    {"module": module, "name": name}
                )

    def _traverse(self, node):
        """Walks the Java AST to gather class and method declarations.

        Args:
            node (Node): Current AST node delivered by tree-sitter.

        Returns:
            None: Invokes helper extractors and continues recursion.
        """
        if node.type in ("class_declaration", "interface_declaration"):
            self._extract_class_info(node)
        elif node.type == "method_declaration":
            # Avoid capturing methods inside class traversal
            if not self._is_in_class(node):
                self._extract_function_info(node)

        for child in node.children:
            self._traverse(child)

    def _is_in_class(self, node):
        """Determines whether a method node resides inside a class or interface.

        Args:
            node (Node): Node whose parent chain is inspected.

        Returns:
            bool: True when a class or interface declaration is an ancestor.
        """
        parent = node.parent
        while parent:
            if parent.type in ("class_declaration", "interface_declaration"):
                return True
            parent = parent.parent
        return False

    def get_docstring(self, node):
        """Extracts a JavaDoc-style comment preceding the declaration.

        Args:
            node (Node): Declaration node potentially annotated with JavaDoc.

        Returns:
            Optional[str]: Cleaned comment block or None when absent.
        """
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
        """Captures metadata for a Java class or interface declaration.

        Args:
            node (Node): Class or interface node identified during traversal.

        Returns:
            None: Appends collected details to the stats dictionary.
        """
        name_node = node.child_by_field_name("name")
        class_name = name_node.text.decode("utf-8") if name_node else "AnonymousClass"

        methods = []
        body_node = node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                if child.type == "method_declaration":
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
        """Captures metadata for a Java method declaration.

        Args:
            node (Node): Method_declaration node produced by tree-sitter.
            is_method (bool): Indicates whether the node belongs to a class body.

        Returns:
            Dict[str, Any]: Function description including parameters and docs.
        """
        name_node = node.child_by_field_name("name")
        name = name_node.text.decode("utf-8") if name_node else "AnonymousMethod"

        params_node = node.child_by_field_name("parameters")
        parameters = self._parse_parameters(params_node)

        type_node = node.child_by_field_name("type")
        return_type = type_node.text.decode("utf-8") if type_node else ""

        function_info = {
            "name": name,
            "parameters": parameters,
            "return_type": return_type,
            "docstring": self.get_docstring(node),
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
        }

        if not is_method:
            self.stats["functions"].append(function_info)

        return function_info

    def _parse_parameters(self, params_node):
        """Parses Java formal parameters into structured metadata.

        Args:
            params_node (Node|None): formal_parameters node supplied by tree-sitter.

        Returns:
            List[Dict[str, Optional[str]]]: Extracted parameter representations.
        """
        parameters = []
        if not params_node:
            return parameters

        for i in range(params_node.child_count):
            child = params_node.child(i)
            info = self._extract_parameter_info(child)
            if info:
                parameters.append(info)
        return parameters

    def _extract_parameter_info(self, param_node):
        """Extracts argument name and type from a Java parameter node.

        Args:
            param_node (Node): formal_parameter or spread_parameter node.

        Returns:
            Optional[Dict[str, Optional[str]]]: Parameter metadata or None.
        """
        if param_node.type not in {"formal_parameter", "spread_parameter"}:
            return None

        name_node = self._find_identifier(param_node)
        if not name_node:
            return None

        name = name_node.text.decode("utf-8")
        param_text = param_node.text.decode("utf-8")
        idx = param_text.rfind(name)
        type_annotation = param_text[:idx].strip() if idx != -1 else ""
        return {"name": name, "type": type_annotation or ""}

    def _find_identifier(self, node):
        """Recursively searches for an identifier child within the node.

        Args:
            node (Node): Parent node whose subtree should be scanned.

        Returns:
            Optional[Node]: Identifier node if one is found; otherwise None.
        """
        if node.type == "identifier":
            return node
        for i in range(node.child_count):
            child = node.child(i)
            found = self._find_identifier(child)
            if found:
                return found
        return None


class RustAnalyzer(LanguageAnalyzer):
    """
    Analyzer for Rust code.
    """

    def __init__(self, code: str, language: Language):
        """Initializes Rust analyzer bookkeeping structures.

        Args:
            code (str): Rust source code scheduled for analysis.
            language (Language): Tree-sitter Rust language instance.

        Returns:
            None: Sets up dependency tracking and scans use statements.
        """
        super().__init__(code, language)
        self.stats.setdefault("outer_dependencies", [])
        self._collect_rust_uses()

    def _collect_rust_uses(self):
        """Parses use statements to record external module dependencies.

        Args:
            None

        Returns:
            None: Extends outer_dependencies with module and symbol data.
        """
        # collect use statements like: use crate::module::func;
        for m in re.finditer(
            r"use\s+([A-Za-z0-9_:]+)(::\{([^}]+)\})?::?([A-Za-z0-9_\*]+)?\s*;",
            self.code,
        ):
            # This regex captures several groups; we'll simplify extraction
            full = m.group(0)
            try:
                # try explicit patterns: use a::b::c::d;
                parts = full.replace("use", "").replace(";", "").strip().split()[-1]
                # split by '::'
                comps = parts.split("::")
                if comps:
                    if len(comps) > 1:
                        name = comps[-1]
                        module = "::".join(comps[:-1])
                    else:
                        module = comps[0]
                        name = ""
                    self.stats.setdefault("outer_dependencies", []).append(
                        {"module": module, "name": name}
                    )
            except Exception:
                continue

    def _traverse(self, node):
        """Walks the Rust AST to gather type and function declarations.

        Args:
            node (Node): Current AST node processed during traversal.

        Returns:
            None: Delegates to helper extractors and continues recursion.
        """
        if node.type in ("struct_item", "enum_item"):
            self._extract_class_info(node)
        elif node.type == "function_item":
            # Avoid capturing methods inside impl traversal
            if not self._is_in_impl(node):
                self._extract_function_info(node)

        for child in node.children:
            self._traverse(child)

    def _is_in_impl(self, node):
        """Checks whether a function node resides inside an impl block.

        Args:
            node (Node): Node whose parent chain is reviewed.

        Returns:
            bool: True when an impl_item ancestor exists; otherwise False.
        """
        parent = node.parent
        while parent:
            if parent.type == "impl_item":
                return True
            parent = parent.parent
        return False

    def get_docstring(self, node):
        """Extracts triple-slash documentation comments preceding the node.

        Args:
            node (Node): Declaration node potentially documented with /// lines.

        Returns:
            Optional[str]: Concatenated comment text or None when absent.
        """
        lines = self.code.split("\n")
        start_line = node.start_point[0]
        doc_lines = []
        for i in range(start_line - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith("///"):
                doc_lines.insert(0, line[3:].strip())
            elif line:
                break
        if doc_lines:
            return "\n".join(doc_lines)
        return ""

    def _extract_class_info(self, node):
        """Collects metadata for Rust struct or enum declarations.

        Args:
            node (Node): struct_item or enum_item node discovered during traversal.

        Returns:
            None: Appends declaration details to the analyzer statistics.
        """
        name_node = node.child_by_field_name("name")
        class_name = name_node.text.decode("utf-8") if name_node else "AnonymousStruct"

        # For Rust, methods are in impl blocks, not nested here
        methods = []

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
        """Collects metadata for Rust free functions.

        Args:
            node (Node): function_item node parsed by tree-sitter.
            is_method (bool): Indicates method context, though unused for Rust.

        Returns:
            Dict[str, Any]: Function description including signature information.
        """
        name_node = node.child_by_field_name("name")
        name = name_node.text.decode("utf-8") if name_node else "AnonymousFunction"

        params_node = node.child_by_field_name("parameters")
        parameters = self._parse_parameters(params_node)

        return_type_node = node.child_by_field_name("return_type")
        return_type = return_type_node.text.decode("utf-8") if return_type_node else ""

        function_info = {
            "name": name,
            "parameters": parameters,
            "return_type": return_type,
            "docstring": self.get_docstring(node),
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
        }

        if not is_method:
            self.stats["functions"].append(function_info)

        return function_info

    def _parse_parameters(self, params_node):
        """Parses Rust parameter lists into structured metadata.

        Args:
            params_node (Node|None): Parameters node obtained from tree-sitter.

        Returns:
            List[Dict[str, Optional[str]]]: Collected parameter descriptors.
        """
        parameters = []
        if not params_node:
            return parameters

        for i in range(params_node.child_count):
            child = params_node.child(i)
            info = self._extract_parameter_info(child)
            if info:
                parameters.append(info)
        return parameters

    def _extract_parameter_info(self, param_node):
        """Extracts name and type hints from Rust function parameters.

        Args:
            param_node (Node): Parameter node taken from the parameters list.

        Returns:
            Optional[Dict[str, Optional[str]]]: Parameter descriptor or None.
        """
        if param_node.type == "self_parameter":
            return {"name": "self", "type": ""}

        if param_node.type != "parameter":
            return None

        param_text = param_node.text.decode("utf-8").strip()
        name_node = self._find_identifier(param_node)
        name = name_node.text.decode("utf-8") if name_node else None

        type_annotation = ""
        if ":" in param_text:
            _, type_part = param_text.split(":", 1)
            type_annotation = type_part.strip() or ""

        if not name:
            pattern_part = (
                param_text.split(":", 1)[0].strip() if ":" in param_text else param_text
            )
            name = pattern_part if pattern_part else None

        return {"name": name, "type": type_annotation or ""} if name else None

    def _find_identifier(self, node):
        """Searches the subtree for the first identifier node.

        Args:
            node (Node): Parent node in which to locate identifiers.

        Returns:
            Optional[Node]: Identifier node if found; otherwise None.
        """
        if node.type == "identifier":
            return node
        for i in range(node.child_count):
            child = node.child(i)
            found = self._find_identifier(child)
            if found:
                return found
        return None


def get_analyzer(language: Language) -> type[LanguageAnalyzer]:
    """Resolves the analyzer class that matches the provided language.

    Args:
        language (Language): Tree-sitter language instance describing the file.

    Returns:
        type[LanguageAnalyzer]: Analyzer subclass prepared for that language.
    """
    if language == Language(tspython.language()):
        return PythonAnalyzer
    if language == Language(tsjavascript.language()):
        return JavaScriptAnalyzer
    if language == Language(tsgo.language()):
        return GoAnalyzer
    if language == Language(tscpp.language()):
        return CppAnalyzer
    if language == Language(tsjava.language()):
        return JavaAnalyzer
    if language == Language(tsrust.language()):
        return RustAnalyzer
    # Return a default analyzer that does nothing if language is not supported
    return LanguageAnalyzer


def analyze_code_from_string(
    code_string: str, language: Language
) -> Optional[Dict[str, Any]]:
    """Analyzes a code snippet string and produces structural statistics.

    Args:
        code_string (str): Source code content to inspect.
        language (Language): Tree-sitter language used for parsing.

    Returns:
        Optional[Dict[str, Any]]: Aggregated metadata or None on failure.
    """
    try:
        AnalyzerClass = get_analyzer(language)
        analyzer = AnalyzerClass(code_string, language)
        return analyzer.analyze()
    except Exception as e:
        return None


def analyze_file_with_tree_sitter(file_path: str) -> Optional[Dict[str, Any]]:
    """Analyzes a file on disk and returns structured tree-sitter insights.

    Args:
        file_path (str): Path to the source file that should be read.

    Returns:
        Optional[Dict[str, Any]]: Structured analysis data or None on error.
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
    """Normalizes raw analysis statistics into a summarized dictionary.

    Args:
        stats (Dict[str, Any]): Raw statistics captured by the analyzers.

    Returns:
        Dict[str, Any]: Summary dictionary ready for downstream consumption.
    """
    if not stats:
        return {}

    summary = {
        "classes": [
            {"name": cls["name"], "description": cls.get("docstring", "")}
            for cls in stats.get("classes", [])
        ],
        "functions": [
            {
                "name": func["name"],
                "description": func.get("docstring", ""),
                "return_type": func.get("return_type", ""),
            }
            for func in stats.get("functions", [])
        ],
    }

    if "outer_dependencies" in stats:
        # ensure consistent simple representation
        summary["outer_dependencies"] = [
            {"module": d.get("module", ""), "name": d.get("name", "")}
            for d in stats.get("outer_dependencies", [])
        ]

    formatted_stats = {"summary": summary}
    formatted_stats.update(stats)
    return formatted_stats


def format_tree_sitter_analysis_results_to_prompt(stats: Dict[str, Any]) -> str:
    """Formats analysis statistics into a human-readable string for prompts.

    Args:
        stats (Dict[str, Any]): Raw statistics captured by the analyzers.
    Returns:
        str: Formatted string representation of the analysis.
    """
    # this func format the results to prompt str to prevent token overflows
    if not stats:
        return "No analysis data available."
    lines = []

    # Format classes compactly
    classes = stats.get("classes", [])
    if classes:
        class_items = []
        for cls in classes:
            doc = (cls.get("docstring") or "").replace("\n", " ").strip()
            doc_str = f" - {doc}" if doc else ""
            class_items.append(f"{cls['name']}{doc_str}")
        lines.append("Classes: " + "; ".join(class_items))

    # Format functions compactly
    functions = stats.get("functions", [])
    if functions:
        func_items = []
        for func in functions:
            doc = (func.get("docstring") or "").replace("\n", " ").strip()
            ret_type = func.get("return_type", "")
            ret_str = f" -> {ret_type}" if ret_type else ""
            doc_str = f" - {doc}" if doc else ""
            func_items.append(f"{func['name']}{ret_str}{doc_str}")
        lines.append("Functions: " + "; ".join(func_items))

    # Format dependencies compactly
    deps = stats.get("outer_dependencies", [])
    if deps:
        dep_items = []
        for dep in deps:
            module = dep.get("module", "")
            name = dep.get("name", "")
            dep_str = f"{module}.{name}" if name else module
            if dep_str and dep_str not in dep_items:  # avoid duplicates
                dep_items.append(dep_str)
        if dep_items:
            lines.append("Dependencies: " + ", ".join(dep_items))

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    import json

    # use "uv run python -m utils.code_analyzer <file_path>" to test
    if len(sys.argv) < 2:
        print("Usage: uv run utils/code_analyzer.py <file_path>")
        sys.exit(1)

    path = sys.argv[1]
    if os.path.isfile(path):
        file_stats = analyze_file_with_tree_sitter(path)
        if file_stats:
            # print(format_tree_sitter_analysis_results(file_stats))
            # write to a json file
            # with open("analysis_results.json", "w", encoding="utf-8") as f:
            #     json.dump(format_tree_sitter_analysis_results(file_stats), f, indent=2)
            # print("Analysis results written to analysis_results.json")
            # write to a txt file
            with open("analysis_results.txt", "w", encoding="utf-8") as f:
                f.write(format_tree_sitter_analysis_results_to_prompt(file_stats))
            print("Analysis results written to analysis_results.txt")
            # print(format_tree_sitter_analysis_results_to_prompt(file_stats))
            # print(json.dumps(format_tree_sitter_analysis_results(file_stats), indent=2))
    else:
        print("Please provide a valid file path.")
