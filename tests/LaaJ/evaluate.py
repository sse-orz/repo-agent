"""
This file is used to evaluate the cost time and quality of the documentation.

- Few-shot evaluation: use ICL examples to evaluate the documentation.
- QAFactEval: use QAFactEval method to evaluate the documentation.

@misc{fabbri-etal-2022-qafacteval,
    title = {QAFactEval: Improved QA-Based Factual Consistency Evaluation for Summarization},
    author = {Alexander R. Fabbri and Chien-Sheng Wu and Wenhao Liu and Caiming Xiong},
    year={2022},
    eprint={2112.08542},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url = {https://arxiv.org/abs/2112.08542},
}
"""

import os
import json
import traceback
from time import time
from textwrap import dedent
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional

from agents.rag_agent import RAGAgent
from app.models.rag import RAGAnswerData
from agents.sub_graph.parent import ParentGraphBuilder
from config import CONFIG


@dataclass
class EvaluateInput:
    owner: str
    repo: str
    platform: str
    mode: str
    max_workers: int
    wiki_directory: str
    repo_directory: str
    branch_mode: str
    ratios: dict[str, float]
    question: str


class EvaluateAgent:
    # Class constants
    SPECIAL_DOC_TYPES = [
        "pr_documentation.md",
        "overview_documentation.md",
        "commit_documentation.md",
        "release_note_documentation.md",
    ]

    CODE_EXTENSIONS = [
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cc",
        ".py",
        ".js",
        ".ts",
        ".java",
        ".go",
        ".rs",
    ]

    def __init__(self, inputs: EvaluateInput):
        self.sub_graph_builder = ParentGraphBuilder(branch_mode=inputs.branch_mode)
        self.rag_agent = RAGAgent()
        self.inputs = inputs
        self.llm = CONFIG.get_llm()
        self.num_questions = 5  # for QAFactEval

    @staticmethod
    def is_special_documentation(doc_path: str) -> bool:
        """Check if a documentation file is a special type that doesn't need a code file."""
        filename = os.path.basename(doc_path)
        return any(
            filename == special_type for special_type in EvaluateAgent.SPECIAL_DOC_TYPES
        )

    @staticmethod
    def _parse_llm_json_response(content: str) -> Optional[dict]:
        """Parse JSON from LLM response, handling cases where JSON is embedded in text."""
        try:
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _extract_llm_content(response) -> str:
        """Extract content from LLM response object."""
        return response.content if hasattr(response, "content") else str(response)

    @staticmethod
    def _calculate_word_similarity(text1: str, text2: str) -> float:
        """Calculate similarity score based on word overlap (0-100)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        overlap = len(words1 & words2)
        total_unique = len(words1 | words2)
        if total_unique == 0:
            return 0.0
        return round((overlap / total_unique) * 100, 2)

    @staticmethod
    def _is_failed_answer(answer: str) -> bool:
        """Check if an answer indicates failure."""
        failure_phrases = [
            "Not mentioned in the documentation",
            "Failed to extract answer from the documentation",
        ]
        return any(phrase in answer for phrase in failure_phrases)

    def _create_result(
        self,
        doc_path: str,
        code_path: Optional[str],
        evaluation: dict,
        method: str = "Unknown",
    ) -> dict:
        """Create a standardized result dictionary."""
        return {
            "doc_path": doc_path,
            "code_path": code_path if code_path else None,
            "evaluation": evaluation,
            "evaluation_method": method,
            "timestamp": datetime.now().isoformat(),
        }

    def _create_error_result(
        self, doc_path: str, code_path: Optional[str], error: str, method: str = "Error"
    ) -> dict:
        """Create a standardized error result dictionary."""
        return self._create_result(
            doc_path, code_path, {"score": None, "error": error}, method
        )

    def _read_file_safely(self, file_path: str) -> Optional[str]:
        """Safely read a file and return its content, or None if it fails."""
        if not file_path or not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[WARN] Failed to read file {file_path}: {e}")
            return None

    def run_sub_graph(self) -> float:
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        start_time = time()
        self.sub_graph_builder.run(
            inputs={
                "owner": self.inputs.owner,
                "repo": self.inputs.repo,
                "platform": self.inputs.platform,
                "mode": self.inputs.mode,  # "fast" or "smart"
                "ratios": self.inputs.ratios,
                "max_workers": self.inputs.max_workers,
                "date": date,
                "wiki_root_path": self.inputs.wiki_directory,
                "log": False,
            },
            config={
                "configurable": {
                    "thread_id": f"wiki-generation-{date}",
                }
            },
            count_time=True,
        )
        end_time = time()
        elapsed_time = end_time - start_time
        return elapsed_time

    def run_rag_agent(self) -> tuple[RAGAnswerData, float]:
        start_time = time()
        repo_name = f"{self.inputs.platform}:{self.inputs.owner}/{self.inputs.repo}"
        repo_dir = f"{self.inputs.owner}_{self.inputs.repo}"
        self.rag_agent.init_repo(repo_dir)
        data = self.rag_agent.ask(self.inputs.question, repo_name, repo_dir)
        end_time = time()
        elapsed_time = end_time - start_time
        return data, elapsed_time

    def get_icl_examples(self) -> list[dict]:
        return [
            {
                "documentation": dedent(
                    """
                    Calculates the sum of two numbers.
                    
                    Args:
                        a (int): The first number to add.
                        b (int): The second number to add.
                    
                    Returns:
                        int: The sum of a and b.
                    
                    Example:
                        >>> add(3, 5)
                        8
                        >>> add(-1, 1)
                        0
                """
                ).strip(),
                "evaluation": {
                    "score": 92,
                    "aspects": {
                        "clarity": "Very clear and unambiguous. The function purpose is immediately obvious from the first line.",
                        "completeness": "Excellent completeness. Includes function description, parameter types and descriptions, return type and description, and practical usage examples.",
                        "accuracy": "Accurate description with correct parameter and return type information. Examples demonstrate correct usage.",
                        "conciseness": "Concise and well-structured. No unnecessary words, follows standard documentation format efficiently.",
                    },
                },
            },
            {
                "documentation": "This does some math stuff with numbers. It takes inputs and gives you a result back. You can use it to add things together. The function is pretty useful for calculations. It works with integers and returns an integer value. Make sure to pass the right types or it might not work correctly.",
                "evaluation": {
                    "score": 35,
                    "aspects": {
                        "clarity": "Vague and unclear. Uses imprecise language like 'some math stuff' and 'things' instead of specific descriptions.",
                        "completeness": "Poor completeness. Missing parameter names, types, and descriptions. No return value specification. No usage examples. Contains redundant information.",
                        "accuracy": "Partially accurate but lacks precision. Warning about types is helpful but vague. Does not specify what happens with wrong types.",
                        "conciseness": "Not concise at all. Contains redundant phrases and repetitive information. Could be condensed to a fraction of the length while conveying more useful information.",
                    },
                },
            },
        ]

    def get_sub_graph_prompt(
        self, icl_examples: list[dict], documentation: str, code_content: str = ""
    ) -> str:
        icl_examples_content = ""
        for i, example in enumerate(icl_examples, 1):
            icl_examples_content += (
                dedent(
                    f"""
                Example {i}:
                Documentation Content:
                {example["documentation"]}
                Evaluation Result:
                {json.dumps(example["evaluation"], ensure_ascii=False, indent=2)}
                """
                ).strip()
                + "\n\n"
            )

        code_section = ""
        if code_content:
            code_section = f"""
            Code File Content:
            {code_content}
            """

        content = dedent(
            f"""
            You are a professional documentation quality evaluation expert. Please evaluate the quality of the documentation based on the following criteria:

            Evaluation Criteria:
            1. Clarity: Whether the documentation is clear and easy to understand, with unambiguous expression
            2. Completeness: Whether it contains necessary information (such as function description, parameter explanations, return values, usage examples, etc.)
            3. Accuracy: Whether the description is accurate, and whether there are errors or misleading information
            4. Conciseness: Whether it is concise and to the point, avoiding unnecessary redundancy

            Scoring Scale: 1-100 points (100 being the highest)

            Please refer to the following examples:

            {icl_examples_content}

            Now evaluate the documentation:
            Documentation Content:
            {documentation}
            {code_section}
            ---
            Return a JSON object **with no additional text**, following this exact structure:
            {{
                "score": <1-100 integer>,
                "aspects": {{
                    "clarity": "<assessment with reasoning>",
                    "completeness": "<assessment with reasoning>",
                    "accuracy": "<assessment with reasoning; must reference code behavior if code is provided>",
                    "conciseness": "<assessment with reasoning>"
                }}
            }}
            Requirements:
            - If code is provided: All judgments **must be grounded directly in the provided code**. If the documentation describes functionality not present in the code, flag it as inaccurate. If the documentation misses important behaviors shown in the code, flag it as incomplete.
            - If no code is provided: Evaluate based on the documentation's internal consistency, clarity, and completeness. For accuracy, assess whether the documentation is logically sound and well-structured.
            - Be objective, specific, and technical.
            - Do not include commentary outside the JSON.
            """
        ).strip()
        return content

    def get_documentation_list(self) -> List[Tuple[str, str]]:
        """
        Get list of documentation files and their corresponding code files.
        Returns list of tuples: (doc_path, code_path)
        For special documentation types (pr, overview, commit, release_note), code_path will be empty string.
        """
        repo_dirname = f"{self.inputs.owner}_{self.inputs.repo}"
        wiki_path = os.path.join(self.inputs.wiki_directory, repo_dirname)
        repo_path = os.path.join(self.inputs.repo_directory, repo_dirname)
        return self._get_documentation_list(wiki_path, repo_path, verbose=True)

    def _get_documentation_list(
        self,
        wiki_path: str,
        repo_path: str,
        verbose: bool = True,
    ) -> List[Tuple[str, str]]:
        """Internal method to get documentation list with code file mappings."""
        documentation_pairs = []

        if not os.path.exists(wiki_path):
            if verbose:
                print(f"[WARN] Documentation path does not exist: {wiki_path}")
            return documentation_pairs

        # traverse the documentation directory
        for root, dirs, files in os.walk(wiki_path):
            for file in files:
                if not file.endswith(".md"):
                    continue

                doc_path = os.path.join(root, file)
                matched_code_path = ""

                # check if it is a special documentation type
                if self.is_special_documentation(doc_path):
                    if verbose:
                        print(
                            f"[INFO] Found special documentation (no code file needed): {doc_path}"
                        )
                    documentation_pairs.append((doc_path, matched_code_path))
                    continue

                # for regular documentation, try to find the corresponding code file
                # process the documentation filename: remove _doc.md or .md suffix
                doc_base = (
                    os.path.basename(doc_path).replace("_doc.md", "").replace(".md", "")
                )

                # search for matching code files
                # e.g. doc_path: ./.wikis/sub/facebook_zstd/lib/compress/zstd_lazy.c_doc.md
                # wiki_path: ./.wikis/sub/facebook_zstd
                # matched_code_path should be ./.repos/facebook_zstd/lib/compress/zstd_lazy.c
                # Get relative path from wiki_path to doc_path's directory
                doc_dir = os.path.dirname(doc_path)
                try:
                    # Get relative path from wiki_path to doc_dir
                    rel_path = os.path.relpath(doc_dir, wiki_path)
                    # If rel_path is ".", it means doc_dir is the same as wiki_path
                    if rel_path == ".":
                        matched_code_path = os.path.join(repo_path, doc_base)
                    else:
                        matched_code_path = os.path.join(repo_path, rel_path, doc_base)
                except ValueError:
                    # If paths are on different drives (Windows), fallback to old method
                    matched_code_path = os.path.join(
                        repo_path, os.path.dirname(doc_path), doc_base
                    )
                documentation_pairs.append((doc_path, matched_code_path))

        return documentation_pairs

    # QAFactEval methods
    def _generate_questions(self, code_content: str) -> List[str]:
        """Generate questions from source code for QAFactEval evaluation."""
        prompt = dedent(
            f"""
        Generate {self.num_questions} key questions based on the following source code, which should be able to verify whether the documentation accurately describes the functionality of the code.
        
        The questions should focus on:
        1. The main functionality of the function/class
        2. The parameter types and meanings
        3. The return value types and meanings
        4. The key behaviors or logic
        5. The boundary cases or special handling
        
        Source code:
        {code_content}
        
        Return the questions in a list format:
        {{
            "questions": [
                "question1",
                "question2",
                ...
            ]
        }}
        """
        ).strip()

        try:
            response = self.llm.invoke(prompt)
            content = self._extract_llm_content(response)

            # Try to parse the JSON
            result = self._parse_llm_json_response(content)
            if result:
                return result.get("questions", [])

            # If the parsing fails, try to extract the questions from the text
            lines = content.split("\n")
            questions = [
                line.strip()
                for line in lines
                if line.strip() and ("?" in line or line.startswith("-"))
            ]
            return questions[: self.num_questions]
        except Exception as e:
            print(f"[WARN] Failed to generate questions: {e}")
            return []

    def _extract_answer_from_doc(self, question: str, documentation: str) -> str:
        """Extract answer from documentation based on a question."""
        prompt = dedent(
            f"""
        Answer the following question based on the following documentation content:
        
        Question: {question}
        
        Documentation content:
        {documentation}
        
        Please directly give the answer. If there is no relevant information in the documentation, please answer "Not mentioned in the documentation".
        """
        ).strip()

        try:
            response = self.llm.invoke(prompt)
            content = self._extract_llm_content(response)
            return content.strip()
        except Exception as e:
            print(f"[WARN] Failed to extract answer from doc: {e}")
            return "Failed to extract answer from the documentation"

    def _extract_answer_from_code(self, question: str, code_content: str) -> str:
        """Extract answer from source code based on a question."""
        prompt = dedent(
            f"""
        Answer the following question based on the following source code:
        
        Question: {question}
        
        Source code:
        {code_content}
        
        Please directly give the answer, based on the actual implementation of the code.
        """
        ).strip()

        try:
            response = self.llm.invoke(prompt)
            content = self._extract_llm_content(response)
            return content.strip()
        except Exception as e:
            print(f"[WARN] Failed to extract answer from code: {e}")
            return "Failed to extract answer from the code"

    def _compare_answers(
        self, question: str, doc_answer: str, code_answer: str
    ) -> float:
        """
        Compare answers and return a matching score (0-100).

        Returns:
            float: Matching score from 0 to 100, where 100 means perfect match
        """
        prompt = dedent(
            f"""
        Evaluate the matching degree between the following two answers (0-100 score):
        
        Question: {question}
        
        Answer1 (from documentation): {doc_answer}
        Answer2 (from code): {code_answer}
        
        Please evaluate the semantic matching degree between these two answers and give a score from 0 to 100.
        - 100: Perfect match, answers express exactly the same meaning
        - 80-99: Very high match, minor differences in wording
        - 60-79: High match, similar meaning with some differences
        - 40-59: Moderate match, partially consistent
        - 20-39: Low match, mostly inconsistent
        - 0-19: Very low match or completely inconsistent
        
        If Answer1 clearly says "Not mentioned in the documentation" or "Failed to extract answer from the documentation", then return 0.
        
        Please only return the JSON format:
        {{
            "score": <0-100 integer>,
            "reason": "Briefly explain the reason"
        }}
        """
        ).strip()

        try:
            response = self.llm.invoke(prompt)
            content = self._extract_llm_content(response)

            # Parse the JSON
            result = self._parse_llm_json_response(content)
            if result:
                score = result.get("score", 0)
                # Ensure score is in valid range
                return max(0, min(100, float(score)))
        except Exception as e:
            print(f"[WARN] Failed to compare answers: {e}")

        # Fallback: use heuristic similarity calculation
        return self._calculate_answer_similarity(doc_answer, code_answer)

    def _calculate_answer_similarity(self, doc_answer: str, code_answer: str) -> float:
        """Calculate similarity score using word overlap heuristic."""
        if self._is_failed_answer(doc_answer):
            return 0.0
        return self._calculate_word_similarity(doc_answer, code_answer)

    def _evaluate_with_qafacteval(
        self, doc_path: str, code_path: str, documentation: str, source_document: str
    ) -> dict:
        """Use QAFactEval method to evaluate (for code documentation)"""
        try:
            # Step 1: Generate questions from the source code
            questions = self._generate_questions(source_document)

            if not questions:
                return self._create_error_result(
                    doc_path, code_path, "Failed to generate questions", "QAFactEval"
                )

            # Step 2 & 3: Extract answers from the documentation and code, and calculate matching scores
            qa_results = []
            scores = []

            for question in questions:
                doc_answer = self._extract_answer_from_doc(question, documentation)
                code_answer = self._extract_answer_from_code(question, source_document)
                matching_score = self._compare_answers(
                    question, doc_answer, code_answer
                )

                scores.append(matching_score)

                qa_results.append(
                    {
                        "question": question,
                        "doc_answer": doc_answer,
                        "code_answer": code_answer,
                        "matching_score": matching_score,
                    }
                )

            # Step 4: Calculate the average matching score (0-100)
            average_score = sum(scores) / len(scores) if scores else 0

            evaluation = {
                "score": round(average_score, 2),
                "num_questions": len(questions),
                "average_score": round(average_score, 2),
                "qa_pairs": qa_results,
            }

            return self._create_result(doc_path, code_path, evaluation, "QAFactEval")

        except Exception as e:
            print(f"[ERROR] QAFactEval evaluation failed for {doc_path}: {e}")
            traceback.print_exc()
            return self._create_error_result(doc_path, code_path, str(e), "QAFactEval")

    def _evaluate_with_icl(
        self,
        doc_path: str,
        code_path: str,
        documentation: str,
        code_content: str,
        icl_examples: List[dict],
    ) -> dict:
        """Use ICL method to evaluate (for special documentation types)"""
        # Build evaluation prompt
        prompt = self.get_sub_graph_prompt(icl_examples, documentation, code_content)

        # Call LLM to evaluate
        llm = CONFIG.get_llm()
        response = llm.invoke(prompt)

        # Parse response
        content = self._extract_llm_content(response)

        # Try to extract JSON from the response
        evaluation = self._parse_llm_json_response(content)
        if not evaluation:
            print(f"[ERROR] Failed to parse JSON for {doc_path}")
            print(f"[DEBUG] Response content: {content[:500]}")
            evaluation = {
                "score": 0,
                "aspects": {
                    "clarity": "Failed to parse LLM response",
                    "completeness": "Failed to parse LLM response",
                    "accuracy": "Failed to parse LLM response",
                    "conciseness": "Failed to parse LLM response",
                },
                "error": "Failed to parse JSON response",
            }

        return self._create_result(doc_path, code_path, evaluation, "ICL")

    def evaluate_single_documentation(
        self, doc_path: str, code_path: str, icl_examples: List[dict]
    ) -> dict:
        """
        Evaluate a single documentation file.
        - For code documentation: use QAFactEval method
        - For special documentation (pr, overview, commit, release_note): use ICL method
        """
        try:
            # Read documentation content
            documentation = self._read_file_safely(doc_path)
            if not documentation:
                return self._create_error_result(
                    doc_path, code_path, "Failed to read documentation file", "Error"
                )

            # check if it is a special documentation type
            is_special = self.is_special_documentation(doc_path)

            if is_special or not code_path:
                # special documentation type or no code file: use ICL method
                code_content = self._read_file_safely(code_path) or ""
                return self._evaluate_with_icl(
                    doc_path, code_path, documentation, code_content, icl_examples
                )
            else:
                # with code file: use QAFactEval method
                source_document = self._read_file_safely(code_path)
                if not source_document:
                    return self._create_result(
                        doc_path,
                        code_path,
                        {
                            "score": None,
                            "note": "No source code available for QAFactEval evaluation",
                        },
                        "QAFactEval",
                    )

                return self._evaluate_with_qafacteval(
                    doc_path, code_path, documentation, source_document
                )

        except Exception as e:
            print(f"[ERROR] Error evaluating {doc_path}: {e}")
            return self._create_error_result(doc_path, code_path, str(e), "Error")

    def run_evaluate_sub_graph_documentation(self) -> dict:
        """
        Evaluate all documentation files generated by sub_graph.
        Returns evaluation results.
        """
        # 1. load documentation files
        documentation_list = self.get_documentation_list()
        if not documentation_list:
            print("[WARN] No documentation files found.")
            return {
                "evaluation_time": datetime.now().isoformat(),
                "total_docs": 0,
                "results": [],
                "average_score": 0,
            }

        print(
            f"[INFO] Found {len(documentation_list)} documentation files to evaluate."
        )

        # 2. get ICL examples
        icl_examples = self.get_icl_examples()

        # 3. evaluate documentation files (with threading for parallel processing)
        results = []
        with ThreadPoolExecutor(max_workers=self.inputs.max_workers) as executor:
            # submit all evaluation tasks
            future_to_doc = {
                executor.submit(
                    self.evaluate_single_documentation,
                    doc_path,
                    code_path,
                    icl_examples,
                ): (doc_path, code_path)
                for doc_path, code_path in documentation_list
            }

            # process completed tasks
            for future in as_completed(future_to_doc):
                doc_path, code_path = future_to_doc[future]
                try:
                    result = future.result()
                    results.append(result)
                    score = result["evaluation"].get("score", 0)
                    method = result.get("evaluation_method", "Unknown")
                    print(
                        f"[INFO] Evaluated: {os.path.basename(doc_path)} - Score: {score} (Method: {method})"
                    )
                except Exception as e:
                    print(f"[ERROR] Exception evaluating {doc_path}: {e}")
                    results.append(
                        self._create_error_result(doc_path, code_path, str(e), "Error")
                    )

        # 4. calculate statistics
        valid_scores = [
            r["evaluation"].get("score", 0)
            for r in results
            if r["evaluation"].get("score") is not None
        ]
        average_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

        evaluation_results = {
            "evaluation_time": datetime.now().isoformat(),
            "total_docs": len(documentation_list),
            "results": results,
            "average_score": average_score,
        }

        print(f"\n[INFO] Documentation evaluation complete!")
        print(f"[INFO] Total documents: {evaluation_results['total_docs']}")
        print(f"[INFO] Average score: {evaluation_results['average_score']:.2f}")

        return evaluation_results

    def evaluate(self, enable_run: bool = False):
        if enable_run:
            sub_graph_time = self.run_sub_graph()
            rag_agent_data, rag_agent_time = self.run_rag_agent()
        else:
            sub_graph_time = 0
            rag_agent_data = None
            rag_agent_time = 0
        documentation_evaluation = self.run_evaluate_sub_graph_documentation()
        self.save_results(
            sub_graph_time, rag_agent_data, rag_agent_time, documentation_evaluation
        )

    def save_results(
        self,
        sub_graph_time,
        rag_agent_data,
        rag_agent_time,
        documentation_evaluation=None,
    ):
        if rag_agent_data:
            rag_agent_data_dict = rag_agent_data.model_dump()
        else:
            rag_agent_data_dict = None

        results = {
            "repo_name": f"{self.inputs.platform}:{self.inputs.owner}/{self.inputs.repo}",
            "repo_dir": f"{self.inputs.owner}_{self.inputs.repo}",
            "mode": self.inputs.mode,
            "max_workers": self.inputs.max_workers,
            "wiki_directory": self.inputs.wiki_directory,
            "branch_mode": self.inputs.branch_mode,
            "ratios": self.inputs.ratios,
            "question": self.inputs.question,
            "sub_graph_time": sub_graph_time,
            "rag_agent_time": rag_agent_time,
            "rag_agent_data": rag_agent_data_dict,
            "documentation_evaluation": documentation_evaluation,
            "llm": {
                "platform": CONFIG.LLM_PLATFORM,
                "model": CONFIG.LLM_MODEL,
            },
        }
        output_dir = f".evaluate/{self.inputs.owner}_{self.inputs.repo}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "evaluate_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Results saved to: {output_path}")


if __name__ == "__main__":
    evaluate_inputs = EvaluateInput(
        owner="xudong7",
        repo="tauri-rbook",
        platform="github",
        mode="fast",
        max_workers=10,
        wiki_directory="./.wikis/evaluate",
        repo_directory="./.repos",
        branch_mode="all",
        ratios={
            "fast": 0.05,
            "smart": 0.75,
        },
        question="What is this repo about?",
    )
    evaluate_agent = EvaluateAgent(inputs=evaluate_inputs)
    evaluate_agent.evaluate(enable_run=True)
