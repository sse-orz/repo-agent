import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import CONFIG
from typing import List, Tuple


class LaaJAgent:  # llm as a judge
    def __init__(self, owner: str = "facebook", repo: str = "zstd", dir: str = ""):
        self.wiki_path = f"./.wikis/{owner}_{repo}/{dir}"
        self.code_path = f"./.repos/{owner}_{repo}/lib/{dir}"
        self.output_file = "./evaluation_results.out"

    # in-context learning examples
    def get_icl_examples(self) -> List[Dict]:
        """
        Get In-Context Learning examples
        代码文件和文档本身都要给，评价内容正确性
        """
        return [
            {
                "documentation": "This function calculates the sum of two numbers.",
                "evaluation": {
                    "score": 9,
                    "aspects": {
                        "clarity": "Very clear, directly states the function's purpose.",
                        "completeness": "Describes the core functionality but does not mention parameters or return value.",
                        "accuracy": "Accurate and correct.",
                        "conciseness": "Concise and to the point.",
                        "context_awareness": "Limited; states purpose but not how it fits into surrounding logic.",
                        "technical_precision": "Basic but correct; lacks details on input types.",
                        "limitations_disclosure": "None; for simple functions this may be acceptable.",
                        "tone_and_style": "Professional and straightforward.",
                    },
                    "overall_feedback": "Excellent documentation that clearly and accurately describes the function's purpose.",
                    "improvement_suggestions": [
                        "Add parameter descriptions.",
                        "Include return value explanation.",
                    ],
                },
            },
            {
                "documentation": "Function that does stuff.",
                "evaluation": {
                    "score": 3,
                    "aspects": {
                        "clarity": "Very vague and unclear about specific functionality.",
                        "completeness": "Missing critical details such as parameters, return values, and purpose.",
                        "accuracy": "Too generic to be meaningful or accurate.",
                        "conciseness": "Overly simplistic to the point of being uninformative.",
                        "context_awareness": "None; provides no indication of use case or context.",
                        "technical_precision": "Very poor; no technical detail at all.",
                        "limitations_disclosure": "None; likely because the functionality is not described.",
                        "tone_and_style": "Neutral but too vague to be useful.",
                    },
                    "overall_feedback": "Poor quality documentation lacking specific and actionable information.",
                    "improvement_suggestions": [
                        "Clearly specify the function's exact purpose.",
                        "Add parameter and return value descriptions.",
                        "Provide usage examples.",
                    ],
                },
            },
            {
                "documentation": "The `RetryingClient` wraps outbound HTTP requests and retries them using exponential backoff. If a request fails due to network errors or 5xx responses, it will attempt up to 5 retries before surfacing the exception.",
                "evaluation": {
                    "score": 8,
                    "aspects": {
                        "clarity": "Clear description of when retries occur and the upper retry bound.",
                        "completeness": "Good coverage of behavior, but missing details on jitter strategy or retry interval configuration.",
                        "accuracy": "Accurate explanation of exponential backoff and error conditions.",
                        "conciseness": "Compact without unnecessary filler.",
                        "context_awareness": "Explains relationship to outbound HTTP operations.",
                        "technical_precision": "Solid, but could specify retry timing parameters.",
                        "limitations_disclosure": "Mentions retry cap but not the lack of circuit-breaking.",
                        "tone_and_style": "Neutral and professional.",
                    },
                    "overall_feedback": "Well-written operational documentation that clearly communicates retry behavior. Adding configurability notes and jitter details would improve completeness.",
                    "improvement_suggestions": [
                        "Mention whether jitter is applied.",
                        "Clarify retry interval defaults and configurability.",
                        "Call out lack of circuit-breaker or cooldown mechanisms.",
                    ],
                },
            },
            {
                "documentation": "This file contains a function `process_data` that takes a list of items and saves them to the database. It loops through the list and commits each item.",
                "evaluation": {
                    "score": 6,
                    "aspects": {
                        "context_awareness": "Low. Doesn't explain what kind of data or why it's being saved.",
                        "technical_precision": "Average. Describes the loop but misses details (Batch commit? Transactional?).",
                        "limitations_disclosure": "Missing. Does not mention performance implications of committing inside a loop.",
                        "tone_and_style": "Acceptable but slightly repetitive.",
                    },
                    "overall_feedback": "Functional but minimal. It describes the code literally but lacks architectural context and performance warnings.",
                    "improvement_suggestions": [
                        "Specify the database interaction model (bulk insert vs single insert).",
                        "Mention the expected data type of the input list.",
                        "Add context about error handling behavior.",
                    ],
                },
            },
            {
                "documentation": "A powerful and easy-to-use helper for handling user images. It magically resizes everything for you so you don't have to worry about it. Best used for profile pictures.",
                "evaluation": {
                    "score": 3,
                    "aspects": {
                        "context_awareness": "Vague. 'Handling user images' is too broad.",
                        "technical_precision": "Poor. Uses fluff words like 'powerful', 'easy-to-use', 'magically'. No mention of algorithms or formats.",
                        "limitations_disclosure": "None. Implies it works for 'everything' which is dangerous.",
                        "tone_and_style": "Subjective and unprofessional. Avoid marketing language in code comments.",
                    },
                    "overall_feedback": "Fails to provide actionable information for a developer. 'Magically' is a red flag in technical documentation.",
                    "improvement_suggestions": [
                        "Remove subjective adjectives (powerful, easy, beautiful).",
                        "Specify supported image formats (JPG, PNG?).",
                        "Explain the resizing algorithm (Lanczos, Bilinear?).",
                        "Define inputs and outputs clearly.",
                    ],
                },
            },
            {
                "documentation": "This module exports `normalize_email`, which trims whitespace, lowercases domains, and validates basic RFC-compliant structure. It does not attempt full RFC 5322 compliance.",
                "evaluation": {
                    "score": 9,
                    "aspects": {
                        "clarity": "Very clear; developer immediately understands what the function does.",
                        "completeness": "Covers scope, capabilities, and boundaries of validation.",
                        "accuracy": "Correctly describes the distinction between basic parsing and full RFC compliance.",
                        "conciseness": "Minimal yet fully informative.",
                        "context_awareness": "Explicit about its purpose in pre-processing user emails.",
                        "technical_precision": "Strong. Avoids overstating capabilities.",
                        "limitations_disclosure": "Excellent; explicitly states what is *not* handled.",
                        "tone_and_style": "Professional, factual.",
                    },
                    "overall_feedback": "Excellent function-level documentation. It sets clear expectations and prevents misuse by clarifying validation limits.",
                    "improvement_suggestions": [
                        "Include examples of accepted and rejected inputs.",
                        "Specify whether internationalized domain names (IDN) are supported.",
                    ],
                },
            },
            {
                "documentation": "Helper method `serialize_user` converts a User model into a JSON-serializable dictionary. It includes id, name, roles, and timestamps. Internal-only fields (password_hash, tokens) are intentionally omitted.",
                "evaluation": {
                    "score": 7,
                    "aspects": {
                        "clarity": "Clear description of included vs excluded fields.",
                        "completeness": "Decent, though missing mention of null-handling and timezone formatting.",
                        "accuracy": "Accurately states what is serialized.",
                        "conciseness": "Tight and to the point.",
                        "context_awareness": "Explains the mapping from model to API output.",
                        "technical_precision": "Could specify exact timestamp format (ISO8601?).",
                        "limitations_disclosure": "Good mention of intentionally omitted fields.",
                        "tone_and_style": "Neutral and informative.",
                    },
                    "overall_feedback": "Useful serialization documentation with correct security hints. Could be improved by specifying formatting rules and optional fields.",
                    "improvement_suggestions": [
                        "Specify timezone (UTC?) and timestamp format.",
                        "Document behavior for missing/None attributes.",
                        "Mention whether nested models are supported.",
                    ],
                },
            },
        ]

    def build_evaluation_prompt(
        self, documentation: str, codecontent: str, icl_examples: List[Dict]
    ) -> str:
        """
        Build evaluation prompt in English
        """
        prompt = """You are a professional documentation quality evaluation expert. Please evaluate the quality of the documentation based on the following criteria:

        Evaluation Criteria:
        1. Clarity: Whether the documentation is clear and easy to understand, with unambiguous expression
        2. Completeness: Whether it contains necessary information (such as function description, parameter explanations, return values, usage examples, etc.)
        3. Accuracy: Whether the description is accurate, and whether there are errors or misleading information
        4. Conciseness: Whether it is concise and to the point, avoiding unnecessary redundancy

        Scoring Scale: 1-10 points (10 being the highest)

        Please refer to the following examples:

        """

        # Add ICL examples
        for i, example in enumerate(icl_examples):
            prompt += f"Example {i+1}:\n"
            prompt += f"Documentation Content: {example['documentation']}\n"
            prompt += f"Evaluation Result: {json.dumps(example['evaluation'], ensure_ascii=False, indent=2)}\n\n"

        prompt += f"""
        Input:
        Documentation Content:
        {documentation}
        Code File Content:
        {codecontent}
        ---
        Return a JSON object **with no additional text**, following this exact structure:
        {{
        "score": <1-10 integer>,
        "aspects": {{
            "clarity": "<assessment with reasoning>",
            "completeness": "<assessment with reasoning>",
            "accuracy": "<assessment with reasoning; must reference code behavior>",
            "conciseness": "<assessment with reasoning>"
        }},
        "overall_feedback": "<overall summary of evaluation>",
        "improvement_suggestions": [
            "<suggestion 1>",
            "<suggestion 2>",
            "..."
        ]
        }}
        Requirements:
        - All judgments **must be grounded directly in the provided code**.
        - If the documentation describes functionality not present in the code, flag it as inaccurate.
        - If the documentation misses important behaviors shown in the code, flag it as incomplete.
        - Be objective, specific, and technical.
        - Do not include commentary outside the JSON.
        Now evaluate.
        """
        return prompt

    def get_documentation_list(
        self,
    ) -> List[
        str
    ]:  # this func is to load all the documentation files from self.wiki_path # return list of documentation path # e.g. ["./.wikis/xxxxxx/xxxx.md", "./.wikis/xxxxxx/xxxx2.md"]
        documentation_pairs = []
        if not os.path.exists(self.wiki_path):
            return documentation_pairs

        code_exts = [".c", ".h", ".cpp", ".hpp", ".cc"]
        # e.g. self.wiki_path = "./.wikis/facebook_zstd/test_update"
        # # under this path, we will find dirs and files
        for root, dirs, files in os.walk(self.wiki_path):
            for file in files:
                if not file.endswith(".md"):
                    continue

                doc_path = os.path.join(root, file)
                # print(f"code_path: {self.code_path}")
                # print(f"documentation: {doc_path}")

                # 处理文档名 → 去掉 _doc.md 或 .md
                doc_base = file.replace("_doc.md", "").replace(".md", "")
                # print(f"去掉 _doc.md 或 .md: {doc_base}")
                matched_code_path = None

                # 搜索代码文件
                for croot, _, cfiles in os.walk(self.code_path):
                    for cfile in cfiles:
                        # 去掉代码文件后缀
                        for ext in code_exts:
                            if cfile.endswith(ext):
                                # print(f"doc:{doc_base}////code:{cfile}")
                                if cfile == doc_base:
                                    matched_code_path = os.path.join(croot, cfile)
                                    break
                        if matched_code_path:
                            break
                    if matched_code_path:
                        break

                if not matched_code_path:
                    print(
                        f"[WARN] No corresponding code file found for documentation: {doc_path}"
                    )
                    continue
                print(f"code file found for documentation: {doc_path}")
                documentation_pairs.append((doc_path, matched_code_path))
        return documentation_pairs

    def evaluate_single_documentation(
        self, doc_path: str, code_path: str, icl_examples: List[Dict]
    ) -> Dict:
        """
        Evaluate a single documentation file
        """
        try:
            # Read documentation content
            with open(doc_path, "r", encoding="utf-8") as f:
                documentation = f.read()

            with open(code_path, "r", encoding="utf-8") as f:
                codecontent = f.read()
            # Build evaluation prompt
            prompt = self.build_evaluation_prompt(
                documentation, codecontent, icl_examples
            )

            # Call LLM to evaluate
            llm = CONFIG.get_llm()
            response = llm.invoke(prompt)

            # Parse response
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Try to extract JSON from the response
            try:
                # Find JSON in the response (in case there's extra text)
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    evaluation = json.loads(json_str)
                else:
                    evaluation = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON for {doc_path}: {e}")
                evaluation = {
                    "score": 0,
                    "aspects": {
                        "clarity": "Failed to parse LLM response",
                        "completeness": "Failed to parse LLM response",
                        "accuracy": "Failed to parse LLM response",
                        "conciseness": "Failed to parse LLM response",
                    },
                    "overall_feedback": "Failed to parse LLM response",
                    "improvement_suggestions": [],
                    "error": str(e),
                }

            return {
                "doc_path": doc_path,
                "evaluation": evaluation,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"Error evaluating {doc_path}: {e}")
            return {
                "doc_path": doc_path,
                "evaluation": {"score": 0, "error": str(e)},
                "timestamp": datetime.now().isoformat(),
            }

    def evaluate(self, max_workers: int = 5):
        # 1. Load the wiki content
        documentation_list = self.get_documentation_list()
        if not documentation_list:
            print("No documentation files found.")
            return

        print(f"Found {len(documentation_list)} documentation files to evaluate.")

        # 2. Get ICL examples
        icl_examples = self.get_icl_examples()

        # 3. ThreadPool to call LLM to evaluate
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all evaluation tasks
            future_to_doc = {
                executor.submit(
                    self.evaluate_single_documentation,
                    doc_path,
                    code_path,
                    icl_examples,
                ): (doc_path, code_path)
                for doc_path, code_path in documentation_list
            }

            # Process completed tasks
            for future in as_completed(future_to_doc):
                doc_path = future_to_doc[future]
                try:
                    result = future.result()
                    results.append(result)
                    score = result["evaluation"].get("score", 0)
                    print(f"Evaluated: {doc_path} - Score: {score}")
                except Exception as e:
                    print(f"Exception evaluating {doc_path}: {e}")

        # 4. Save results to output file
        output_data = {
            "evaluation_time": datetime.now().isoformat(),
            "total_docs": len(documentation_list),
            "results": results,
            "average_score": (
                sum(r["evaluation"].get("score", 0) for r in results) / len(results)
                if results
                else 0
            ),
        }

        os.makedirs(
            (
                os.path.dirname(self.output_file)
                if os.path.dirname(self.output_file)
                else "."
            ),
            exist_ok=True,
        )
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\nEvaluation complete!")
        print(f"Total documents: {output_data['total_docs']}")
        print(f"Average score: {output_data['average_score']:.2f}")
        print(f"Results saved to: {self.output_file}")

        return output_data


if __name__ == "__main__":
    # use "uv run python -m tests.LaaJ.judge"
    judge_agent = LaaJAgent(owner="facebook", repo="zstd", dir="")

    judge_agent.evaluate()

    # prompt = judge_agent.build_evaluation_prompt(documentation="test documentation", icl_examples=judge_agent.get_icl_examples())
    # print(f"PROMPT: {prompt}")

    # doc_list = judge_agent.get_documentation_list()
    # print(f"Documentation List: {doc_list}")
