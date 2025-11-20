import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import CONFIG


class LaaJAgent: # llm as a judge
    def __init__(self, owner: str = "facebook", repo: str = "zstd", dir: str = ""):
        self.wiki_path = f"./.wikis/{owner}_{repo}/{dir}"
        self.output_file = "./evaluation_results.out"

    # in-context learning examples
    def get_icl_examples(self) -> List[Dict]:
        """
        Get In-Context Learning examples
        """
        return [
            {
                "documentation": "This function calculates the sum of two numbers.",
                "evaluation": {
                    "score": 9,
                    "aspects": {
                        "clarity": "Very clear, directly states the function's purpose",
                        "completeness": "Complete description of the main functionality",
                        "accuracy": "Accurate and correct",
                        "conciseness": "Concise and to the point",
                    },
                    "overall_feedback": "Excellent documentation that clearly and accurately describes the function's purpose",
                    "improvement_suggestions": [
                        "Add parameter descriptions",
                        "Include return value explanation",
                    ],
                },
            },
            {
                "documentation": "Function that does stuff.",
                "evaluation": {
                    "score": 3,
                    "aspects": {
                        "clarity": "Very vague, unclear about specific functionality",
                        "completeness": "Missing critical information",
                        "accuracy": "Description is too general",
                        "conciseness": "Overly simplistic",
                    },
                    "overall_feedback": "Poor quality documentation lacking specific information",
                    "improvement_suggestions": [
                        "Clearly specify the function's exact purpose",
                        "Add parameter and return value descriptions",
                        "Provide usage examples",
                    ],
                },
            },
        ]

    def build_evaluation_prompt(
        self, documentation: str, icl_examples: List[Dict]
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
Now please evaluate the following documentation:

Documentation Content:
{documentation}

Please return the evaluation result in JSON format with the following fields:
- score: Overall score (1-10)
- aspects: Evaluation of each dimension with reasoning
  - clarity: Clarity assessment
  - completeness: Completeness assessment  
  - accuracy: Accuracy assessment
  - conciseness: Conciseness assessment
- overall_feedback: Overall feedback
- improvement_suggestions: List of improvement suggestions

Return only the JSON format result without any additional content:
"""
        return prompt

    def get_documentation_list(self) -> List[str]:
        # this func is to load all the documentation files from self.wiki_path
        # return list of documentation path
        # e.g. ["./.wikis/xxxxxx/xxxx.md", "./.wikis/xxxxxx/xxxx2.md"]
        documentation_list = []
        if not os.path.exists(self.wiki_path):
            return documentation_list
        # e.g. self.wiki_path = "./.wikis/facebook_zstd/test_update"
        # under this path, we will find dirs and files
        for root, dirs, files in os.walk(self.wiki_path):
            for file in files:
                if file.endswith(".md"):
                    documentation_list.append(os.path.join(root, file))
        return documentation_list

    def evaluate_single_documentation(
        self, doc_path: str, icl_examples: List[Dict]
    ) -> Dict:
        """
        Evaluate a single documentation file
        """
        try:
            # Read documentation content
            with open(doc_path, "r", encoding="utf-8") as f:
                documentation = f.read()

            # Build evaluation prompt
            prompt = self.build_evaluation_prompt(documentation, icl_examples)

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
                    self.evaluate_single_documentation, doc_path, icl_examples
                ): doc_path
                for doc_path in documentation_list
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
    judge_agent = LaaJAgent(
        owner="facebook", repo="zstd", dir=""
    )

    judge_agent.evaluate()

    # prompt = judge_agent.build_evaluation_prompt(documentation="test documentation", icl_examples=judge_agent.get_icl_examples())
    # print(f"PROMPT: {prompt}")

    # doc_list = judge_agent.get_documentation_list()
    # print(f"Documentation List: {doc_list}")
