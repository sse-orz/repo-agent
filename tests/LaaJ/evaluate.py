# this file is to
# 1. evaluate the time cost of the generated documentation
# 2. evaluate the quality of the generated documentation
# 3. evaluate the time cost of the generated rag answer
# 4. evaluate the quality of the generated rag answer

import os
import json
from time import time
from datetime import datetime
from dataclasses import dataclass

from agents.rag_agent import RAGAgent
from app.models.rag import RAGAnswerData
from agents.sub_graph.parent import ParentGraphBuilder


@dataclass
class EvaluateInput:
    owner: str
    repo: str
    platform: str
    mode: str
    max_workers: int
    directory: str
    branch_mode: str
    ratios: dict[str, float]
    question: str


class EvaluateAgent:
    def __init__(self, inputs: EvaluateInput):
        self.sub_graph_builder = ParentGraphBuilder(branch_mode=inputs.branch_mode)
        self.rag_agent = RAGAgent()
        self.inputs = inputs

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
                "wiki_root_path": self.inputs.directory,
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

    def evaluate(self):
        sub_graph_time = self.run_sub_graph()
        rag_agent_data, rag_agent_time = self.run_rag_agent()
        self.save_results(sub_graph_time, rag_agent_data, rag_agent_time)

    def save_results(self, sub_graph_time, rag_agent_data, rag_agent_time):
        rag_agent_data_dict = rag_agent_data.model_dump()

        results = {
            "repo_name": f"{self.inputs.platform}:{self.inputs.owner}/{self.inputs.repo}",
            "repo_dir": f"{self.inputs.owner}_{self.inputs.repo}",
            "mode": self.inputs.mode,
            "max_workers": self.inputs.max_workers,
            "directory": self.inputs.directory,
            "branch_mode": self.inputs.branch_mode,
            "ratios": self.inputs.ratios,
            "question": self.inputs.question,
            "sub_graph_time": sub_graph_time,
            "rag_agent_time": rag_agent_time,
            "rag_agent_data": rag_agent_data_dict,
        }
        output_dir = ".evaluate"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "evaluate_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    evaluate_inputs = EvaluateInput(
        owner="octocat",
        repo="Hello-World",
        platform="github",
        mode="fast",
        max_workers=10,
        directory="./.wikis/evaluate",
        branch_mode="all",
        ratios={
            "fast": 0.05,
            "smart": 0.75,
        },
        question="What is this repo about?",
    )
    evaluate_agent = EvaluateAgent(inputs=evaluate_inputs)
    evaluate_agent.evaluate()
