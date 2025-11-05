from utils.repo import clone_repo, pull_repo
from agents.wiki_agent import WikiAgent
from agents.rag_agent import RAGAgent
from agents.supervisor_agent import SupervisorAgent
from config import CONFIG
import json


def get_mode() -> str:
    """Get user input for the mode.

    Returns:
        str: The selected mode.
    """
    mode = ""
    while (mode is None) or (mode not in ["generate", "ask"]):
        user_input = input(
            'Choose mode: \nPrint "1" if you want to Generate Wiki Files\nPrint "2" if you want to Ask questions about Wiki Repo\n'
        )
        match user_input:
            case "1":
                mode = "generate"
            case "2":
                mode = "ask"
            case _:
                print("Invalid input. Please enter '1' or '2'.")
    return mode


def get_repo_name() -> str:
    """Get user input for the repository name.

    Returns:
        str: The repository name.
    """
    repo_name = ""
    while repo_name == "":
        user_input = input(
            'Enter the repository name (such as "github:facebook/zstd"): '
        )
        repo_name = user_input.strip()
    return repo_name


def preprocess_repo(repo_name: str) -> tuple[str, str, str, str]:
    """Preprocess the repository name and ensure it is cloned and up to date.

    Args:
        repo_name (str): The repository name in the format "platform:owner/repo".

    Returns:
        tuple[str, str, str, str]: A tuple containing the platform, owner, repo, and repo_path.
    """
    # repo_name = repo_name.replace(" ", "")
    platform, full_repo = repo_name.split(":")
    owner, repo = full_repo.split("/")
    repo_path = f"./.repos/{owner}_{repo}"
    clone_result = clone_repo(platform=platform, owner=owner, repo=repo, dest=repo_path)
    if clone_result:
        print(f"Cloned repository {repo_name} to {repo_path}.")
    else:
        print(f"Repository {repo_name} already exists. Pulling latest changes...")
        pull_result = pull_repo(
            platform=platform, owner=owner, repo=repo, dest=repo_path
        )
        if pull_result:
            print(f"Pulled latest changes for repository {repo_name}.")
        else:
            print(f"Failed to pull latest changes for repository {repo_name}.")

    return platform, owner, repo, repo_path


def display_book():
    # use gitbook to display the generated wiki
    wiki_root = ".wikis"
    display_root = "display"
    import os
    import subprocess

    # make sure nvm has been installed
    # Check if NVM directory exists and nvm.sh file is present
    nvm_dir = os.path.expanduser("~/.nvm")
    nvm_sh_path = os.path.join(nvm_dir, "nvm.sh")

    if not os.path.exists(nvm_sh_path):
        print("NVM is not installed. Installing...")
        # install nvm
        subprocess.run(
            [
                "bash",
                "-c",
                "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash",
            ]
        )
        # Install Node.js v10.24.1 after NVM installation
        subprocess.run(
            [
                "bash",
                "-c",
                "source ~/.nvm/nvm.sh && nvm install v10.24.1 && nvm use v10.24.1",
            ]
        )
    else:
        print("NVM is already installed. Using existing installation...")
        # Ensure Node.js v10.24.1 is available
        subprocess.run(["bash", "-c", "source ~/.nvm/nvm.sh && nvm use v10.24.1"])

    # Check if gitbook is installed globally
    gitbook_check = subprocess.run(
        ["bash", "-c", "source ~/.nvm/nvm.sh && nvm use v10.24.1 && which gitbook"],
        capture_output=True,
        text=True,
    )

    if gitbook_check.returncode != 0:
        print("GitBook is not installed. Installing...")
        subprocess.run(
            [
                "bash",
                "-c",
                "source ~/.nvm/nvm.sh && nvm use v10.24.1 && npm install -g gitbook-cli",
            ]
        )
    else:
        print("GitBook is already installed.")

    subprocess.run(["cp", "-r", wiki_root, display_root])

    os.chdir(display_root)
    subprocess.run(
        ["bash", "-c", "source ~/.nvm/nvm.sh && nvm use v10.24.1 && gitbook init"]
    )

    summary_path = "SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write("# Summary\n\n")
        f.write("* [Introduction](README.md)\n")
        for repo in os.listdir("."):
            if os.path.isdir(repo) and repo == wiki_root:
                # Scan for all .md files in the repo directory
                for root, dirs, files in os.walk(repo):
                    md_files = [file for file in files if file.endswith(".md")]
                    if md_files:
                        rel_root = os.path.relpath(root, ".")
                        indent_level = rel_root.count(os.sep) - 1
                        indent = "  " * indent_level
                        if rel_root != repo:
                            folder_name = os.path.basename(root)
                            f.write(f"{indent}* {folder_name}\n")
                        for md in sorted(md_files):
                            md_path = os.path.join(rel_root, md)
                            f.write(f"{indent}  * [{md}]({md_path})\n")
    # gitbook install
    subprocess.run(
        ["bash", "-c", "source ~/.nvm/nvm.sh && nvm use v10.24.1 && gitbook install"]
    )

    # serve the gitbook
    subprocess.run(
        ["bash", "-c", "source ~/.nvm/nvm.sh && nvm use v10.24.1 && gitbook serve"]
    )


def main():
    # 1.1 check mode for user input (generate wiki files or ask for repo)
    print("Welcome to the Repo Agent!")
    CONFIG.display()
    mode = get_mode()

    match mode:
        case "generate":
            # 1.2 user input (get repo name or url)
            repo_name = get_repo_name()
            print(f"Mode: {mode}, Repository: {repo_name}")
            # 2.1 check if repo exists locally, if not, clone it
            platform, owner, repo, repo_path = preprocess_repo(repo_name)
            # 3.1 check if wiki files have been generated, if not, generate them
            # TODO: implement generate wiki logic
            wiki_path = f"./.wikis/{owner}_{repo}"

            # method 1: wiki agent directly generate wiki files
            # wiki_agent initialization
            # wiki_agent = WikiAgent(repo_path=repo_path, wiki_path=wiki_path)
            # wiki_agent.run()

            # method 2: use supervisor agent to manage the process (including repo info collect, code analysis, doc generation, wiki generation, etc.)
            supervisor = SupervisorAgent(
                repo_path=repo_path,
                wiki_path=wiki_path,
                owner=owner,
                repo_name=repo,
            )

            # Run with intelligent file selection
            result = supervisor.generate(
                max_files=30,  # Select up to 30 important files
                batch_size=10,  # Process 10 files per batch
                max_workers=3,  # Use 3 parallel workers
            )

            # print("\n" + "=" * 80)
            # print("Final Result:")
            # print("=" * 80)
            # print(json.dumps(result, indent=2))
            # 3.2 if exist, check if they are up to date, if not, update them
            # TODO: implement update wiki logic
            # TODO: implement saving wiki files to vector database
            print("Wiki generation completed. Exiting.")
            display_book()
            return
        case "ask":
            # TODO: implement question answering logic here
            print(f"Mode: {mode}")
            rag_agent = RAGAgent()
            rag_agent.run()


if __name__ == "__main__":
    main()
    # display_book()
