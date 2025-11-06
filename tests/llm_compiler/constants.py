# End-of-plan sentinel used to stop LLM generation
END_OF_PLAN = "<END_OF_PLAN>"

# Joiner decision types
JOINER_FINISH = "Finish"
JOINER_REPLAN = "Replan"

# Regex patterns
THOUGHT_PATTERN = r"Thought: ([^\n]*)"
ACTION_PATTERN = r"\n*(\d+)\. (\w+)\((.*)\)(\s*#\w+\n)?"
ID_PATTERN = r"\$\{?(\d+)\}?"  # matches $1 or ${1}

# Scheduling interval (seconds)
SCHEDULING_INTERVAL = 1
