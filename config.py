import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from typing import Dict, Optional

load_dotenv()


class GlobalConfig:
    TOKEN: Dict[str, str]
    LLM_PLATFORM: str
    LLM_MODEL: str
    GOOGLE_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GLM_API_KEY: Optional[str] = None
    MINIMAX_API_KEY: Optional[str] = None
    LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    LANGSMITH_API_KEY: Optional[str] = os.getenv("LANGSMITH_API_KEY", None)
    LANGSMITH_PROJECT: Optional[str] = os.getenv("LANGSMITH_PROJECT", None)

    def __init__(self):
        self.TOKEN = {
            "github": os.getenv("GITHUB_TOKEN"),
            "gitee": os.getenv("GITEE_TOKEN"),
        }
        self.LLM_PLATFORM = os.getenv("LLM_PLATFORM", "ollama").lower()
        match self.LLM_PLATFORM:
            case "ollama":
                self.LLM_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
            case "google":
                self.LLM_MODEL = os.getenv("GOOGLE_MODEL", "gemini-pro")
                self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
            case "deepseek":
                self.LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
                self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
            case "openai":
                self.LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
                self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
            case "glm":
                self.LLM_MODEL = os.getenv("GLM_MODEL", "GLM-4.5")
                self.GLM_API_KEY = os.getenv("GLM_API_KEY", "")
            case "minimax":
                self.LLM_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2")
                self.MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
            case _:
                raise ValueError(f"Unsupported LLM_PLATFORM: {self.LLM_PLATFORM}")

    def display(self) -> None:
        print("Current Configuration:")
        print(f"LLM_MODEL: {self.LLM_MODEL}")
        print(f"GITHUB_TOKEN: {'Set' if self.TOKEN['github'] else 'Not Set'}")
        print(f"GITEE_TOKEN: {'Set' if self.TOKEN['gitee'] else 'Not Set'}")
        print(f"LANGSMITH_TRACING: {self.LANGSMITH_TRACING}")
        print(f"LANGSMITH_API_KEY: {'Set' if self.LANGSMITH_API_KEY else 'Not Set'}")
        print(
            f"LANGSMITH_PROJECT: {self.LANGSMITH_PROJECT if self.LANGSMITH_PROJECT else 'Not Set'}"
        )

    def get_token(self, platform: str) -> Optional[str]:
        return self.TOKEN.get(platform.lower(), None)

    def get_llm(self):
        match self.LLM_PLATFORM:
            case "ollama":
                return ChatOllama(model=self.LLM_MODEL)
            case "google":
                return ChatGoogleGenerativeAI(
                    model=self.LLM_MODEL, api_key=self.GOOGLE_API_KEY
                )
            case "deepseek":
                return ChatDeepSeek(model=self.LLM_MODEL, api_key=self.DEEPSEEK_API_KEY)
            case "openai":
                return ChatOpenAI(
                    model=self.LLM_MODEL, openai_api_key=self.OPENAI_API_KEY
                )
            case "glm":
                return ChatOpenAI(
                    model=self.LLM_MODEL,
                    openai_api_key=self.GLM_API_KEY,
                    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
                )
            case "minimax":
                return ChatOpenAI(
                    model=self.LLM_MODEL,
                    openai_api_key=self.MINIMAX_API_KEY,
                    openai_api_base="https://api.minimaxi.com/v1",
                )
            case _:
                raise ValueError(f"Unsupported LLM_PLATFORM: {self.LLM_PLATFORM}")


CONFIG = GlobalConfig()
