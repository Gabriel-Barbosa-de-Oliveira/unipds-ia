import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()


class GroqLLM(LLM):
    """LLM via LiteLLM que remove a flag `cache_breakpoint` das mensagens.

    O CrewAI 1.15.x marca mensagens com `cache_breakpoint`, mas o caminho
    LiteLLM não a remove, e a Groq rejeita a propriedade (HTTP 400).
    """

    def _prepare_completion_params(self, *args, **kwargs):
        params = super()._prepare_completion_params(*args, **kwargs)
        for message in params.get("messages", []):
            if isinstance(message, dict):
                message.pop("cache_breakpoint", None)
        return params


# Centraliza a inteligência do projeto
nexus_llm = GroqLLM(
    model=os.getenv("NEXUS_MODEL", "groq/openai/gpt-oss-120b"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)
