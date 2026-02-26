from pydantic import BaseModel

class config(BaseModel):
    DEFAULT_MODEL: str = "Qwen/Qwen2.5-3B-Instruct"
    DEFAULT_NEUTRAL_SYSTEM: str = (
        "You are a writing engine. Do not add preambles, disclaimers, or commentary. "
        "Output only the requested text."
    )
    temperature: float = 0.7
    top_p: float = 0.9
    beta: float = 1.0
    top_k: int = 200
    clamp: float = 10.0
