from backend.core.gemini_client import get_gemini_response

class MathAgent:
    @staticmethod
    def handle(query: str) -> str:
        prompt = f"You are a helpful math assistant. Solve or explain: {query}"
        return get_gemini_response(prompt)
    