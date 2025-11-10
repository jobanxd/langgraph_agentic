from backend.core.gemini_client import get_gemini_response

class ScienceAgent:
    @staticmethod
    def handle(query: str) -> str:
        prompt = f"You are a helpful science assistant. Explain clearly: {query}"
        return get_gemini_response(prompt)
    