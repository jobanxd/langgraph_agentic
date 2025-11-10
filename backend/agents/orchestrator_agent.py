from backend.core.gemini_client import get_gemini_response

class OrchestratorAgent:
    @staticmethod
    def decide(query: str) -> str:
        """
        Ask the LLM to decide which agent shoudl handle this query.
        It must respond with exactly one word: "MathAgent" or "ScienceAgent" or "General".
        """

        decision_prompt = f"""
            You are an orchestrator agent. Your job is to decide which sub-agent should handle the  following query.
            
            Query: "{query}

            Available sub-agents:
            - MathAgent: For mathematical questions, calculations, and problem-solving.
            - ScienceAgent: For scientific explanations, concepts, and inquiries.
            - General: For all other types of questions.

            Respond ONLY with one word: "MathAgent", "ScienceAgent", or "General".
            """
        result = get_gemini_response(decision_prompt).strip().lower()
        if result not in ["mathagent", "scienceagent", "general"]:
            result = "General"
        
        return result
    
    @staticmethod
    def handle_general(query: str) -> str:
        prompt = f"You are a helpful general assistant. Answer the following question: {query}"
        return get_gemini_response(prompt)
    
    