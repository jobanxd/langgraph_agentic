import google.generativeai as genai
from backend.core.config import Settings

genai.configure(api_key=Settings.GOOGLE_API_KEY)

def get_gemini_response(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text
