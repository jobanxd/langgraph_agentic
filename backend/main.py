from fastapi import FastAPI
from backend.routers import chat_router
import logging

# Configure global logging
logging.basicConfig(
    level=logging.INFO,  # You can use DEBUG for even more detail
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Agentic AI with Gemini + LangGraph")

app.include_router(chat_router.router)

@app.get("/")
def root():
    return {"message": "Welcome to the Agentic AI with Gemini + LangGraph API!"}

