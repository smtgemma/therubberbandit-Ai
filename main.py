from fastapi import FastAPI
from App.services.extraction.extract_route import router as extraction_router
from fastapi.middleware.cors import CORSMiddleware
from App.services.rating.rating_route import router as rating_router
from App.services.chatbot.chatbot_routes import router as chatbot_router
from App.services.quiz.quiz_routes import router as quiz_router
from App.services.extraction.document_extract_route import router as document_extract_router


app = FastAPI(
              title="Document-AI FastAPI", 
              version="1.0.0"
              )

app.include_router(extraction_router)
app.include_router(rating_router)
app.include_router(chatbot_router)
app.include_router(quiz_router)
app.include_router(document_extract_router)

# In your main app/main.py file, add:


