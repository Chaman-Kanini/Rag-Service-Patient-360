from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import API_TITLE, API_DESCRIPTION, API_VERSION
from app.routes import health, batch, qa, chatbot, patient

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(batch.router)
app.include_router(qa.router)
app.include_router(chatbot.router)
app.include_router(patient.router)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to RAG Pipeline API",
        "version": API_VERSION,
        "docs": "/docs"
    }
