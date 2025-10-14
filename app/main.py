from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import upload, data, health

app = FastAPI(title="****** ARGUS Backend API ******")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(data.router, prefix="/data", tags=["Data"])
app.include_router(health.router, prefix="/health", tags=["Health"])

# Optional: root endpoint
@app.get("/test")
async def root():
    return {"message": "ARGUS Backend API is running!"}

