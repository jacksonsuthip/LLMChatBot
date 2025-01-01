from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routers import chat_router
from app.database import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("Application startup")
        # await init_db()
    except Exception as e:
        print(f"Error during startup: {e}")
        raise e

    yield
    
    try:
        print("Application shutdown")
    except Exception as e:
        print(f"Error during shutdown: {e}")
        raise e
    
app = FastAPI(lifespan=lifespan)

app.include_router(chat_router.router)

# Test Route for Basic Testing
@app.get("/test")
async def Test_endpoint():
    return {"message": "This is a test endpoint"}