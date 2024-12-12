import uvicorn
from routes import *
from fastapi import FastAPI
from middleware.middleware import add_middleware

app = FastAPI(
    title="RAG API",
    description="API for RAG",
    version="0.0.1"
)
# Add middleware to handle all http requests
app.middleware('http')(add_middleware)

routes = [index_router, llm_inference_router]
for route in routes:
    app.include_router(route)

if __name__ == "__main__":
    uvicorn.run(app=app, host="localhost", port=8000, reload=True)

    