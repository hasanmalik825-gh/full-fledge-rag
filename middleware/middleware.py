import time
from constants import IP_WHITELIST
from fastapi import HTTPException, Request, status

async def add_middleware(request: Request, call_next):
    start_time = time.time()

    client_host = request.client.host
    url_path = request.url.path
    print(f"Received a request: {request.method} | {url_path}")
    print(f"Client host: {client_host}")

    if client_host not in IP_WHITELIST:
        print(f"Unauthorized access for {client_host}!")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized access for your IP")
    if url_path not in ["/", "/llm-inference", "/docs", "/openapi.json"]:
        print(f"Invalid path: {url_path}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path")

    response = await call_next(request)

    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    return response