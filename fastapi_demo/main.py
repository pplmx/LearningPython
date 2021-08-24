import time

import uvicorn
from fastapi import FastAPI, Request

from fastapi_demo.router import demo

app = FastAPI()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time_ns()
    response = await call_next(request)
    process_time = time.time_ns() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


app.include_router(demo.router)

if __name__ == '__main__':
    uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=True)
