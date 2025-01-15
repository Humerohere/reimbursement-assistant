from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
import os

from common.routes import common_router

load_dotenv()

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Reimbursement Assistant", description="", version="0.0.1")

# Include routers
app.include_router(common_router)


# ESSENTIAL KEYS

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", None)
LLAMA_UPLOAD_URL = os.getenv("LLAMA_UPLOAD_URL", None)
LLAMA_JOB_STATUS_URL = os.getenv("LLAMA_JOB_STATUS_URL", None)
LLAMA_RESULT_URL = os.getenv("LLAMA_RESULT_URL", None)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
