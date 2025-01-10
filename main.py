from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
from common.routes import common_router

load_dotenv()


app = FastAPI(title="Reimbursement Assistant", description="", version="0.0.1")

# Include routers
app.include_router(common_router)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
