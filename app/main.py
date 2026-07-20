import logging
import os
import warnings

# 抑制 LangGraph 弃用警告
warnings.filterwarnings("ignore", category=PendingDeprecationWarning, module="langgraph")


import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.main import api_router


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="评测应用",
    description="评测应用"
)

# FastAPI 可观测埋点（本地环境容错）
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app=app, exclude_spans=["receive", "send"])
except Exception as exc:
    logger.warning(f"OpenTelemetry instrumentation skipped in local env: {exc}")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#引入自定义URL路由
app.include_router(api_router)

#附带一个静态Html Chat页面，用于调试SSE响应
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == '__main__':
    logger.info("Starting API server")
    uvicorn.run(
        app="app.main:app",
        host="0.0.0.0", #127.0.0.1
        port=8080,
        log_level="info",
        workers=4 #根据单机资源自行设置
    )