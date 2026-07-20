from fastapi import APIRouter

from app.api.routes import health, simulator, evaluate, pipeline


api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(simulator.router)
api_router.include_router(evaluate.router)
api_router.include_router(pipeline.router)

