from fastapi import APIRouter
from app.api.v1.endpoints import verify

api_router = APIRouter()
api_router.include_router(verify.router, prefix="/face", tags=["face"])
