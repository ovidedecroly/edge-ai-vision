import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.websocket.manager import ws_manager

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws/detections")
async def ws_detections(websocket: WebSocket):
    await ws_manager.connect(websocket, "detections")
    try:
        while True:
            # Keep-alive ping/pong
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, "detections")

@router.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    await ws_manager.connect(websocket, "events")
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, "events")

@router.websocket("/ws/system-metrics")
async def ws_system_metrics(websocket: WebSocket):
    await ws_manager.connect(websocket, "system-metrics")
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, "system-metrics")

@router.websocket("/ws/gate-status")
async def ws_gate_status(websocket: WebSocket):
    await ws_manager.connect(websocket, "gate-status")
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, "gate-status")
