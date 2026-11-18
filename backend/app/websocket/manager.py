import logging
from typing import Dict, List, Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_channels: Dict[str, List[WebSocket]] = {
            "detections": [],
            "events": [],
            "system-metrics": [],
            "gate-status": []
        }

    async def connect(self, websocket: WebSocket, channel: str):
        await websocket.accept()
        if channel not in self.active_channels:
            self.active_channels[channel] = []
        self.active_channels[channel].append(websocket)
        logger.info("WebSocket connected to channel '%s'. Total: %d", channel, len(self.active_channels[channel]))

    def disconnect(self, websocket: WebSocket, channel: str):
        if channel in self.active_channels and websocket in self.active_channels[channel]:
            self.active_channels[channel].remove(websocket)
            logger.info("WebSocket disconnected from channel '%s'. Remaining: %d", channel, len(self.active_channels[channel]))

    async def broadcast(self, channel: str, message: Any):
        if channel not in self.active_channels:
            return
        dead_connections = []
        for connection in self.active_channels[channel]:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)

        for dead in dead_connections:
            self.disconnect(dead, channel)

ws_manager = ConnectionManager()
