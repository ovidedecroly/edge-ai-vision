from typing import List
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database.session import get_db
from app.database import models
from app.schemas.gate import GateResponse, DeviceResponse, GateCommandRequest, GateCommandResponse
from app.auth.dependencies import get_current_user, require_roles
from app.websocket.manager import ws_manager

router = APIRouter()

@router.get("/devices", response_model=List[DeviceResponse])
async def list_devices(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    stmt = select(models.Device)
    result = await db.execute(stmt)
    return result.scalars().all()

@router.get("/gates", response_model=List[GateResponse])
async def list_gates(
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    stmt = select(models.Gate)
    result = await db.execute(stmt)
    return result.scalars().all()

@router.post("/gates/{gate_id}/request-open", response_model=GateCommandResponse)
async def request_open_gate(
    gate_id: str,
    cmd_in: GateCommandRequest,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(require_roles(["Administrator", "Operator"]))
):
    stmt = select(models.Gate).where(models.Gate.id == gate_id)
    gate = (await db.execute(stmt)).scalar_one_or_none()
    if not gate:
        raise HTTPException(status_code=404, detail="Gate not found")

    # Safety check: if safety beam is broken or emergency stop is active, gate cannot open
    if gate.current_state == "emergency_stop":
        raise HTTPException(status_code=400, detail="Gate is in Emergency Stop state. Reset required.")

    gate.current_state = "opening"
    gate.last_command = cmd_in.command
    gate.relay_state = True

    cmd_record = models.GateCommand(
        gate_id=gate.id,
        command=cmd_in.command,
        source="manual_operator",
        issued_by_user_id=current_user.id,
        status="executed",
        reason=cmd_in.reason or "Manual command from web UI"
    )
    db.add(cmd_record)

    audit = models.AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action="manual_gate",
        target_type="gate",
        target_id=gate.id,
        details={"command": cmd_in.command, "reason": cmd_in.reason}
    )
    db.add(audit)
    await db.commit()
    await db.refresh(cmd_record)

    # Broadcast to WebSocket
    await ws_manager.broadcast("gate-status", {
        "gate_id": gate.id,
        "state": gate.current_state,
        "command": cmd_in.command,
        "updated_at": datetime.now(timezone.utc).isoformat()
    })

    return cmd_record
