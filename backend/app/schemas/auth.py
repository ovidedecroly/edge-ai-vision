from typing import Optional, List
from pydantic import BaseModel, EmailStr, ConfigDict

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    refresh_token: Optional[str] = None
    expires_in: int = 7200

class TokenPayload(BaseModel):
    sub: Optional[str] = None
    username: Optional[str] = None
    role: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = ""
    is_active: bool = True
    is_superuser: bool = False

class UserCreate(UserBase):
    password: str
    roles: List[str] = ["Viewer"]

class RoleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    description: str

class UserResponse(UserBase):
    model_config = ConfigDict(from_attributes=True)
    id: str
    roles: List[RoleResponse] = []
