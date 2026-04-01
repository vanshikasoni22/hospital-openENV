from pydantic import BaseModel
from typing import Literal

class Patient(BaseModel):
    symptoms: str
    age: int
    heart_rate: int
    blood_pressure: int


class Action(BaseModel):
    priority: int  # 1–5
    department: Literal["cardiology", "neurology", "general"]