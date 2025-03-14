import enum

from datetime import datetime
from datetime import timedelta

import pydantic


class Status(enum.StrEnum):
    COMPLETE = "complete"
    PENDING = "pending"
    SCHEDULED = "scheduled"


class Statistics(pydantic.BaseModel):
    prompt_n: int
    prompt_ms: float
    prompt_per_token_ms: float
    prompt_per_second: float
    predicted_n: int
    predicted_ms: float
    predicted_per_token_ms: float
    predicted_per_second: float


class Timing(pydantic.BaseModel):
    delta: timedelta = None
    finish_job: datetime = None
    start_execution: datetime = None
    start_job: datetime


class Job(pydantic.BaseModel):
    job_id: str
    email_text: str
    prompt: str
    status: Status
    timing: Timing


class JobRequest(pydantic.BaseModel):
    email_text: str
    prompt: str = None


class CompletedJob(Job):
    result: dict | str
    statistics: Statistics
