import argparse
import queue
import threading
import uuid

from datetime import datetime
from pathlib import Path

import fastapi
import openai
import uvicorn

from loguru import logger

from utils import CompletedJob
from utils import Job
from utils import JobRequest
from utils import Statistics
from utils import Status
from utils import Timing

app = fastapi.FastAPI()
lock = threading.Lock()

pending_jobs: dict[str, Job] = {}
completed_jobs: dict[str, CompletedJob] = {}
job_queue = queue.Queue()
active_processes = []
settings = {}


def get_parser() -> argparse.ArgumentParser:
    _parser = argparse.ArgumentParser()
    _parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host IP of the backend service",
    )
    _parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port of the backend service",
    )
    _parser.add_argument(
        "--max-processes",
        type=int,
        default=2,
        help="Maximum number of processes to run in parallel",
    )
    _parser.add_argument(
        "--llm-host",
        type=str,
        default="localhost",
        help="Host of remote LLM instance",
    )
    _parser.add_argument(
        "--llm-port",
        type=int,
        default=8080,
        help="Port of remote LLM instance",
    )
    _parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="API key for remote LLM calls",
    )
    _parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Max tokens for remote LLM call",
    )
    _parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for remote LLM call",
    )
    _parser.add_argument(
        "--system-prompt",
        type=Path,
        default=None,
        help="Path to default system prompt file",
    )

    return _parser


def clean_answer(answer: str) -> str:
    start = answer.find("{")
    end = answer.rfind("}")
    return answer[start:end + 1] if start != -1 and end != -1 else answer


def call_llm(prompt: str) -> tuple[str, Statistics]:
    host = settings.get("llm_host", "localhost")
    port = settings.get("llm_port", "8080")
    base_url = f"http://{host}:{port}"
    api_key = settings.get("llm_api_key", "examplekey01")
    max_tokens = settings.get("max_tokens", 1000)
    timeout = settings.get("timeout", 300)

    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
    )

    response = client.completions.create(
        prompt=prompt,
        model=client.models.list().data[0].id,
        max_tokens=max_tokens,
        timeout=timeout,
        stream=False,
    )

    return response.content, Statistics(**response.timings)


def generate_answer(job: Job):
    prompt = f"<s>[INST] <<SYS>>\n{job.prompt}\n<</SYS>>\n\n {job.email_text} [/INST] </s>"
    logger.debug(f"Call model for {job.job_id} ...")

    with lock:
        start_execution = datetime.now()
        (answer, statistics) = call_llm(prompt)
        finish_job = datetime.now()

    logger.debug("Output received from model.")
    delta = finish_job - start_execution
    cleaned_answer = clean_answer(answer)

    job = CompletedJob(
        job_id=job.job_id,
        email_text=job.email_text,
        prompt=job.prompt,
        status=Status.COMPLETE,
        timing=Timing(
            delta=delta,
            finish_job=finish_job,
            start_execution=start_execution,
            start_job=job.timing.start_job,
        ),
        result=cleaned_answer,
        statistics=statistics,
    )

    completed_jobs[job.job_id] = job
    pending_jobs.pop(job.job_id)
    logger.debug(f"Completed job with ID: {job.job_id}")


def start_process(job_q: queue.Queue):
    while True:
        if job_q.qsize() == 0:
            break

        check_processes()
        _, job = job_q.get()
        pending_jobs[job.job_id] = job
        generate_answer(job)


def check_processes():
    global active_processes
    active_processes = [p for p in active_processes if p.is_alive()]
    if len(active_processes) < settings.get("max_processes", 2):
        active_processes.append(threading.Thread(target=start_process, args=(job_queue,)))

    [process.start() for process in active_processes if not process.is_alive()]


def get_default_prompt() -> str:
    default_prompt = """
    You are an assistant to fill in a formless shipment request into a form which can be processed by a computer.
    You have to fill the following template:
    {"orderID":"","orderDate":"","expectedShipDate":"","fromAddress":{"companyName":"","address":{"street":"","city":"","state":"","postalCode":"","country":""}},"toAddress":{"customerId":"","firstName":"","lastName":"","address":{"street":"","city":"","state":"","postalCode":"","country":""}},"shippingDetails":{"carrier":"","shippingMode":"","shippingType":""},"orderDetails":{"items":{"put the itemId <string> here":{"itemId":"","category":"","quantityUnits":"","quantity":"","weightUnits":"","weight":"","dimensionUnits":"","dimensions":"","insuranceAmount":""}},"totalInsuranceAmount":""}}

    Use the following format for the dates: yyyy-MM-ddTHH:mm:ss. If there is no mention of a date you should not set one but leave the space empty.
    Extract the data to fill the template from the following information and only return the created JSON string.
    The information to extract is in German.
    You may encounter the German number format. They use 1.000,00 instead of 1,000.00
    The values in the weight and dimensions are floats.
    Use the ISO 3166 Alpha-2 code Country Codes for the country instead of the full name of the country.
    The returned JSON string needs to be valid JSON.
    Explanation of the different elements: In the task there should be the information whether it is a order or offer or similar. The destinationAddress is the address where the goods should be delivered to. The startAddress is the location where the goods are picked up. The cargoType is a field where something like "Palette" for pallets or "Sack" for sacks, "Kiste" for chests or "sonstiges" if it is not one of those 3.
    Do not explain any of the different elements.
    """
    global settings

    if path := settings.get("system_prompt"):
        with open(path) as f:
            system_prompt = f.read()
        return system_prompt

    return default_prompt


@app.post("/create/")
def create_job(job_request: JobRequest):
    if not job_request.email_text:
        raise fastapi.HTTPException(status_code=400, detail="Please specify a valid e-mail text.")

    start_job = datetime.now()
    job_id = uuid.uuid4()
    prompt = job_request.prompt if job_request.prompt else get_default_prompt()

    job = Job(
        job_id=str(job_id),
        email_text=job_request.email_text,
        prompt=prompt,
        status=Status.PENDING,
        timing=Timing(start_job=start_job),
    )

    job_queue.put((start_job.timestamp(), job))
    check_processes()
    
    return {"job_id": job_id, "status": Status.SCHEDULED}


@app.get("/jobs/")
def get_all_jobs() -> list[Job]:
    return sorted(list(pending_jobs.values()) + list(completed_jobs.values()), key=lambda job: job.timing.start_job)


@app.get("/job/{job_id}")
def get_job(job_id: str) -> CompletedJob:
    if job := completed_jobs.pop(job_id, None):
        return job
    else:
        raise fastapi.HTTPException(status_code=404, detail=f"No completed job with ID {job_id} found.")


if __name__ == "backend":
    parser = get_parser()
    args = parser.parse_args()
    settings = vars(args)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    settings = vars(args)

    uvicorn.run("backend:app", host=settings.get("host", "0.0.0.0"), port=int(settings.get("port", 8000)))
