import argparse
import ast
import json

import gradio as gr
import requests

from ibm_theme import IBMTheme
from utils import JobRequest

custom_css = """
h1, h2 {
    color: #0F62FE;
}
"""

headers = {
    "Content-Type": "application/json",
}


def send_request(email_text: str, url: str, prompt: str = None) -> dict:
    data = JobRequest(
        email_text=email_text,
        prompt=prompt,
    )

    return requests.post(
        url=f"{url}/create",
        json=data.model_dump(),
        headers=headers,
    ).json()


def get_all_jobs(url: str) -> str:
    return json.dumps(requests.get(f"{url}/jobs", headers=headers).json(), indent=4)


def get_job(url: str, job_id: str) -> str:
    response: dict = requests.get(f"{url}/job/{job_id}", headers=headers).json()

    if "result" in response:
        try:
            result_object = ast.literal_eval(response["result"])
        except ValueError:
            result_object = response["result"]
        response["result"] = result_object

    return json.dumps(response, indent=4)


def get_parser() -> argparse.ArgumentParser:
    _parser = argparse.ArgumentParser()
    _parser.add_argument(
        "--host",
        help="The host, where the frontend is available at",
        default="0.0.0.0",
    )
    _parser.add_argument(
        "--port",
        type=int,
        help="The port, where the frontend is available at",
        default=7860
    )

    return _parser


def main(host: str = "0.0.0.0", port: int = 7860):
    title = "Logistic Data Extraction Demo"

    with gr.Blocks(title=title, theme=IBMTheme(), css=custom_css) as demo:
        gr.Markdown(f"# {title}")
        with gr.Accordion(label="Advanced Settings", open=False):
            url_box = gr.Textbox(label="Please enter the backend url", value="http://0.0.0.0:8000")

        with gr.Tab("Create new request"):
            email_box = gr.Textbox(label="E-Mail text to extract information from:")
            prompt_box = gr.Textbox(label="Optional prompt:")
            job_id = gr.Textbox(label="Job ID and Status:")
            extract_btn = gr.Button("Send request", variant="primary")
            extract_btn.click(
                fn=send_request,
                inputs=[email_box, url_box, prompt_box],
                outputs=[job_id],
                api_name="send_request",
            )

        with gr.Tab("Job Queue"):
            jobs_box = gr.Textbox(label="Job Queue:")
            query_btn = gr.Button("Query Job Queue", variant="primary")
            query_btn.click(
                fn=get_all_jobs,
                inputs=[url_box],
                outputs=[jobs_box],
                api_name="get_job_queue",
            )

        with gr.Tab("Get job result"):
            job_id_box = gr.Textbox(label="Job ID:")
            job_result_box = gr.Textbox(label="Job Result:")
            result_btn = gr.Button("Get Job Result", variant="primary")
            result_btn.click(
                fn=get_job,
                inputs=[url_box, job_id_box],
                outputs=[job_result_box],
                api_name="get_job_result",
            )

    demo.launch(server_name=host, server_port=port, debug=True)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.host, args.port)
