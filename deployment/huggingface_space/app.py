"""
HuggingFace Gradio Space Entrypoint for XRFM Model API.
100% Free on HuggingFace CPU Basic hardware.
"""

import os
import sys

import gradio as gr
import uvicorn

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from api.main import app as fastapi_app
from inference.engine import GenerationEngine
from model.gpt import GPTModel
from tokenizer.bpe import BytePairEncoder

# Initialize model & engine
model = GPTModel("config/config.yaml")
engine = GenerationEngine(model)
tokenizer = BytePairEncoder()
if os.path.exists("tokenizer/vocab.json"):
    tokenizer.load("tokenizer/vocab.json")


def generate_text(prompt: str, max_tokens: int = 256, temperature: float = 0.7):
    import torch

    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    output_ids = engine.generate(input_ids, max_new_tokens=int(max_tokens), temperature=float(temperature))
    prompt_len = input_ids.shape[1]
    new_ids = output_ids[prompt_len:]
    return tokenizer.decode(new_ids.tolist())


# Create Gradio UI
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=3, placeholder="Ask XRFM custom language model anything...", label="Prompt"),
        gr.Slider(minimum=16, maximum=1024, value=256, step=16, label="Max Tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
    ],
    outputs=gr.Textbox(label="XRFM Generation Output"),
    title="XR Foundation Model (XRFM) Live API Server",
    description="Custom PyTorch Foundation Model serving OpenAI-compatible APIs & live inferences.",
)

# Mount Gradio interface on FastAPI application
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
