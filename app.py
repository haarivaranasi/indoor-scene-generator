import os
from huggingface_hub import login
from dotenv import load_dotenv
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline

load_dotenv()  # Load the environment variables from .env file
hf_token = os.getenv("HF_TOKEN")
login(hf_token)




# Hugging Face token (hardcoded or use os.environ)
login("Your Hugging Face token here")  # ⚠ Replace with your own or read from env

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32
).to(device)
pipe.enable_attention_slicing()

def generate_images(prompt):
    image = pipe(prompt, num_inference_steps=20, height=256, width=256).images[0]
    return [image, image, image]

gr.Interface(
    fn=generate_images,
    inputs=gr.Textbox(label="Describe Your Indoor Scene", placeholder="e.g. A cozy living room with bookshelves"),
    outputs=["image", "image", "image"],
    title="🧠 Zero-Shot Indoor Scene Generator",
    description="Enter a room description and generate 3 AI indoor scenes."
).launch(server_name="0.0.0.0", server_port=7860)
