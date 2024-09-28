import logging
import random
import warnings
import os
import gradio as gr
import numpy as np
import spaces
import torch
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from gradio_imageslider import ImageSlider
from PIL import Image
from huggingface_hub import snapshot_download

css = """
#col-container {
    margin: 0 auto;
    max-width: 512px;
}
"""

if torch.cuda.is_available():
    power_device = "GPU"
    device = "cuda"
else:
    power_device = "CPU"
    device = "cpu"


huggingface_token = os.getenv("HUGGINFACE_TOKEN")

model_path = snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev", 
    repo_type="model", 
    ignore_patterns=["*.md", "*..gitattributes"],
    local_dir="FLUX.1-dev",
    token="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", # type a new token-id.
)


# Load pipeline
controlnet = FluxControlNetModel.from_pretrained(
    "jasperai/Flux.1-dev-Controlnet-Upscaler", torch_dtype=torch.bfloat16
).to(device)
pipe = FluxControlNetPipeline.from_pretrained(
    model_path, controlnet=controlnet, torch_dtype=torch.bfloat16
)

#pipe.to(device)
pipe.enable_sequential_cpu_offload()

MAX_SEED = 1000000
MAX_PIXEL_BUDGET = 4096 * 4096


def process_input(input_image, upscale_factor, **kwargs):
    w, h = input_image.size
    w_original, h_original = w, h
    aspect_ratio = w / h

    was_resized = False

    if w * h * upscale_factor**2 > MAX_PIXEL_BUDGET:
        warnings.warn(
            f"Requested output image is too large ({w * upscale_factor}x{h * upscale_factor}). Resizing to ({int(aspect_ratio * MAX_PIXEL_BUDGET ** 0.5 // upscale_factor), int(MAX_PIXEL_BUDGET ** 0.5 // aspect_ratio // upscale_factor)}) pixels."
        )
        gr.Info(
            f"Requested output image is too large ({w * upscale_factor}x{h * upscale_factor}). Resizing input to ({int(aspect_ratio * MAX_PIXEL_BUDGET ** 0.5 // upscale_factor), int(MAX_PIXEL_BUDGET ** 0.5 // aspect_ratio // upscale_factor)}) pixels budget."
        )
        input_image = input_image.resize(
            (
                int(aspect_ratio * MAX_PIXEL_BUDGET**0.5 // upscale_factor),
                int(MAX_PIXEL_BUDGET**0.5 // aspect_ratio // upscale_factor),
            )
        )
        was_resized = True

    # resize to multiple of 8
    w, h = input_image.size
    w = w - w % 8
    h = h - h % 8

    return input_image.resize((w, h)), w_original, h_original, was_resized


@spaces.GPU#(duration=42)
def infer(
    seed,
    randomize_seed,
    input_image,
    num_inference_steps,
    upscale_factor,
    controlnet_conditioning_scale,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    true_input_image = input_image
    input_image, w_original, h_original, was_resized = process_input(
        input_image, upscale_factor
    )

    # rescale with upscale factor
    w, h = input_image.size
    control_image = input_image.resize((w * upscale_factor, h * upscale_factor))

    generator = torch.Generator().manual_seed(seed)

    gr.Info("Upscaling image...")
    image = pipe(
        prompt="",
        control_image=control_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=3.5,
        height=control_image.size[1],
        width=control_image.size[0],
        generator=generator,
    ).images[0]

    if was_resized:
        gr.Info(
            f"Resizing output image to targeted {w_original * upscale_factor}x{h_original * upscale_factor} size."
        )

    # resize to target desired size
    image = image.resize((w_original * upscale_factor, h_original * upscale_factor))
    
    output_name = "Flux.1-dev Upscaler.jpg"
    image.save(output_name, format='JPEG', quality=95)
    # convert to numpy
    return [true_input_image, image, seed]


with gr.Blocks(css=css) as demo:
    # with gr.Column(elem_id="col-container"):
    gr.Markdown(
        f"""
    # ⚡ Flux.1-dev Upscaler ControlNet ⚡
    This is an interactive demo of [Flux.1-dev Upscaler ControlNet](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler) taking as input a low resolution image to generate a high resolution image.
    Currently running on {power_device}.
    """
    )

    with gr.Row():
        run_button = gr.Button(value="Run")

    with gr.Row():
        with gr.Column(scale=4):
            input_im = gr.Image(label="Input Image", type="pil")
        with gr.Column(scale=1):
            num_inference_steps = gr.Slider(
                label="Number of Inference Steps",
                minimum=5,
                maximum=40,
                step=5,
                value=25,
            )
            upscale_factor = gr.Slider(
                label="Upscale Factor",
                minimum=1,
                maximum=8,
                step=1,
                value=4,
            )
            controlnet_conditioning_scale = gr.Slider(
                label="Controlnet Conditioning Scale",
                minimum=0.3,
                maximum=0.9,
                step=0.05,
                value=0.6,
            )
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=42,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

    with gr.Row():
        result = ImageSlider(label="Input / Output", type="pil", interactive=True)


    gr.on(
        [run_button.click],
        fn=infer,
        inputs=[
            seed,
            randomize_seed,
            input_im,
            num_inference_steps,
            upscale_factor,
            controlnet_conditioning_scale,
        ],
        outputs=result,
        show_api=False,
        # show_progress="minimal",
    )

demo.queue().launch(share=False, show_api=False)
