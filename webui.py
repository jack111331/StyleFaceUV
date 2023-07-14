import gradio as gr
import cv2
import numpy as np
import einops
import torch
from typing import Optional

from utils.utility import instantiate_from_config
import yaml

class State:
    current_style_codes: Optional[torch.Tensor] = None
    selected_style_code: Optional[torch.Tensor] = None

state = State()

def generate_fn(model):
    batch_num = 4

    noise = torch.randn(batch_num, 512, device='cuda')
    sampled_style_code = model.generator_2d.get_latent_Wplus(noise)
    state.current_style_codes = sampled_style_code.detach().cpu()

    sampled_img, _ = model.generator_2d(sampled_style_code, truncation=1, truncation_latent=None, input_is_Wplus=True)
    sampled_img = einops.rearrange(sampled_img, 'b c h w -> b h w c')
    sampled_img = ((sampled_img.detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
    return sampled_img

def get_select_index(evt: gr.SelectData):
    state.selected_style_code = state.current_style_codes[evt.index:evt.index+1]

def generate_gltf_and_render_on_model3d(model):
    model.save_as_gltf(torch.Tensor(state.selected_style_code).to('cuda:0'))
    return "test.gltf"

if __name__ == '__main__':

    with open('config/config_webui.yml', 'r') as stream:
        config = yaml.safe_load(stream)

    model_config = config['model']
    model = instantiate_from_config(model_config)
    model = model.eval().to('cuda:0')

    with gr.Blocks(analytics_enabled=False) as demo:
        with gr.Row():
            with gr.Column():
                generate_glry = gr.Gallery(label="Generated 2D image gallery")

                with gr.Row():
                    generate_btn = gr.Button("Generate a sample")
                    # generate_btn.click(fn=generate_fn, outputs=output_img, api_name="generate")
                    generate_btn.click(fn=lambda: generate_fn(model),
                                       outputs=generate_glry, api_name="generate")

            output_model_3d = gr.Model3D(label="3D view")
            generate_glry.select(fn=get_select_index).then(lambda: generate_gltf_and_render_on_model3d(model), None, output_model_3d)

    demo.launch()