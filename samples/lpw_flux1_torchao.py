#%%
from diffusers import DiffusionPipeline, FluxTransformer2DModel
from torchao.quantization import quantize_, int8_weight_only
import torch
from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1

# model_path = "black-forest-labs/FLUX.1-schnell"
model_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-dev_main"

transformer = FluxTransformer2DModel.from_pretrained(
    model_path
    , subfolder = "transformer"
    , torch_dtype = torch.bfloat16
)
quantize_(transformer, int8_weight_only())

pipe = DiffusionPipeline.from_pretrained(
    model_path
    , transformer = transformer
    , torch_dtype = torch.bfloat16
)

pipe.enable_model_cpu_offload()

#%%
prompt = """\
A dreamy, soft-focus photograph capturing a romantic Jane Austen movie scene, 
in the style of Agnes Cecile. Delicate watercolors, misty background, 
Regency-era couple, tender embrace, period clothing, flowing dress, dappled sunlight, 
ethereal glow, gentle expressions, intricate lace, muted pastels, serene countryside, 
timeless romance, poetic atmosphere, wistful mood, look at camera.
"""

prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
    pipe        = pipe
    , prompt    = prompt
)
image = pipe(
    prompt_embeds               = prompt_embeds
    , pooled_prompt_embeds      = pooled_prompt_embeds
    , width                     = 896
    , height                    = 1280
    , num_inference_steps       = 20
    , guidance_scale            = 4.0
    , generator                 = torch.Generator().manual_seed(1234)
).images[0]
display(image)