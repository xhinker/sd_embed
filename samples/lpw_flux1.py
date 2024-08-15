#%%
import torch
from optimum.quanto import freeze, qfloat8, quantize
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1

dtype = torch.bfloat16
bfl_repo = "black-forest-labs/FLUX.1-schnell"
pipe = FluxPipeline.from_pretrained(
    pretrained_model_name_or_path   = bfl_repo
    , torch_dtype                   = torch.bfloat16
)

#%%
weight_quant = qfloat8
quantize(pipe.transformer, weights=weight_quant)
freeze(pipe.transformer)

quantize(pipe.text_encoder, weights=weight_quant)
freeze(pipe.text_encoder)

quantize(pipe.text_encoder_2, weights=weight_quant)
freeze(pipe.text_encoder_2)

#%%
prompt = """\
A dreamy, soft-focus photograph capturing a romantic Jane Austen movie scene, 
in the style of Agnes Cecile. Delicate watercolors, misty background, 
Regency-era couple, tender embrace, period clothing, flowing dress, dappled sunlight, 
ethereal glow, gentle expressions, intricate lace, muted pastels, serene countryside, 
timeless romance, poetic atmosphere, wistful mood, look at camera.
"""

pipe_device = 'cuda:0'
pipe.to(pipe_device)
prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
    pipe = pipe
    , prompt = prompt
)

seed = 1234
image = pipe(
    prompt_embeds               = prompt_embeds
    , pooled_prompt_embeds      = pooled_prompt_embeds
    , width                     = 1024 #1280 #1680 #1024 
    , height                    = 1680 #1280 #1024 #1680 #1024
    , num_inference_steps       = 20
    , generator                 = torch.Generator().manual_seed(seed)
    , guidance_scale            = 3.5
).images[0]
display(image)

del prompt_embeds,pooled_prompt_embeds
pipe.to('cpu')
torch.cuda.empty_cache()