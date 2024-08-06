#%%
import torch
from optimum.quanto import freeze, qfloat8, quantize
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast
from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1

dtype = torch.bfloat16

# bfl_repo = "black-forest-labs/FLUX.1-schnell"
# bfl_repo = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-schnell_main"
bfl_repo = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-dev_main"
revision = "refs/pr/1"

scheduler       = FlowMatchEulerDiscreteScheduler.from_pretrained(
    bfl_repo
    , subfolder     = "scheduler"
    , revision      = revision
)
text_encoder    = CLIPTextModel.from_pretrained(
    bfl_repo
    , subfolder     = "text_encoder"
    , torch_dtype   = dtype
)
tokenizer       = CLIPTokenizer.from_pretrained(
    bfl_repo
    , subfolder     = "tokenizer" 
    , torch_dtype=dtype
)
text_encoder_2  = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision)
tokenizer_2     = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision)
vae             = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
transformer     = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision)

#%%
quantize(transformer, weights=qfloat8)
freeze(transformer)

quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)

#%%
pipe = FluxPipeline(
    scheduler         = scheduler
    , text_encoder    = text_encoder
    , tokenizer       = tokenizer
    , text_encoder_2  = None 
    , tokenizer_2     = tokenizer_2
    , vae             = vae
    , transformer     = None
)
pipe.text_encoder_2 = text_encoder_2
pipe.transformer = transformer

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