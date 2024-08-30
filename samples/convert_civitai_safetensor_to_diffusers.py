#%%
from sd_embed.conversion_tools import Flux1Convertor
convertor = Flux1Convertor()

#%%
model_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/flux_checkpoints/STOIQONewrealityFLUXSD_F1DPreAlpha.safetensors"
convertor.convert_to_diffuses(input_safetensor_file_path=model_path)

#%% load model
import torch
from diffusers import DiffusionPipeline, FluxTransformer2DModel
from torchao.quantization import quantize_, int8_weight_only, int4_weight_only

model_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_models/flux_checkpoints/STOIQONewrealityFLUXSD_F1DPreAlpha"
transformer = FluxTransformer2DModel.from_pretrained(
    model_path
    , subfolder = "transformer"
    , torch_dtype = torch.bfloat16
)
quantize_(transformer, int8_weight_only())

# model_path = "black-forest-labs/FLUX.1-schnell"
model_path = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/black-forest-labs/FLUX.1-dev_main"
pipe = DiffusionPipeline.from_pretrained(
    model_path
    , transformer = transformer
    , torch_dtype = torch.bfloat16
)

pipe.enable_model_cpu_offload()

#%%
from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1

prompt = """cute cat running on top of a starship from movie star war, vast cosmos as the background, small fighter flying around"""

prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
    pipe        = pipe
    , prompt    = prompt
)

#%%
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