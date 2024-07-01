#%%
import gc
import torch
from diffusers import StableDiffusionPipeline
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sd15

model_path = "/home/andrewzhu/storage_1t_1/sd15_models/deliberate_v2"
# model_path = "stablediffusionapi/deliberate-v2"
pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16
)

#%%
pipe.to('cuda')

prompt = """A whimsical and creative image depicting a hybrid creature that is a mix of a waffle and a hippopotamus. 
This imaginative creature features the distinctive, bulky body of a hippo, 
but with a texture and appearance resembling a golden-brown, crispy waffle. 
The creature might have elements like waffle squares across its skin and a syrup-like sheen. 
It's set in a surreal environment that playfully combines a natural water habitat of a hippo with elements of a breakfast table setting, 
possibly including oversized utensils or plates in the background. 
The image should evoke a sense of playful absurdity and culinary fantasy.
"""

neg_prompt = """\
skin spots,acnes,skin blemishes,age spot,(ugly:1.2),(duplicate:1.2),(morbid:1.21),(mutilated:1.2),\
(tranny:1.2),mutated hands,(poorly drawn hands:1.5),blurry,(bad anatomy:1.2),(bad proportions:1.3),\
extra limbs,(disfigured:1.2),(missing arms:1.2),(extra legs:1.2),(fused fingers:1.5),\
(too many fingers:1.5),(unclear eyes:1.2),lowers,bad hands,missing fingers,extra digit,\
bad hands,missing fingers,(extra arms and legs),(worst quality:2),(low quality:2),\
(normal quality:2),lowres,((monochrome)),((grayscale))
"""

(
    prompt_embeds
    , prompt_neg_embeds
) = get_weighted_text_embeddings_sd15(
    pipe
    , prompt = prompt
    , neg_prompt = neg_prompt
)

image = pipe(
    prompt_embeds                   = prompt_embeds
    , negative_prompt_embeds        = prompt_neg_embeds
    , num_inference_steps           = 30
    , height                        = 768
    , width                         = 896
    , guidance_scale                = 8.0
    , generator                     = torch.Generator("cuda").manual_seed(2)
).images[0]
display(image)

del prompt_embeds, prompt_neg_embeds
pipe.to('cpu')
gc.collect()
torch.cuda.empty_cache()


#%% without lpw 
pipe.to('cuda')
image = pipe(
    prompt                          = prompt
    , negative_prompt               = neg_prompt
    , num_inference_steps           = 30
    , height                        = 768
    , width                         = 896
    , guidance_scale                = 8.0
    , generator                     = torch.Generator("cuda").manual_seed(2)
).images[0]
display(image)

pipe.to('cpu')
gc.collect()
torch.cuda.empty_cache()