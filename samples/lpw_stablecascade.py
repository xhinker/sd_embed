#%%
import gc
import torch
from sd_embed.embedding_funcs import get_weighted_text_embeddings_s_cascade
from diffusers import StableCascadePriorPipeline, StableCascadeDecoderPipeline

prior = StableCascadePriorPipeline.from_pretrained(
    "stabilityai/stable-cascade-prior",
    variant='bf16',
    torch_dtype=torch.bfloat16)

decoder = StableCascadeDecoderPipeline.from_pretrained(
    "stabilityai/stable-cascade",
    variant='bf16',
    torch_dtype=torch.float16)
#%%


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

generator = torch.Generator(device='cuda').manual_seed(3)

# prior
prior.to('cuda')

(
    prompt_embeds
    , negative_prompt_embeds
    , pooled_prompt_embeds
    , negative_prompt_embeds_pooled
) = get_weighted_text_embeddings_s_cascade(prior, prompt, neg_prompt)

prior_output = prior(
    prompt_embeds                   = prompt_embeds
    , negative_prompt_embeds        = negative_prompt_embeds
    , prompt_embeds_pooled          = pooled_prompt_embeds
    , negative_prompt_embeds_pooled = negative_prompt_embeds_pooled
    , num_inference_steps           = 30
    , guidance_scale                = 8
    , height                        = 1024
    , width                         = 1024 + 512
    , generator                     = generator
)

del prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_prompt_embeds_pooled
prior.to('cpu')

# decoder
decoder.to('cuda')

(
    prompt_embeds
    , negative_prompt_embeds
    , pooled_prompt_embeds
    , negative_prompt_embeds_pooled
) = get_weighted_text_embeddings_s_cascade(decoder, prompt, neg_prompt)

image = decoder(
    prompt_embeds                   = prompt_embeds
    , negative_prompt_embeds        = negative_prompt_embeds
    , prompt_embeds_pooled          = pooled_prompt_embeds
    , negative_prompt_embeds_pooled = negative_prompt_embeds_pooled
    , image_embeddings              = prior_output.image_embeddings.half()
    , num_inference_steps           = 10
    , guidance_scale                = 0
    , generator                     = generator
).images[0]

display(image)

del prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_prompt_embeds_pooled
decoder.to('cpu')
gc.collect()
torch.cuda.empty_cache()