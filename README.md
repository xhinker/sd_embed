# Stable Diffusion Long Prompt Weighted Embedding


<a href="https://www.packtpub.com/product/using-stable-diffusion-with-python/9781835086377"><img src="https://m.media-amazon.com/images/I/81qJBJlgGEL._SL1500_.jpg" alt="Using Stable Diffusion with Python" height="256px" align="right"></a>

Overcoming the 77-token prompt limitation, generating long-weighted prompt embeddings for Stable Diffusion, this module supports generating embedding and pooled embeddings for long prompt weighted. The generated embedding is compatible with [Huggingface Diffusers](https://github.com/huggingface/diffusers). 

The prompt format is compatible with [AUTOMATIC1111 stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

* Support unlimited prompt length for SD1.5 and SDXL
* Support weighting like `a (white:1.2) cat`
* Support parentheses like `a ((white)) cat`
* For SD3, support max 512 tokens (T5 model support max 512 tokens)

Support Stable Diffusion v1.5, SDXL and **Stable Diffusion 3**.

The detailed implementation is covered in chapter 10 of book [Using Stable Diffusion with Python](https://www.amazon.com/Using-Stable-Diffusion-Python-Generation/dp/1835086373/ref=sr_1_1?crid=2EF28F3KPIZMI&dib=eyJ2IjoiMSJ9.quUjZV6jP2UJs4Uv72YiPA.IPU_TA7myv0fiuvYuspHdYbCcFWhg7USvj1p9KI_4RM&dib_tag=se&keywords=Using+Stable+Diffusion+with+Python&qid=1717696681&sprefix=using+stable+diffusion+with+python%2Caps%2C184&sr=8-1)

## Install 

```sh
pip install git+https://github.com/xhinker/sd_embed.git@main
```

## Stable Diffusion 3

Generate long prompt weighted embeddings for Stable Diffusion 3. 

Load up SD3 model:
```py
import gc
import torch
from diffusers import StableDiffusion3Pipeline
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sd3

model_path = "stabilityai/stable-diffusion-3-medium-diffusers"
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16
)
```

Generate the embedding and use it to generate images:
```py
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
    , pooled_prompt_embeds
    , negative_pooled_prompt_embeds
) = get_weighted_text_embeddings_sd3(
    pipe
    , prompt = prompt
    , neg_prompt = neg_prompt
)

image = pipe(
    prompt_embeds                   = prompt_embeds
    , negative_prompt_embeds        = prompt_neg_embeds
    , pooled_prompt_embeds          = pooled_prompt_embeds
    , negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
    , num_inference_steps           = 30
    , height                        = 1024 
    , width                         = 1024 + 512
    , guidance_scale                = 4.0
    , generator                     = torch.Generator("cuda").manual_seed(2)
).images[0]
display(image)

del prompt_embeds, prompt_neg_embeds,pooled_prompt_embeds, negative_pooled_prompt_embeds
pipe.to('cpu')
gc.collect()
torch.cuda.empty_cache()
```