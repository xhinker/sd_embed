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
pip install git+git@github.com:xhinker/sd_embed.git@main
```

## Stable Diffusion 3

Generate long prompt weighted embeddings for Stable Diffusion 3. 

```py

```