## -----------------------------------------------------------------------------
# Generate unlimited size prompt with weighting for SD3&SDXL&SD15
# If you use sd_embed in your research, please cite the following work:
# 
# ```
# @misc{sd_embed_2024,
#   author       = {Shudong Zhu(Andrew Zhu)},
#   title        = {Long Prompt Weighted Stable Diffusion Embedding},
#   howpublished = {\url{https://github.com/xhinker/sd_embed}},
#   year         = {2024},
# }
# ```
# Author: Andrew Zhu
# Book: Using Stable Diffusion with Python, https://www.amazon.com/Using-Stable-Diffusion-Python-Generation/dp/1835086373
# Github: https://github.com/xhinker
# Medium: https://medium.com/@xhinker
## -----------------------------------------------------------------------------

from sd_embed.prompt_parser import parse_prompt_attention
from transformers import CLIPTokenizer,T5EncoderModel,T5Tokenizer
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import StableDiffusion3Pipeline
import math
from diffusers import FluxPipeline

def get_prompts_tokens_with_weights(
    clip_tokenizer: CLIPTokenizer
    , prompt: str = None
):
    """
    Get prompt token ids and weights, this function works for both prompt and negative prompt
    
    Args:
        pipe (CLIPTokenizer)
            A CLIPTokenizer
        prompt (str)
            A prompt string with weights
            
    Returns:
        text_tokens (list)
            A list contains token ids
        text_weight (list) 
            A list contains the correspodent weight of token ids
    
    Example:
        import torch
        from diffusers_plus.tools.sd_embeddings import get_prompts_tokens_with_weights
        from transformers import CLIPTokenizer

        clip_tokenizer = CLIPTokenizer.from_pretrained(
            "stablediffusionapi/deliberate-v2"
            , subfolder = "tokenizer"
            , dtype = torch.float16
        )

        token_id_list, token_weight_list = get_prompts_tokens_with_weights(
            clip_tokenizer = clip_tokenizer
            ,prompt = "a (red:1.5) cat"*70
        )
    """
    if (prompt is None) or (len(prompt)<1):
        prompt = "empty"
    
    texts_and_weights = parse_prompt_attention(prompt)
    text_tokens,text_weights = [],[]
    for word, weight in texts_and_weights:
        # tokenize and discard the starting and the ending token
        token = clip_tokenizer(
            word
            , truncation = False        # so that tokenize whatever length prompt
        ).input_ids[1:-1]
        # the returned token is a 1d list: [320, 1125, 539, 320]
        
        # merge the new tokens to the all tokens holder: text_tokens
        text_tokens = [*text_tokens,*token]
        
        # each token chunk will come with one weight, like ['red cat', 2.0]
        # need to expand weight for each token.
        chunk_weights = [weight] * len(token) 
        
        # append the weight back to the weight holder: text_weights
        text_weights = [*text_weights, *chunk_weights]
    return text_tokens,text_weights

def get_prompts_tokens_with_weights_t5(
    t5_tokenizer: T5Tokenizer
    , prompt: str
):
    """
    Get prompt token ids and weights, this function works for both prompt and negative prompt
    """
    if (prompt is None) or (len(prompt)<1):
        prompt = "empty"
    
    texts_and_weights = parse_prompt_attention(prompt)
    text_tokens,text_weights = [],[]
    for word, weight in texts_and_weights:
        # tokenize and discard the starting and the ending token
        token = t5_tokenizer(
            word
            , truncation            = False        # so that tokenize whatever length prompt
            , add_special_tokens    = True
        ).input_ids
        # the returned token is a 1d list: [320, 1125, 539, 320]
        
        # merge the new tokens to the all tokens holder: text_tokens
        text_tokens = [*text_tokens,*token]
        
        # each token chunk will come with one weight, like ['red cat', 2.0]
        # need to expand weight for each token.
        chunk_weights = [weight] * len(token) 
        
        # append the weight back to the weight holder: text_weights
        text_weights = [*text_weights, *chunk_weights]
    return text_tokens,text_weights

def group_tokens_and_weights(
    token_ids: list
    , weights: list
    , pad_last_block = False
):
    """
    Produce tokens and weights in groups and pad the missing tokens
    
    Args:
        token_ids (list)
            The token ids from tokenizer
        weights (list)
            The weights list from function get_prompts_tokens_with_weights
        pad_last_block (bool)
            Control if fill the last token list to 75 tokens with eos
    Returns:
        new_token_ids (2d list)
        new_weights (2d list)
    
    Example:
        from diffusers_plus.tools.sd_embeddings import group_tokens_and_weights
        token_groups,weight_groups = group_tokens_and_weights(
            token_ids = token_id_list
            , weights = token_weight_list
        )
    """
    bos,eos = 49406,49407
    
    # this will be a 2d list 
    new_token_ids = []
    new_weights   = []  
    while len(token_ids) >= 75:
        # get the first 75 tokens
        head_75_tokens = [token_ids.pop(0) for _ in range(75)]
        head_75_weights = [weights.pop(0) for _ in range(75)]
        
        # extract token ids and weights
        temp_77_token_ids = [bos] + head_75_tokens + [eos]
        temp_77_weights   = [1.0] + head_75_weights + [1.0]
        
        # add 77 token and weights chunk to the holder list
        new_token_ids.append(temp_77_token_ids)
        new_weights.append(temp_77_weights)
    
    # padding the left
    if len(token_ids) > 0:
        padding_len         = 75 - len(token_ids) if pad_last_block else 0
        
        temp_77_token_ids   = [bos] + token_ids + [eos] * padding_len + [eos]
        new_token_ids.append(temp_77_token_ids)
        
        temp_77_weights     = [1.0] + weights   + [1.0] * padding_len + [1.0]
        new_weights.append(temp_77_weights)
        
    return new_token_ids, new_weights

@torch.inference_mode()
def get_weighted_text_embeddings_sd15(
        pipe: StableDiffusionPipeline
        , prompt : str          = ""
        , neg_prompt: str       = ""
        , pad_last_block: bool  = False
        , clip_skip: int        = 0
        , device: str           = None
):
    """
    This function can process long prompt with weights, no length limitation 
    for Stable Diffusion v1.5
    
    Args:
        pipe (StableDiffusionPipeline)
        prompt (str)
        neg_prompt (str)
    Returns:
        prompt_embeds (torch.Tensor)
        neg_prompt_embeds (torch.Tensor)
    
    Example:
        from diffusers import StableDiffusionPipeline
        text2img_pipe = StableDiffusionPipeline.from_pretrained(
            "stablediffusionapi/deliberate-v2"
            , torch_dtype = torch.float16
            , safety_checker = None
        ).to("cuda:0")
        prompt_embeds, neg_prompt_embeds = get_weighted_text_embeddings_v15(
            pipe = text2img_pipe
            , prompt = "a (white) cat" 
            , neg_prompt = "blur"
        )
        image = text2img_pipe(
            prompt_embeds = prompt_embeds
            , negative_prompt_embeds = neg_prompt_embeds
            , generator = torch.Generator(text2img_pipe.device).manual_seed(2)
        ).images[0]
    """
    device = device if device else pipe.device

    original_clip_layers = pipe.text_encoder.text_model.encoder.layers
    if clip_skip > 0:
        pipe.text_encoder.text_model.encoder.layers = original_clip_layers[:-clip_skip]
    
    eos = pipe.tokenizer.eos_token_id 
    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, prompt
    )
    neg_prompt_tokens, neg_prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, neg_prompt
    )
    
    # padding the shorter one
    prompt_token_len        = len(prompt_tokens)
    neg_prompt_token_len    = len(neg_prompt_tokens)
    if prompt_token_len > neg_prompt_token_len:
        # padding the neg_prompt with eos token
        neg_prompt_tokens   = (
            neg_prompt_tokens  + 
            [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        neg_prompt_weights  = (
            neg_prompt_weights + 
            [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    else:
        # padding the prompt
        prompt_tokens       = (
            prompt_tokens  
            + [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        prompt_weights      = (
            prompt_weights 
            + [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    
    embeds = []
    neg_embeds = []
    
    prompt_token_groups ,prompt_weight_groups = group_tokens_and_weights(
        prompt_tokens.copy()
        , prompt_weights.copy()
        , pad_last_block = pad_last_block
    )
    
    neg_prompt_token_groups, neg_prompt_weight_groups = group_tokens_and_weights(
        neg_prompt_tokens.copy()
        , neg_prompt_weights.copy()
        , pad_last_block = pad_last_block
    )
        
    # get prompt embeddings one by one is not working
    # we must embed prompt group by group
    for i in range(len(prompt_token_groups)):
        # get positive prompt embeddings with weights
        token_tensor = torch.tensor(
            [prompt_token_groups[i]]
            ,dtype = torch.long, device = device
        )
        weight_tensor = torch.tensor(
            prompt_weight_groups[i]
            , dtype     = torch.float16
            , device    = device
        )
        
        token_embedding = pipe.text_encoder(token_tensor)[0].squeeze(0) 
        for j in range(len(weight_tensor)):
            token_embedding[j] = token_embedding[j] * weight_tensor[j]
        embeds.append(token_embedding.unsqueeze(0).cpu())
        
        # get negative prompt embeddings with weights
        neg_token_tensor = torch.tensor(
            [neg_prompt_token_groups[i]]
            , dtype = torch.long, device = device
        )
        neg_weight_tensor = torch.tensor(
            neg_prompt_weight_groups[i]
            , dtype     = torch.float16
            , device    = device
        )
        neg_token_embedding = pipe.text_encoder(neg_token_tensor)[0].squeeze(0) 
        for z in range(len(neg_weight_tensor)):
            neg_token_embedding[z] = (
                neg_token_embedding[z] * neg_weight_tensor[z]
            )
        neg_embeds.append(neg_token_embedding.unsqueeze(0).cpu())

        # Free VRAM & RAM
        token_tensor.cpu()
        weight_tensor.cpu()
        neg_token_tensor.cpu()
        neg_weight_tensor.cpu()
        del token_tensor, \
            weight_tensor, \
            token_embedding, \
            neg_token_tensor, \
            neg_weight_tensor, \
            neg_token_embedding
        torch.cuda.empty_cache()
    
    prompt_embeds = torch.cat(embeds, dim = 1).to(device)
    neg_prompt_embeds = torch.cat(neg_embeds, dim = 1).to(device)
    
    # recover clip layers
    if clip_skip > 0:
        pipe.text_encoder.text_model.encoder.layers = original_clip_layers

    return prompt_embeds, neg_prompt_embeds

@torch.inference_mode()
def get_weighted_text_embeddings_sdxl(
        pipe: StableDiffusionXLPipeline
        , prompt : str          = ""
        , neg_prompt: str       = ""
        , pad_last_block: bool  = True
        , device: str           = None
):
    """
    This function can process long prompt with weights, no length limitation 
    for Stable Diffusion XL
    
    Args:
        pipe (StableDiffusionPipeline)
        prompt (str)
        neg_prompt (str)
    Returns:
        prompt_embeds (torch.Tensor)
        neg_prompt_embeds (torch.Tensor)
    
    Example:
        from diffusers import StableDiffusionPipeline
        text2img_pipe = StableDiffusionPipeline.from_pretrained(
            "stablediffusionapi/deliberate-v2"
            , torch_dtype = torch.float16
            , safety_checker = None
        ).to("cuda:0")
        prompt_embeds, neg_prompt_embeds = get_weighted_text_embeddings_v15(
            pipe = text2img_pipe
            , prompt = "a (white) cat" 
            , neg_prompt = "blur"
        )
        image = text2img_pipe(
            prompt_embeds = prompt_embeds
            , negative_prompt_embeds = neg_prompt_embeds
            , generator = torch.Generator(text2img_pipe.device).manual_seed(2)
        ).images[0]
    """
    import math
    eos = pipe.tokenizer.eos_token_id 
    
    device = device if device else pipe.device

    # tokenizer 1
    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, prompt
    )

    neg_prompt_tokens, neg_prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, neg_prompt
    )
    
    # tokenizer 2
    prompt_tokens_2, prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, prompt
    )

    neg_prompt_tokens_2, neg_prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, neg_prompt
    )
    
    # padding the shorter one
    prompt_token_len        = len(prompt_tokens)
    neg_prompt_token_len    = len(neg_prompt_tokens)
    
    if prompt_token_len > neg_prompt_token_len:
        # padding the neg_prompt with eos token
        neg_prompt_tokens   = (
            neg_prompt_tokens  + 
            [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        neg_prompt_weights  = (
            neg_prompt_weights + 
            [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    else:
        # padding the prompt
        prompt_tokens       = (
            prompt_tokens  
            + [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        prompt_weights      = (
            prompt_weights 
            + [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    
    # padding the shorter one for token set 2
    prompt_token_len_2        = len(prompt_tokens_2)
    neg_prompt_token_len_2    = len(neg_prompt_tokens_2)
    
    if prompt_token_len_2 > neg_prompt_token_len_2:
        # padding the neg_prompt with eos token
        neg_prompt_tokens_2   = (
            neg_prompt_tokens_2  + 
            [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        neg_prompt_weights_2  = (
            neg_prompt_weights_2 + 
            [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    else:
        # padding the prompt
        prompt_tokens_2       = (
            prompt_tokens_2  
            + [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        prompt_weights_2      = (
            prompt_weights_2 
            + [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    
    embeds = []
    neg_embeds = []
    
    prompt_token_groups, prompt_weight_groups = group_tokens_and_weights(
        prompt_tokens.copy()
        , prompt_weights.copy()
        , pad_last_block = pad_last_block
    )
    
    neg_prompt_token_groups, neg_prompt_weight_groups = group_tokens_and_weights(
        neg_prompt_tokens.copy()
        , neg_prompt_weights.copy()
        , pad_last_block = pad_last_block
    )
    
    prompt_token_groups_2, prompt_weight_groups_2 = group_tokens_and_weights(
        prompt_tokens_2.copy()
        , prompt_weights_2.copy()
        , pad_last_block = pad_last_block
    )
    
    neg_prompt_token_groups_2, neg_prompt_weight_groups_2 = group_tokens_and_weights(
        neg_prompt_tokens_2.copy()
        , neg_prompt_weights_2.copy()
        , pad_last_block = pad_last_block
    )
        
    # get prompt embeddings one by one is not working. 
    for i in range(len(prompt_token_groups)):
        # get positive prompt embeddings with weights
        token_tensor = torch.tensor(
            [prompt_token_groups[i]]
            ,dtype = torch.long, device = device
        )
        weight_tensor = torch.tensor(
            prompt_weight_groups[i]
            , dtype     = torch.float16
            , device    = device
        )
        
        token_tensor_2 = torch.tensor(
            [prompt_token_groups_2[i]]
            ,dtype = torch.long, device = device
        )
        
        # use first text encoder
        prompt_embeds_1 = pipe.text_encoder(
            token_tensor.to(device)
            , output_hidden_states = True
        )
        prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[-2].cpu()

        # use second text encoder
        prompt_embeds_2 = pipe.text_encoder_2(
            token_tensor_2.to(device)
            , output_hidden_states = True
        )
        prompt_embeds_2_hidden_states = prompt_embeds_2.hidden_states[-2].cpu()
        pooled_prompt_embeds = prompt_embeds_2[0].cpu()

        prompt_embeds_list = [prompt_embeds_1_hidden_states, prompt_embeds_2_hidden_states]
        token_embedding = torch.concat(prompt_embeds_list, dim=-1).squeeze(0).to(device)
        
        for j in range(len(weight_tensor)):
            if weight_tensor[j] != 1.0:
                #ow = weight_tensor[j] - 1
                
                # optional process
                # To map number of (0,1) to (-1,1)
                # tanh_weight = (math.exp(ow)/(math.exp(ow) + 1) - 0.5) * 2
                # weight = 1 + tanh_weight
                
                # add weight method 1:
                # token_embedding[j] = token_embedding[j] * weight
                # token_embedding[j] = (
                #     token_embedding[-1] + (token_embedding[j] - token_embedding[-1]) * weight
                # )
                
                # add weight method 2:
                # token_embedding[j] = (
                #     token_embedding[-1] + (token_embedding[j] - token_embedding[-1]) * weight_tensor[j]
                # )
                
                # add weight method 3:
                token_embedding[j] = token_embedding[j] * weight_tensor[j]

        token_embedding = token_embedding.unsqueeze(0)
        embeds.append(token_embedding.cpu())
        
        # get negative prompt embeddings with weights
        neg_token_tensor = torch.tensor(
            [neg_prompt_token_groups[i]]
            , dtype = torch.long, device = device
        )
        neg_token_tensor_2 = torch.tensor(
            [neg_prompt_token_groups_2[i]]
            , dtype = torch.long, device = device
        )
        neg_weight_tensor = torch.tensor(
            neg_prompt_weight_groups[i]
            , dtype     = torch.float16
            , device    = device
        )
        
        # use first text encoder
        neg_prompt_embeds_1 = pipe.text_encoder(
            neg_token_tensor.to(device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_1_hidden_states = neg_prompt_embeds_1.hidden_states[-2].cpu()

        # use second text encoder
        neg_prompt_embeds_2 = pipe.text_encoder_2(
            neg_token_tensor_2.to(device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2.hidden_states[-2].cpu()
        negative_pooled_prompt_embeds = neg_prompt_embeds_2[0].cpu()

        neg_prompt_embeds_list = [neg_prompt_embeds_1_hidden_states, neg_prompt_embeds_2_hidden_states]
        neg_token_embedding = torch.concat(neg_prompt_embeds_list, dim=-1).squeeze(0).to(device)
        
        for z in range(len(neg_weight_tensor)):
            if neg_weight_tensor[z] != 1.0:
                
                # ow = neg_weight_tensor[z] - 1
                # neg_weight = 1 + (math.exp(ow)/(math.exp(ow) + 1) - 0.5) * 2
                
                # add weight method 1:
                # neg_token_embedding[z] = neg_token_embedding[z] * neg_weight
                # neg_token_embedding[z] = (
                #     neg_token_embedding[-1] + (neg_token_embedding[z] - neg_token_embedding[-1]) * neg_weight
                # )
                
                # add weight method 2:
                # neg_token_embedding[z] = (
                #     neg_token_embedding[-1] + (neg_token_embedding[z] - neg_token_embedding[-1]) * neg_weight_tensor[z]
                # )
                
                # add weight method 3:
                neg_token_embedding[z] = neg_token_embedding[z] * neg_weight_tensor[z]
                
        neg_token_embedding = neg_token_embedding.unsqueeze(0)
        neg_embeds.append(neg_token_embedding.cpu())

        # Free VRAM & RAM
        del prompt_embeds_1_hidden_states, \
            prompt_embeds_2_hidden_states, \
            neg_prompt_embeds_1_hidden_states, \
            neg_prompt_embeds_2_hidden_states, \
            prompt_embeds_1, \
            prompt_embeds_2, \
            neg_prompt_embeds_1, \
            neg_prompt_embeds_2
        torch.cuda.empty_cache()
    
    prompt_embeds           = torch.cat(embeds, dim = 1).to(device)
    negative_prompt_embeds  = torch.cat(neg_embeds, dim = 1).to(device)
    
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

@torch.inference_mode()
def get_weighted_text_embeddings_sdxl_refiner(
        pipe: StableDiffusionXLPipeline
        , prompt : str       = ""
        , neg_prompt: str    = ""
        , device: str        = None
):
    """
    This function can process long prompt with weights, no length limitation 
    for Stable Diffusion XL
    
    Args:
        pipe (StableDiffusionPipeline)
        prompt (str)
        neg_prompt (str)
    Returns:
        prompt_embeds (torch.Tensor)
        neg_prompt_embeds (torch.Tensor)
    
    Example:
        from diffusers import StableDiffusionPipeline
        text2img_pipe = StableDiffusionPipeline.from_pretrained(
            "stablediffusionapi/deliberate-v2"
            , torch_dtype = torch.float16
            , safety_checker = None
        ).to("cuda:0")
        prompt_embeds, neg_prompt_embeds = get_weighted_text_embeddings_v15(
            pipe = text2img_pipe
            , prompt = "a (white) cat" 
            , neg_prompt = "blur"
        )
        image = text2img_pipe(
            prompt_embeds = prompt_embeds
            , negative_prompt_embeds = neg_prompt_embeds
            , generator = torch.Generator(text2img_pipe.device).manual_seed(2)
        ).images[0]
    """
    import math
    eos = 49407 #pipe.tokenizer.eos_token_id 
    
    device = device if device else pipe.device

    # tokenizer 2
    prompt_tokens_2, prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, prompt
    )

    neg_prompt_tokens_2, neg_prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, neg_prompt
    )
    
    # padding the shorter one for token set 2
    prompt_token_len_2        = len(prompt_tokens_2)
    neg_prompt_token_len_2    = len(neg_prompt_tokens_2)
    
    if prompt_token_len_2 > neg_prompt_token_len_2:
        # padding the neg_prompt with eos token
        neg_prompt_tokens_2   = (
            neg_prompt_tokens_2  + 
            [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        neg_prompt_weights_2  = (
            neg_prompt_weights_2 + 
            [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    else:
        # padding the prompt
        prompt_tokens_2       = (
            prompt_tokens_2  
            + [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        prompt_weights_2      = (
            prompt_weights_2 
            + [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    
    embeds = []
    neg_embeds = []
    
    prompt_token_groups_2, prompt_weight_groups_2 = group_tokens_and_weights(
        prompt_tokens_2.copy()
        , prompt_weights_2.copy()
    )
    
    neg_prompt_token_groups_2, neg_prompt_weight_groups_2 = group_tokens_and_weights(
        neg_prompt_tokens_2.copy()
        , neg_prompt_weights_2.copy()
    )
        
    # get prompt embeddings one by one is not working. 
    for i in range(len(prompt_token_groups_2)):
        # get positive prompt embeddings with weights        
        token_tensor_2 = torch.tensor(
            [prompt_token_groups_2[i]]
            ,dtype = torch.long, device = device
        )
        
        weight_tensor_2 = torch.tensor(
            prompt_weight_groups_2[i]
            , dtype     = torch.float16
            , device    = device
        )

        # use second text encoder
        prompt_embeds_2 = pipe.text_encoder_2(
            token_tensor_2.to(device)
            , output_hidden_states = True
        )
        prompt_embeds_2_hidden_states = prompt_embeds_2.hidden_states[-2]
        pooled_prompt_embeds = prompt_embeds_2[0]

        prompt_embeds_list = [prompt_embeds_2_hidden_states]
        token_embedding = torch.concat(prompt_embeds_list, dim=-1).squeeze(0)
        
        for j in range(len(weight_tensor_2)):
            if weight_tensor_2[j] != 1.0:
                ow = weight_tensor_2[j] - 1
                
                # optional process
                # To map number of (0,1) to (-1,1)
                tanh_weight = (math.exp(ow)/(math.exp(ow) + 1) - 0.5) * 2
                weight = 1 + tanh_weight
                
                # add weight method 1:
                # token_embedding[j] = token_embedding[j] * weight
                # token_embedding[j] = (
                #     token_embedding[-1] + (token_embedding[j] - token_embedding[-1]) * weight
                # )
                
                # add weight method 2:
                token_embedding[j] = (
                    token_embedding[-1] + (token_embedding[j] - token_embedding[-1]) * weight_tensor_2[j]
                )

        token_embedding = token_embedding.unsqueeze(0)
        
        # Free VRAM
        embeds.append(token_embedding.cpu())
        token_tensor_2.cpu()
        weight_tensor_2.cpu()
        prompt_embeds_2_hidden_states.cpu()
        del token_tensor_2, \
            weight_tensor_2, \
            prompt_embeds_2, \
            prompt_embeds_2_hidden_states, \
            token_embedding
        torch.cuda.empty_cache()

        # get negative prompt embeddings with weights
        neg_token_tensor_2 = torch.tensor(
            [neg_prompt_token_groups_2[i]]
            , dtype = torch.long, device = device
        )
        neg_weight_tensor_2 = torch.tensor(
            neg_prompt_weight_groups_2[i]
            , dtype     = torch.float16
            , device    = device
        )
        
        # use second text encoder
        neg_prompt_embeds_2 = pipe.text_encoder_2(
            neg_token_tensor_2.to(device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2.hidden_states[-2]
        negative_pooled_prompt_embeds = neg_prompt_embeds_2[0]

        neg_prompt_embeds_list = [neg_prompt_embeds_2_hidden_states]
        neg_token_embedding = torch.concat(neg_prompt_embeds_list, dim=-1).squeeze(0)
        
        for z in range(len(neg_weight_tensor_2)):
            if neg_weight_tensor_2[z] != 1.0:
                
                ow = neg_weight_tensor_2[z] - 1
                #neg_weight = 1 + (math.exp(ow)/(math.exp(ow) + 1) - 0.5) * 2
                
                # add weight method 1:
                # neg_token_embedding[z] = neg_token_embedding[z] * neg_weight
                # neg_token_embedding[z] = (
                #     neg_token_embedding[-1] + (neg_token_embedding[z] - neg_token_embedding[-1]) * neg_weight
                # )
                
                # add weight method 2:
                neg_token_embedding[z] = (
                    neg_token_embedding[-1] + (neg_token_embedding[z] - neg_token_embedding[-1]) * neg_weight_tensor_2[z]
                )
                
        neg_token_embedding = neg_token_embedding.unsqueeze(0)
        
        # Free VRAM
        neg_embeds.append(neg_token_embedding.cpu())
        neg_token_tensor_2.cpu()
        neg_weight_tensor_2.cpu()
        neg_prompt_embeds_2_hidden_states.cpu()
        del neg_token_tensor_2, \
            neg_weight_tensor_2, \
            neg_prompt_embeds_2, \
            neg_prompt_embeds_2_hidden_states,\
            neg_token_embedding
        torch.cuda.empty_cache()

    prompt_embeds = torch.cat([e.to(device) for e in embeds], dim = 1)

    # Free RAM as needed
    del embeds
    
    negative_prompt_embeds = torch.cat([e.to(device) for e in neg_embeds], dim = 1)
    del neg_embeds

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

@torch.inference_mode()
def get_weighted_text_embeddings_sdxl_2p(
        pipe: StableDiffusionXLPipeline
        , prompt : str       = ""
        , prompt_2 : str     = None
        , neg_prompt: str    = ""
        , neg_prompt_2: str  = None
        , device: str        = None
):
    """
    This function can process long prompt with weights, no length limitation 
    for Stable Diffusion XL, support two prompt sets.
    
    Args:
        pipe (StableDiffusionPipeline)
        prompt (str)
        neg_prompt (str)
    Returns:
        prompt_embeds (torch.Tensor)
        neg_prompt_embeds (torch.Tensor)
    
    Example:
        from diffusers import StableDiffusionPipeline
        text2img_pipe = StableDiffusionPipeline.from_pretrained(
            "stablediffusionapi/deliberate-v2"
            , torch_dtype = torch.float16
            , safety_checker = None
        ).to("cuda:0")
        prompt_embeds, neg_prompt_embeds = get_weighted_text_embeddings_v15(
            pipe = text2img_pipe
            , prompt = "a (white) cat" 
            , neg_prompt = "blur"
        )
        image = text2img_pipe(
            prompt_embeds = prompt_embeds
            , negative_prompt_embeds = neg_prompt_embeds
            , generator = torch.Generator(text2img_pipe.device).manual_seed(2)
        ).images[0]
    """
    prompt_2        = prompt_2 or prompt
    neg_prompt_2    = neg_prompt_2 or neg_prompt
    
    import math
    eos = pipe.tokenizer.eos_token_id 
    
    device = device if device else pipe.device

    # tokenizer 1
    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, prompt
    )

    neg_prompt_tokens, neg_prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, neg_prompt
    )
    
    # tokenizer 2
    prompt_tokens_2, prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, prompt_2
    )

    neg_prompt_tokens_2, neg_prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, neg_prompt_2
    )
    
    # padding the shorter one
    prompt_token_len        = len(prompt_tokens)
    neg_prompt_token_len    = len(neg_prompt_tokens)
    
    if prompt_token_len > neg_prompt_token_len:
        # padding the neg_prompt with eos token
        neg_prompt_tokens   = (
            neg_prompt_tokens  + 
            [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        neg_prompt_weights  = (
            neg_prompt_weights + 
            [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    else:
        # padding the prompt
        prompt_tokens       = (
            prompt_tokens  
            + [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        prompt_weights      = (
            prompt_weights 
            + [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    
    # padding the shorter one for token set 2
    prompt_token_len_2        = len(prompt_tokens_2)
    neg_prompt_token_len_2    = len(neg_prompt_tokens_2)
    
    if prompt_token_len_2 > neg_prompt_token_len_2:
        # padding the neg_prompt with eos token
        neg_prompt_tokens_2   = (
            neg_prompt_tokens_2  + 
            [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        neg_prompt_weights_2  = (
            neg_prompt_weights_2 + 
            [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    else:
        # padding the prompt
        prompt_tokens_2       = (
            prompt_tokens_2  
            + [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        prompt_weights_2      = (
            prompt_weights_2
            + [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    
    # now, need to ensure prompt and prompt_2 has the same lemgth
    prompt_token_len        = len(prompt_tokens)
    prompt_token_len_2      = len(prompt_tokens_2)
    if prompt_token_len > prompt_token_len_2:
        prompt_tokens_2     = prompt_tokens_2   + [eos] * abs(prompt_token_len - prompt_token_len_2)
        prompt_weights_2    = prompt_weights_2  + [1.0] * abs(prompt_token_len - prompt_token_len_2)
    else:
        prompt_tokens       = prompt_tokens     + [eos] * abs(prompt_token_len - prompt_token_len_2)
        prompt_weights      = prompt_weights    + [1.0] * abs(prompt_token_len - prompt_token_len_2)
    
    # now, need to ensure neg_prompt and net_prompt_2 has the same lemgth
    neg_prompt_token_len        = len(neg_prompt_tokens)
    neg_prompt_token_len_2      = len(neg_prompt_tokens_2)
    if neg_prompt_token_len > neg_prompt_token_len_2:
        neg_prompt_tokens_2     = neg_prompt_tokens_2   + [eos] * abs(neg_prompt_token_len - neg_prompt_token_len_2)
        neg_prompt_weights_2    = neg_prompt_weights_2  + [1.0] * abs(neg_prompt_token_len - neg_prompt_token_len_2)
    else:
        neg_prompt_tokens       = neg_prompt_tokens     + [eos] * abs(neg_prompt_token_len - neg_prompt_token_len_2)
        neg_prompt_weights      = neg_prompt_weights    + [1.0] * abs(neg_prompt_token_len - neg_prompt_token_len_2)
    
    embeds = []
    neg_embeds = []
    
    prompt_token_groups, prompt_weight_groups = group_tokens_and_weights(
        prompt_tokens.copy()
        , prompt_weights.copy()
    )
    
    neg_prompt_token_groups, neg_prompt_weight_groups = group_tokens_and_weights(
        neg_prompt_tokens.copy()
        , neg_prompt_weights.copy()
    )
    
    prompt_token_groups_2, prompt_weight_groups_2 = group_tokens_and_weights(
        prompt_tokens_2.copy()
        , prompt_weights_2.copy()
    )
    
    neg_prompt_token_groups_2, neg_prompt_weight_groups_2 = group_tokens_and_weights(
        neg_prompt_tokens_2.copy()
        , neg_prompt_weights_2.copy()
    )
        
    # get prompt embeddings one by one is not working. 
    for i in range(len(prompt_token_groups)):
        # get positive prompt embeddings with weights
        token_tensor = torch.tensor(
            [prompt_token_groups[i]]
            ,dtype = torch.long, device = device
        )
        weight_tensor = torch.tensor(
            prompt_weight_groups[i]
            , device    = device
        )
        
        token_tensor_2 = torch.tensor(
            [prompt_token_groups_2[i]]
            , device = device
        )
        
        weight_tensor_2 = torch.tensor(
            prompt_weight_groups_2[i]
            , device    = device
        )
        
        # use first text encoder
        prompt_embeds_1 = pipe.text_encoder(
            token_tensor.to(device)
            , output_hidden_states = True
        )
        prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[-2]

        # use second text encoder
        prompt_embeds_2 = pipe.text_encoder_2(
            token_tensor_2.to(device)
            , output_hidden_states = True
        )
        prompt_embeds_2_hidden_states = prompt_embeds_2.hidden_states[-2]
        pooled_prompt_embeds = prompt_embeds_2[0]
        
        prompt_embeds_1_hidden_states = prompt_embeds_1_hidden_states.squeeze(0)
        prompt_embeds_2_hidden_states = prompt_embeds_2_hidden_states.squeeze(0)
        
        for j in range(len(weight_tensor)):
            if weight_tensor[j] != 1.0:
                prompt_embeds_1_hidden_states[j] = (
                    prompt_embeds_1_hidden_states[-1] + (prompt_embeds_1_hidden_states[j] - prompt_embeds_1_hidden_states[-1]) * weight_tensor[j]
                )
            
            if weight_tensor_2[j] != 1.0:
                prompt_embeds_2_hidden_states[j] = (
                    prompt_embeds_2_hidden_states[-1] + (prompt_embeds_2_hidden_states[j] - prompt_embeds_2_hidden_states[-1]) * weight_tensor_2[j]
                )
                              
        prompt_embeds_1_hidden_states = prompt_embeds_1_hidden_states.unsqueeze(0)
        prompt_embeds_2_hidden_states = prompt_embeds_2_hidden_states.unsqueeze(0)
        
        prompt_embeds_list = [prompt_embeds_1_hidden_states, prompt_embeds_2_hidden_states]
        token_embedding = torch.cat(prompt_embeds_list, dim=-1)
        
        embeds.append(token_embedding.cpu())
        
        # get negative prompt embeddings with weights
        neg_token_tensor = torch.tensor(
            [neg_prompt_token_groups[i]]
            , device = device
        )
        neg_token_tensor_2 = torch.tensor(
            [neg_prompt_token_groups_2[i]]
            , device = device
        )
        neg_weight_tensor = torch.tensor(
            neg_prompt_weight_groups[i]
            , device    = device
        )
        neg_weight_tensor_2 = torch.tensor(
            neg_prompt_weight_groups_2[i]
            , device    = device
        )
        
        # use first text encoder
        neg_prompt_embeds_1 = pipe.text_encoder(
            neg_token_tensor.to(device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_1_hidden_states = neg_prompt_embeds_1.hidden_states[-2]

        # use second text encoder
        neg_prompt_embeds_2 = pipe.text_encoder_2(
            neg_token_tensor_2.to(device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2.hidden_states[-2]
        negative_pooled_prompt_embeds = neg_prompt_embeds_2[0]
        
        neg_prompt_embeds_1_hidden_states = neg_prompt_embeds_1_hidden_states.squeeze(0)
        neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2_hidden_states.squeeze(0)
        
        for z in range(len(neg_weight_tensor)):
            if neg_weight_tensor[z] != 1.0:
                neg_prompt_embeds_1_hidden_states[z] = (
                    neg_prompt_embeds_1_hidden_states[-1] + (neg_prompt_embeds_1_hidden_states[z] - neg_prompt_embeds_1_hidden_states[-1]) * neg_weight_tensor[z]
                )
            
            if neg_weight_tensor_2[z] != 1.0:
                neg_prompt_embeds_2_hidden_states[z] = (
                    neg_prompt_embeds_2_hidden_states[-1] + (neg_prompt_embeds_2_hidden_states[z] - neg_prompt_embeds_2_hidden_states[-1]) * neg_weight_tensor_2[z]
                )
        
        neg_prompt_embeds_1_hidden_states = neg_prompt_embeds_1_hidden_states.unsqueeze(0)
        neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2_hidden_states.unsqueeze(0)

        neg_prompt_embeds_list = [neg_prompt_embeds_1_hidden_states, neg_prompt_embeds_2_hidden_states]
        neg_token_embedding = torch.cat(neg_prompt_embeds_list, dim=-1)
        
        neg_embeds.append(neg_token_embedding.cpu())

        # Free VRAM
        token_tensor.cpu()
        weight_tensor.cpu()
        neg_token_tensor.cpu()
        neg_weight_tensor.cpu()
        del token_tensor,\
            weight_tensor,\
            token_embedding, \
            neg_token_tensor,\
            neg_weight_tensor,\
            neg_token_embedding
        torch.cuda.empty_cache()
    
    prompt_embeds           = torch.cat(embeds, dim=1).to(device)
    negative_prompt_embeds  = torch.cat(neg_embeds, dim=1).to(device)
    
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

@torch.inference_mode()
def get_weighted_text_embeddings_sd3(
    pipe: StableDiffusion3Pipeline
    , prompt : str      = ""
    , neg_prompt: str   = ""
    , pad_last_block    = True
    , use_t5_encoder    = True
    , device: str       = None
):
    """
    This function can process long prompt with weights, no length limitation 
    for Stable Diffusion 3
    
    Args:
        pipe (StableDiffusionPipeline)
        prompt (str)
        neg_prompt (str)
    Returns:
        sd3_prompt_embeds (torch.Tensor)
        sd3_neg_prompt_embeds (torch.Tensor)
        pooled_prompt_embeds (torch.Tensor)
        negative_pooled_prompt_embeds (torch.Tensor)
    """
    import math
    eos = pipe.tokenizer.eos_token_id

    device = device if device else pipe.device

    # tokenizer 1
    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, prompt
    )

    neg_prompt_tokens, neg_prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, neg_prompt
    )
    
    # tokenizer 2
    prompt_tokens_2, prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, prompt
    )

    neg_prompt_tokens_2, neg_prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, neg_prompt
    )
    
    # tokenizer 3
    prompt_tokens_3, prompt_weights_3 = get_prompts_tokens_with_weights_t5(
        pipe.tokenizer_3, prompt
    )

    neg_prompt_tokens_3, neg_prompt_weights_3 = get_prompts_tokens_with_weights_t5(
        pipe.tokenizer_3, neg_prompt
    )
    
    # padding the shorter one
    prompt_token_len        = len(prompt_tokens)
    neg_prompt_token_len    = len(neg_prompt_tokens)
    
    if prompt_token_len > neg_prompt_token_len:
        # padding the neg_prompt with eos token
        neg_prompt_tokens   = (
            neg_prompt_tokens  + 
            [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        neg_prompt_weights  = (
            neg_prompt_weights + 
            [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    else:
        # padding the prompt
        prompt_tokens       = (
            prompt_tokens  
            + [eos] * abs(neg_prompt_token_len - prompt_token_len)
        )
        prompt_weights      = (
            prompt_weights 
            + [1.0] * abs(neg_prompt_token_len - prompt_token_len)
        )
    
    # padding the shorter one for token set 2
    prompt_token_len_2        = len(prompt_tokens_2)
    neg_prompt_token_len_2    = len(neg_prompt_tokens_2)
    
    if prompt_token_len_2 > neg_prompt_token_len_2:
        # padding the neg_prompt with eos token
        neg_prompt_tokens_2   = (
            neg_prompt_tokens_2  + 
            [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        neg_prompt_weights_2  = (
            neg_prompt_weights_2 + 
            [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    else:
        # padding the prompt
        prompt_tokens_2       = (
            prompt_tokens_2  
            + [eos] * abs(neg_prompt_token_len_2 - prompt_token_len_2)
        )
        prompt_weights_2      = (
            prompt_weights_2 
            + [1.0] * abs(neg_prompt_token_len_2 - prompt_token_len_2)
        )
    
    embeds = []
    neg_embeds = []
    
    prompt_token_groups, prompt_weight_groups = group_tokens_and_weights(
        prompt_tokens.copy()
        , prompt_weights.copy()
        , pad_last_block = pad_last_block
    )
    
    neg_prompt_token_groups, neg_prompt_weight_groups = group_tokens_and_weights(
        neg_prompt_tokens.copy()
        , neg_prompt_weights.copy()
        , pad_last_block = pad_last_block
    )
    
    prompt_token_groups_2, prompt_weight_groups_2 = group_tokens_and_weights(
        prompt_tokens_2.copy()
        , prompt_weights_2.copy()
        , pad_last_block = pad_last_block
    )
    
    neg_prompt_token_groups_2, neg_prompt_weight_groups_2 = group_tokens_and_weights(
        neg_prompt_tokens_2.copy()
        , neg_prompt_weights_2.copy()
        , pad_last_block = pad_last_block
    )
        
    # get prompt embeddings one by one is not working. 
    for i in range(len(prompt_token_groups)):
        # get positive prompt embeddings with weights
        token_tensor = torch.tensor(
            [prompt_token_groups[i]]
            , dtype = torch.long, device = device
        )
        weight_tensor = torch.tensor(
            prompt_weight_groups[i]
            , dtype     = torch.float16
            , device    = device
        )
        
        token_tensor_2 = torch.tensor(
            [prompt_token_groups_2[i]]
            , dtype = torch.long, device = device
        )
        
        # use first text encoder
        prompt_embeds_1 = pipe.text_encoder(
            token_tensor.to(device)
            , output_hidden_states=True
        )
        prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[-2]
        pooled_prompt_embeds_1 = prompt_embeds_1[0]

        # use second text encoder
        prompt_embeds_2 = pipe.text_encoder_2(
            token_tensor_2.to(device)
            , output_hidden_states=True
        )
        prompt_embeds_2_hidden_states = prompt_embeds_2.hidden_states[-2]
        pooled_prompt_embeds_2 = prompt_embeds_2[0]

        prompt_embeds_list = [prompt_embeds_1_hidden_states, prompt_embeds_2_hidden_states]
        token_embedding = torch.cat(prompt_embeds_list, dim=-1).squeeze(0).to(device)
        
        for j in range(len(weight_tensor)):
            if weight_tensor[j] != 1.0:
                #ow = weight_tensor[j] - 1
                
                # optional process
                # To map number of (0,1) to (-1,1)
                # tanh_weight = (math.exp(ow)/(math.exp(ow) + 1) - 0.5) * 2
                # weight = 1 + tanh_weight
                
                # add weight method 1:
                # token_embedding[j] = token_embedding[j] * weight
                # token_embedding[j] = (
                #     token_embedding[-1] + (token_embedding[j] - token_embedding[-1]) * weight
                # )
                
                # add weight method 2:
                # token_embedding[j] = (
                #     token_embedding[-1] + (token_embedding[j] - token_embedding[-1]) * weight_tensor[j]
                # )
                
                # add weight method 3:
                token_embedding[j] = token_embedding[j] * weight_tensor[j]

        token_embedding = token_embedding.unsqueeze(0)
        
        # Free VRAM
        embeds.append(token_embedding.cpu())
        token_tensor.cpu()
        weight_tensor.cpu()
        token_tensor_2.cpu()
        prompt_embeds_1_hidden_states.cpu()
        prompt_embeds_2_hidden_states.cpu()
        del token_tensor,\
            weight_tensor, \
            token_tensor_2, \
            prompt_embeds_1, \
            prompt_embeds_2, \
            prompt_embeds_1_hidden_states, \
            prompt_embeds_2_hidden_states, \
            token_embedding
        torch.cuda.empty_cache()

        # get negative prompt embeddings with weights
        neg_token_tensor = torch.tensor(
            [neg_prompt_token_groups[i]]
            , dtype = torch.long, device = device
        )
        neg_token_tensor_2 = torch.tensor(
            [neg_prompt_token_groups_2[i]]
            , dtype = torch.long, device = device
        )
        neg_weight_tensor = torch.tensor(
            neg_prompt_weight_groups[i]
            , dtype     = torch.float16
            , device    = device
        )
        
        # use first text encoder
        neg_prompt_embeds_1 = pipe.text_encoder(
            neg_token_tensor.to(device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_1_hidden_states = neg_prompt_embeds_1.hidden_states[-2]
        negative_pooled_prompt_embeds_1 = neg_prompt_embeds_1[0]

        # use second text encoder
        neg_prompt_embeds_2 = pipe.text_encoder_2(
            neg_token_tensor_2.to(device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2.hidden_states[-2]
        negative_pooled_prompt_embeds_2 = neg_prompt_embeds_2[0]

        neg_prompt_embeds_list = [neg_prompt_embeds_1_hidden_states, neg_prompt_embeds_2_hidden_states]
        neg_token_embedding = torch.cat(neg_prompt_embeds_list, dim=-1).squeeze(0).to(device)

        for z in range(len(neg_weight_tensor)):
            if neg_weight_tensor[z] != 1.0:
                
                # ow = neg_weight_tensor[z] - 1
                # neg_weight = 1 + (math.exp(ow)/(math.exp(ow) + 1) - 0.5) * 2
                
                # add weight method 1:
                # neg_token_embedding[z] = neg_token_embedding[z] * neg_weight
                # neg_token_embedding[z] = (
                #     neg_token_embedding[-1] + (neg_token_embedding[z] - neg_token_embedding[-1]) * neg_weight
                # )
                
                # add weight method 2:
                # neg_token_embedding[z] = (
                #     neg_token_embedding[-1] + (neg_token_embedding[z] - neg_token_embedding[-1]) * neg_weight_tensor[z]
                # )
                
                # add weight method 3:
                neg_token_embedding[z] = neg_token_embedding[z] * neg_weight_tensor[z]

        neg_token_embedding = neg_token_embedding.unsqueeze(0)

        # Free VRAM
        neg_embeds.append(neg_token_embedding.cpu())
        neg_token_tensor.cpu()
        neg_weight_tensor.cpu()
        neg_token_tensor_2.cpu()
        neg_prompt_embeds_1_hidden_states.cpu()
        neg_prompt_embeds_2_hidden_states.cpu()
        del neg_token_tensor, \
            neg_weight_tensor, \
            neg_token_tensor_2, \
            neg_prompt_embeds_1, \
            neg_prompt_embeds_2,\
            neg_prompt_embeds_1_hidden_states, \
            neg_prompt_embeds_2_hidden_states, \
            neg_token_embedding
        torch.cuda.empty_cache()

    prompt_embeds = torch.cat([e.to(device) for e in embeds], dim=1)
    
    # Free RAM as needed
    del embeds
    
    negative_prompt_embeds = torch.cat([e.to(device) for e in neg_embeds], dim=1)
    del neg_embeds

    pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)
    negative_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds_1, negative_pooled_prompt_embeds_2], dim=-1)

    if use_t5_encoder and pipe.text_encoder_3:
        # ----------------- generate positive t5 embeddings --------------------
        prompt_tokens_3 = torch.tensor([prompt_tokens_3], dtype=torch.long)

        t5_prompt_embeds    = pipe.text_encoder_3(prompt_tokens_3.to(device))[0].squeeze(0)
        t5_prompt_embeds    = t5_prompt_embeds.to(device=device)

        # add weight to t5 prompt
        for z in range(len(prompt_weights_3)):
            if prompt_weights_3[z] != 1.0:
                t5_prompt_embeds[z] = t5_prompt_embeds[z] * prompt_weights_3[z]
        t5_prompt_embeds = t5_prompt_embeds.unsqueeze(0).cpu()
    else:
        t5_prompt_embeds    = torch.zeros(1, 4096, dtype=prompt_embeds.dtype).unsqueeze(0).cpu()

    # merge with the clip embedding 1 and clip embedding 2
    clip_prompt_embeds = torch.nn.functional.pad(
        prompt_embeds.cpu(), (0, t5_prompt_embeds.shape[-1] - prompt_embeds.shape[-1])
    )
    sd3_prompt_embeds = torch.cat([clip_prompt_embeds.to(device), t5_prompt_embeds.to(device)], dim=-2)

    # Free VRAM
    clip_prompt_embeds.cpu()
    t5_prompt_embeds.cpu()
    del clip_prompt_embeds, \
        t5_prompt_embeds
    torch.cuda.empty_cache()

    if use_t5_encoder and pipe.text_encoder_3:  
        # ---------------------- get neg t5 embeddings -------------------------
        neg_prompt_tokens_3 = torch.tensor([neg_prompt_tokens_3], dtype=torch.long)

        t5_neg_prompt_embeds    = pipe.text_encoder_3(neg_prompt_tokens_3.to(device))[0].squeeze(0)
        t5_neg_prompt_embeds    = t5_neg_prompt_embeds.to(device=device)

        # add weight to neg t5 embeddings
        for z in range(len(neg_prompt_weights_3)):
            if neg_prompt_weights_3[z] != 1.0:
                t5_neg_prompt_embeds[z] = t5_neg_prompt_embeds[z] * neg_prompt_weights_3[z]
        t5_neg_prompt_embeds = t5_neg_prompt_embeds.unsqueeze(0).cpu()
    else:
        t5_neg_prompt_embeds    = torch.zeros(1, 4096, dtype=prompt_embeds.dtype).unsqueeze(0).cpu()

    clip_neg_prompt_embeds = torch.nn.functional.pad(
        negative_prompt_embeds.cpu(), (0, t5_neg_prompt_embeds.shape[-1] - negative_prompt_embeds.shape[-1])
    )
    sd3_neg_prompt_embeds = torch.cat([clip_neg_prompt_embeds.to(device), t5_neg_prompt_embeds.to(device)], dim=-2)

    # Free VRAM
    clip_neg_prompt_embeds.cpu()
    t5_neg_prompt_embeds.cpu()
    del clip_neg_prompt_embeds, \
        t5_neg_prompt_embeds
    torch.cuda.empty_cache()

    # padding 
    import torch.nn.functional as F
    size_diff = sd3_neg_prompt_embeds.size(1) - sd3_prompt_embeds.size(1)
    # Calculate padding. Format for pad is (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
    # Since we are padding along the second dimension (axis=1), we need (0, 0, padding_top, padding_bottom, 0, 0)
    # Here padding_top will be 0 and padding_bottom will be size_diff

    # Check if padding is needed
    if size_diff > 0:
        padding = (0, 0, 0, abs(size_diff), 0, 0)
        sd3_prompt_embeds = F.pad(sd3_prompt_embeds, padding)
    elif size_diff < 0:
        padding = (0, 0, 0, abs(size_diff), 0, 0)
        sd3_neg_prompt_embeds = F.pad(sd3_neg_prompt_embeds, padding)

    return sd3_prompt_embeds, sd3_neg_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

@torch.inference_mode()
def get_weighted_text_embeddings_flux1(
    pipe: FluxPipeline
    , prompt : str      = ""
    , prompt2: str      = None
    , device            = None
):
    """
    This function can process long prompt with weights for flux1 model
    
    Args:

    Returns:

    """
    prompt2 = prompt if prompt2 is None else prompt2
    if device is None:
        device = pipe.device
    
    # tokenizer 1 - openai/clip-vit-large-patch14
    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, prompt
    )
    
    # tokenizer 2 - google/t5-v1_1-xxl
    prompt_tokens_2, prompt_weights_2 = get_prompts_tokens_with_weights_t5(
        pipe.tokenizer_2, prompt2
    )
    
    prompt_token_groups, prompt_weight_groups = group_tokens_and_weights(
        prompt_tokens.copy()
        , prompt_weights.copy()
        , pad_last_block = True
    )
        
    # # get positive prompt embeddings, flux1 use only text_encoder 1 pooled embeddings
    # token_tensor = torch.tensor(
    #     [prompt_token_groups[0]]
    #     , dtype = torch.long, device = device
    # )
    # # use first text encoder
    # prompt_embeds_1 = pipe.text_encoder(
    #     token_tensor.to(device)
    #     , output_hidden_states  = False
    # )
    # pooled_prompt_embeds_1  = prompt_embeds_1.pooler_output
    # prompt_embeds           = pooled_prompt_embeds_1.to(dtype = pipe.text_encoder.dtype, device = device)
    
    # use avg pooling embeddings
    pool_embeds_list = []
    for token_group in prompt_token_groups:
        token_tensor = torch.tensor(
            [token_group]
            , dtype = torch.long
            , device = device
        )
        prompt_embeds_1 = pipe.text_encoder(
            token_tensor.to(device)
            , output_hidden_states  = False
        )
        pooled_prompt_embeds = prompt_embeds_1.pooler_output.squeeze(0)
        pool_embeds_list.append(pooled_prompt_embeds)
        
    prompt_embeds = torch.stack(pool_embeds_list,dim=0)
    
    # get the avg pool
    prompt_embeds = prompt_embeds.mean(dim=0, keepdim=True)
    # prompt_embeds = prompt_embeds.unsqueeze(0)
    prompt_embeds = prompt_embeds.to(dtype = pipe.text_encoder.dtype, device = device)
            
    # generate positive t5 embeddings 
    prompt_tokens_2 = torch.tensor([prompt_tokens_2],dtype=torch.long)
    
    t5_prompt_embeds    = pipe.text_encoder_2(prompt_tokens_2.to(device))[0].squeeze(0)
    t5_prompt_embeds    = t5_prompt_embeds.to(device=device)
    
    # add weight to t5 prompt
    for z in range(len(prompt_weights_2)):
        if prompt_weights_2[z] != 1.0:
            t5_prompt_embeds[z] = t5_prompt_embeds[z] * prompt_weights_2[z]
    t5_prompt_embeds = t5_prompt_embeds.unsqueeze(0)
    
    t5_prompt_embeds = t5_prompt_embeds.to(dtype = pipe.text_encoder_2.dtype, device = device)
    
    return t5_prompt_embeds,prompt_embeds
