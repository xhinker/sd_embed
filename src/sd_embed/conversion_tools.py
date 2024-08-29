'''
This file aims to convert models from Civitai to diffusers format
'''
import os
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from diffusers import AutoencoderKL, FluxTransformer2DModel

class Flux1Convertor:
    '''
    The class holds tools for Flux related conversion
    '''
    def __init__(self, dtype = "bf16") -> None:
        self.dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
    
    def swap_scale_shift(self, weight):
        '''
        in SD3 original implementation of AdaLayerNormContinuous, it split linear projection output into shift, scale;
        # while in diffusers it split into scale, shift. Here we swap the linear projection weights in order to be able to use diffusers implementation
        '''
        shift, scale = weight.chunk(2, dim=0)
        new_weight = torch.cat([scale, shift], dim=0)
        return new_weight

    def convert_flux_transformer_checkpoint_to_diffusers(
        self
        , original_state_dict
        , num_layers
        , num_single_layers
        , inner_dim
        , mlp_ratio = 4.0
    ):
        converted_state_dict = {}

        ## time_text_embed.timestep_embedder <-  time_in
        converted_state_dict["time_text_embed.timestep_embedder.linear_1.weight"] = original_state_dict.pop(
            "time_in.in_layer.weight"
        )
        converted_state_dict["time_text_embed.timestep_embedder.linear_1.bias"] = original_state_dict.pop(
            "time_in.in_layer.bias"
        )
        converted_state_dict["time_text_embed.timestep_embedder.linear_2.weight"] = original_state_dict.pop(
            "time_in.out_layer.weight"
        )
        converted_state_dict["time_text_embed.timestep_embedder.linear_2.bias"] = original_state_dict.pop(
            "time_in.out_layer.bias"
        )

        ## time_text_embed.text_embedder <- vector_in
        converted_state_dict["time_text_embed.text_embedder.linear_1.weight"] = original_state_dict.pop(
            "vector_in.in_layer.weight"
        )
        converted_state_dict["time_text_embed.text_embedder.linear_1.bias"] = original_state_dict.pop(
            "vector_in.in_layer.bias"
        )
        converted_state_dict["time_text_embed.text_embedder.linear_2.weight"] = original_state_dict.pop(
            "vector_in.out_layer.weight"
        )
        converted_state_dict["time_text_embed.text_embedder.linear_2.bias"] = original_state_dict.pop(
            "vector_in.out_layer.bias"
        )

        # guidance
        has_guidance = any("guidance" in k for k in original_state_dict)
        if has_guidance:
            converted_state_dict["time_text_embed.guidance_embedder.linear_1.weight"] = original_state_dict.pop(
                "guidance_in.in_layer.weight"
            )
            converted_state_dict["time_text_embed.guidance_embedder.linear_1.bias"] = original_state_dict.pop(
                "guidance_in.in_layer.bias"
            )
            converted_state_dict["time_text_embed.guidance_embedder.linear_2.weight"] = original_state_dict.pop(
                "guidance_in.out_layer.weight"
            )
            converted_state_dict["time_text_embed.guidance_embedder.linear_2.bias"] = original_state_dict.pop(
                "guidance_in.out_layer.bias"
            )

        # context_embedder
        converted_state_dict["context_embedder.weight"] = original_state_dict.pop("txt_in.weight")
        converted_state_dict["context_embedder.bias"] = original_state_dict.pop("txt_in.bias")

        # x_embedder
        converted_state_dict["x_embedder.weight"] = original_state_dict.pop("img_in.weight")
        converted_state_dict["x_embedder.bias"] = original_state_dict.pop("img_in.bias")

        # double transformer blocks
        for i in range(num_layers):
            block_prefix = f"transformer_blocks.{i}."
            # norms.
            ## norm1
            converted_state_dict[f"{block_prefix}norm1.linear.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.img_mod.lin.weight"
            )
            converted_state_dict[f"{block_prefix}norm1.linear.bias"] = original_state_dict.pop(
                f"double_blocks.{i}.img_mod.lin.bias"
            )
            ## norm1_context
            converted_state_dict[f"{block_prefix}norm1_context.linear.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.txt_mod.lin.weight"
            )
            converted_state_dict[f"{block_prefix}norm1_context.linear.bias"] = original_state_dict.pop(
                f"double_blocks.{i}.txt_mod.lin.bias"
            )
            # Q, K, V
            sample_q, sample_k, sample_v = torch.chunk(
                original_state_dict.pop(f"double_blocks.{i}.img_attn.qkv.weight"), 3, dim=0
            )
            context_q, context_k, context_v = torch.chunk(
                original_state_dict.pop(f"double_blocks.{i}.txt_attn.qkv.weight"), 3, dim=0
            )
            sample_q_bias, sample_k_bias, sample_v_bias = torch.chunk(
                original_state_dict.pop(f"double_blocks.{i}.img_attn.qkv.bias"), 3, dim=0
            )
            context_q_bias, context_k_bias, context_v_bias = torch.chunk(
                original_state_dict.pop(f"double_blocks.{i}.txt_attn.qkv.bias"), 3, dim=0
            )
            converted_state_dict[f"{block_prefix}attn.to_q.weight"] = torch.cat([sample_q])
            converted_state_dict[f"{block_prefix}attn.to_q.bias"] = torch.cat([sample_q_bias])
            converted_state_dict[f"{block_prefix}attn.to_k.weight"] = torch.cat([sample_k])
            converted_state_dict[f"{block_prefix}attn.to_k.bias"] = torch.cat([sample_k_bias])
            converted_state_dict[f"{block_prefix}attn.to_v.weight"] = torch.cat([sample_v])
            converted_state_dict[f"{block_prefix}attn.to_v.bias"] = torch.cat([sample_v_bias])
            converted_state_dict[f"{block_prefix}attn.add_q_proj.weight"] = torch.cat([context_q])
            converted_state_dict[f"{block_prefix}attn.add_q_proj.bias"] = torch.cat([context_q_bias])
            converted_state_dict[f"{block_prefix}attn.add_k_proj.weight"] = torch.cat([context_k])
            converted_state_dict[f"{block_prefix}attn.add_k_proj.bias"] = torch.cat([context_k_bias])
            converted_state_dict[f"{block_prefix}attn.add_v_proj.weight"] = torch.cat([context_v])
            converted_state_dict[f"{block_prefix}attn.add_v_proj.bias"] = torch.cat([context_v_bias])
            # qk_norm
            converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.img_attn.norm.query_norm.scale"
            )
            converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.img_attn.norm.key_norm.scale"
            )
            converted_state_dict[f"{block_prefix}attn.norm_added_q.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.txt_attn.norm.query_norm.scale"
            )
            converted_state_dict[f"{block_prefix}attn.norm_added_k.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.txt_attn.norm.key_norm.scale"
            )
            # ff img_mlp
            converted_state_dict[f"{block_prefix}ff.net.0.proj.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.img_mlp.0.weight"
            )
            converted_state_dict[f"{block_prefix}ff.net.0.proj.bias"] = original_state_dict.pop(
                f"double_blocks.{i}.img_mlp.0.bias"
            )
            converted_state_dict[f"{block_prefix}ff.net.2.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.img_mlp.2.weight"
            )
            converted_state_dict[f"{block_prefix}ff.net.2.bias"] = original_state_dict.pop(
                f"double_blocks.{i}.img_mlp.2.bias"
            )
            converted_state_dict[f"{block_prefix}ff_context.net.0.proj.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.txt_mlp.0.weight"
            )
            converted_state_dict[f"{block_prefix}ff_context.net.0.proj.bias"] = original_state_dict.pop(
                f"double_blocks.{i}.txt_mlp.0.bias"
            )
            converted_state_dict[f"{block_prefix}ff_context.net.2.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.txt_mlp.2.weight"
            )
            converted_state_dict[f"{block_prefix}ff_context.net.2.bias"] = original_state_dict.pop(
                f"double_blocks.{i}.txt_mlp.2.bias"
            )
            # output projections.
            converted_state_dict[f"{block_prefix}attn.to_out.0.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.img_attn.proj.weight"
            )
            converted_state_dict[f"{block_prefix}attn.to_out.0.bias"] = original_state_dict.pop(
                f"double_blocks.{i}.img_attn.proj.bias"
            )
            converted_state_dict[f"{block_prefix}attn.to_add_out.weight"] = original_state_dict.pop(
                f"double_blocks.{i}.txt_attn.proj.weight"
            )
            converted_state_dict[f"{block_prefix}attn.to_add_out.bias"] = original_state_dict.pop(
                f"double_blocks.{i}.txt_attn.proj.bias"
            )

        # single transformer blocks
        for i in range(num_single_layers):
            block_prefix = f"single_transformer_blocks.{i}."
            # norm.linear  <- single_blocks.0.modulation.lin
            converted_state_dict[f"{block_prefix}norm.linear.weight"] = original_state_dict.pop(
                f"single_blocks.{i}.modulation.lin.weight"
            )
            converted_state_dict[f"{block_prefix}norm.linear.bias"] = original_state_dict.pop(
                f"single_blocks.{i}.modulation.lin.bias"
            )
            # Q, K, V, mlp
            mlp_hidden_dim = int(inner_dim * mlp_ratio)
            split_size = (inner_dim, inner_dim, inner_dim, mlp_hidden_dim)
            q, k, v, mlp = torch.split(original_state_dict.pop(f"single_blocks.{i}.linear1.weight"), split_size, dim=0)
            q_bias, k_bias, v_bias, mlp_bias = torch.split(
                original_state_dict.pop(f"single_blocks.{i}.linear1.bias"), split_size, dim=0
            )
            converted_state_dict[f"{block_prefix}attn.to_q.weight"] = torch.cat([q])
            converted_state_dict[f"{block_prefix}attn.to_q.bias"] = torch.cat([q_bias])
            converted_state_dict[f"{block_prefix}attn.to_k.weight"] = torch.cat([k])
            converted_state_dict[f"{block_prefix}attn.to_k.bias"] = torch.cat([k_bias])
            converted_state_dict[f"{block_prefix}attn.to_v.weight"] = torch.cat([v])
            converted_state_dict[f"{block_prefix}attn.to_v.bias"] = torch.cat([v_bias])
            converted_state_dict[f"{block_prefix}proj_mlp.weight"] = torch.cat([mlp])
            converted_state_dict[f"{block_prefix}proj_mlp.bias"] = torch.cat([mlp_bias])
            # qk norm
            converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(
                f"single_blocks.{i}.norm.query_norm.scale"
            )
            converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(
                f"single_blocks.{i}.norm.key_norm.scale"
            )
            # output projections.
            converted_state_dict[f"{block_prefix}proj_out.weight"] = original_state_dict.pop(
                f"single_blocks.{i}.linear2.weight"
            )
            converted_state_dict[f"{block_prefix}proj_out.bias"] = original_state_dict.pop(
                f"single_blocks.{i}.linear2.bias"
            )

        converted_state_dict["proj_out.weight"] = original_state_dict.pop("final_layer.linear.weight")
        converted_state_dict["proj_out.bias"] = original_state_dict.pop("final_layer.linear.bias")
        converted_state_dict["norm_out.linear.weight"] = self.swap_scale_shift(
            original_state_dict.pop("final_layer.adaLN_modulation.1.weight")
        )
        converted_state_dict["norm_out.linear.bias"] = self.swap_scale_shift(
            original_state_dict.pop("final_layer.adaLN_modulation.1.bias")
        )

        return converted_state_dict

    def convert_comfyui_transformer_checkpoint_to_diffusers(
        self
        , original_state_dict
        , num_layers
        , num_single_layers
        , inner_dim, mlp_ratio = 4.0
    ):
        converted_state_dict = {}
        
        prefix = "model.diffusion_model."

        ## time_text_embed.timestep_embedder <-  time_in
        converted_state_dict["time_text_embed.timestep_embedder.linear_1.weight"] = original_state_dict.pop(
            f"{prefix}time_in.in_layer.weight"
        )
        converted_state_dict["time_text_embed.timestep_embedder.linear_1.bias"] = original_state_dict.pop(
            f"{prefix}time_in.in_layer.bias"
        )
        converted_state_dict["time_text_embed.timestep_embedder.linear_2.weight"] = original_state_dict.pop(
            f"{prefix}time_in.out_layer.weight"
        )
        converted_state_dict["time_text_embed.timestep_embedder.linear_2.bias"] = original_state_dict.pop(
            f"{prefix}time_in.out_layer.bias"
        )

        ## time_text_embed.text_embedder <- vector_in
        converted_state_dict["time_text_embed.text_embedder.linear_1.weight"] = original_state_dict.pop(
            f"{prefix}vector_in.in_layer.weight"
        )
        converted_state_dict["time_text_embed.text_embedder.linear_1.bias"] = original_state_dict.pop(
            f"{prefix}vector_in.in_layer.bias"
        )
        converted_state_dict["time_text_embed.text_embedder.linear_2.weight"] = original_state_dict.pop(
            f"{prefix}vector_in.out_layer.weight"
        )
        converted_state_dict["time_text_embed.text_embedder.linear_2.bias"] = original_state_dict.pop(
            f"{prefix}vector_in.out_layer.bias"
        )

        # guidance
        has_guidance = any("guidance" in k for k in original_state_dict)
        if has_guidance:
            converted_state_dict["time_text_embed.guidance_embedder.linear_1.weight"] = original_state_dict.pop(
                f"{prefix}guidance_in.in_layer.weight"
            )
            converted_state_dict["time_text_embed.guidance_embedder.linear_1.bias"] = original_state_dict.pop(
                f"{prefix}guidance_in.in_layer.bias"
            )
            converted_state_dict["time_text_embed.guidance_embedder.linear_2.weight"] = original_state_dict.pop(
                f"{prefix}guidance_in.out_layer.weight"
            )
            converted_state_dict["time_text_embed.guidance_embedder.linear_2.bias"] = original_state_dict.pop(
                f"{prefix}guidance_in.out_layer.bias"
            )

        # context_embedder
        converted_state_dict["context_embedder.weight"] = original_state_dict.pop(f"{prefix}txt_in.weight")
        converted_state_dict["context_embedder.bias"] = original_state_dict.pop(f"{prefix}txt_in.bias")

        # x_embedder
        converted_state_dict["x_embedder.weight"] = original_state_dict.pop(f"{prefix}img_in.weight")
        converted_state_dict["x_embedder.bias"] = original_state_dict.pop(f"{prefix}img_in.bias")

        # double transformer blocks
        for i in range(num_layers):
            block_prefix = f"transformer_blocks.{i}."
            # norms.
            ## norm1
            converted_state_dict[f"{block_prefix}norm1.linear.weight"] = original_state_dict.pop(
                f"{prefix}double_blocks.{i}.img_mod.lin.weight"
            )
            converted_state_dict[f"{block_prefix}norm1.linear.bias"] = original_state_dict.pop(
                f"{prefix}double_blocks.{i}.img_mod.lin.bias"
            )
            ## norm1_context
            converted_state_dict[f"{block_prefix}norm1_context.linear.weight"] = original_state_dict.pop(
                f"{prefix}double_blocks.{i}.txt_mod.lin.weight"
            )
            converted_state_dict[f"{block_prefix}norm1_context.linear.bias"] = original_state_dict.pop(
                f"{prefix}double_blocks.{i}.txt_mod.lin.bias"
            )
            # Q, K, V
            sample_q, sample_k, sample_v = torch.chunk(
                original_state_dict.pop(f"{prefix}double_blocks.{i}.img_attn.qkv.weight"), 3, dim=0
            )
            context_q, context_k, context_v = torch.chunk(
                original_state_dict.pop(f"{prefix}double_blocks.{i}.txt_attn.qkv.weight"), 3, dim=0
            )
            sample_q_bias, sample_k_bias, sample_v_bias = torch.chunk(
                original_state_dict.pop(f"{prefix}double_blocks.{i}.img_attn.qkv.bias"), 3, dim=0
            )
            context_q_bias, context_k_bias, context_v_bias = torch.chunk(
                original_state_dict.pop(f"{prefix}double_blocks.{i}.txt_attn.qkv.bias"), 3, dim=0
            )
            converted_state_dict[f"{block_prefix}attn.to_q.weight"] = torch.cat([sample_q])
            converted_state_dict[f"{block_prefix}attn.to_q.bias"] = torch.cat([sample_q_bias])
            converted_state_dict[f"{block_prefix}attn.to_k.weight"] = torch.cat([sample_k])
            converted_state_dict[f"{block_prefix}attn.to_k.bias"] = torch.cat([sample_k_bias])
            converted_state_dict[f"{block_prefix}attn.to_v.weight"] = torch.cat([sample_v])
            converted_state_dict[f"{block_prefix}attn.to_v.bias"] = torch.cat([sample_v_bias])
            converted_state_dict[f"{block_prefix}attn.add_q_proj.weight"] = torch.cat([context_q])
            converted_state_dict[f"{block_prefix}attn.add_q_proj.bias"] = torch.cat([context_q_bias])
            converted_state_dict[f"{block_prefix}attn.add_k_proj.weight"] = torch.cat([context_k])
            converted_state_dict[f"{block_prefix}attn.add_k_proj.bias"] = torch.cat([context_k_bias])
            converted_state_dict[f"{block_prefix}attn.add_v_proj.weight"] = torch.cat([context_v])
            converted_state_dict[f"{block_prefix}attn.add_v_proj.bias"] = torch.cat([context_v_bias])
            # qk_norm
            converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(
                f"{prefix}double_blocks.{i}.img_attn.norm.query_norm.scale"
            )
            converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(
                f"{prefix}double_blocks.{i}.img_attn.norm.key_norm.scale"
            )
            converted_state_dict[f"{block_prefix}attn.norm_added_q.weight"] = original_state_dict.pop(
                f"{prefix}double_blocks.{i}.txt_attn.norm.query_norm.scale"
            )
            converted_state_dict[f"{block_prefix}attn.norm_added_k.weight"] = original_state_dict.pop(
                f"{prefix}double_blocks.{i}.txt_attn.norm.key_norm.scale"
            )
            # ff img_mlp
            converted_state_dict[f"{block_prefix}ff.net.0.proj.weight"] = original_state_dict.pop(
                f"{prefix}double_blocks.{i}.img_mlp.0.weight"
            )
            converted_state_dict[f"{block_prefix}ff.net.0.proj.bias"] = original_state_dict.pop(
                f"{prefix}double_blocks.{i}.img_mlp.0.bias"
            )
            converted_state_dict[f"{block_prefix}ff.net.2.weight"] = original_state_dict.pop(
                f"{prefix}double_blocks.{i}.img_mlp.2.weight"
            )
            converted_state_dict[f"{block_prefix}ff.net.2.bias"] = original_state_dict.pop(
                f"{prefix}double_blocks.{i}.img_mlp.2.bias"
            )
            converted_state_dict[f"{block_prefix}ff_context.net.0.proj.weight"] = original_state_dict.pop(
                f"{prefix}double_blocks.{i}.txt_mlp.0.weight"
            )
            converted_state_dict[f"{block_prefix}ff_context.net.0.proj.bias"] = original_state_dict.pop(
                f"{prefix}double_blocks.{i}.txt_mlp.0.bias"
            )
            converted_state_dict[f"{block_prefix}ff_context.net.2.weight"] = original_state_dict.pop(
                f"{prefix}double_blocks.{i}.txt_mlp.2.weight"
            )
            converted_state_dict[f"{block_prefix}ff_context.net.2.bias"] = original_state_dict.pop(
                f"{prefix}double_blocks.{i}.txt_mlp.2.bias"
            )
            # output projections.
            converted_state_dict[f"{block_prefix}attn.to_out.0.weight"] = original_state_dict.pop(
                f"{prefix}double_blocks.{i}.img_attn.proj.weight"
            )
            converted_state_dict[f"{block_prefix}attn.to_out.0.bias"] = original_state_dict.pop(
                f"{prefix}double_blocks.{i}.img_attn.proj.bias"
            )
            converted_state_dict[f"{block_prefix}attn.to_add_out.weight"] = original_state_dict.pop(
                f"{prefix}double_blocks.{i}.txt_attn.proj.weight"
            )
            converted_state_dict[f"{block_prefix}attn.to_add_out.bias"] = original_state_dict.pop(
                f"{prefix}double_blocks.{i}.txt_attn.proj.bias"
            )

        # single transfomer blocks
        for i in range(num_single_layers):
            block_prefix = f"single_transformer_blocks.{i}."
            # norm.linear  <- single_blocks.0.modulation.lin
            converted_state_dict[f"{block_prefix}norm.linear.weight"] = original_state_dict.pop(
                f"{prefix}single_blocks.{i}.modulation.lin.weight"
            )
            converted_state_dict[f"{block_prefix}norm.linear.bias"] = original_state_dict.pop(
                f"{prefix}single_blocks.{i}.modulation.lin.bias"
            )
            # Q, K, V, mlp
            mlp_hidden_dim = int(inner_dim * mlp_ratio)
            split_size = (inner_dim, inner_dim, inner_dim, mlp_hidden_dim)
            q, k, v, mlp = torch.split(original_state_dict.pop(f"{prefix}single_blocks.{i}.linear1.weight"), split_size, dim=0)
            q_bias, k_bias, v_bias, mlp_bias = torch.split(
                original_state_dict.pop(f"{prefix}single_blocks.{i}.linear1.bias"), split_size, dim=0
            )
            converted_state_dict[f"{block_prefix}attn.to_q.weight"] = torch.cat([q])
            converted_state_dict[f"{block_prefix}attn.to_q.bias"] = torch.cat([q_bias])
            converted_state_dict[f"{block_prefix}attn.to_k.weight"] = torch.cat([k])
            converted_state_dict[f"{block_prefix}attn.to_k.bias"] = torch.cat([k_bias])
            converted_state_dict[f"{block_prefix}attn.to_v.weight"] = torch.cat([v])
            converted_state_dict[f"{block_prefix}attn.to_v.bias"] = torch.cat([v_bias])
            converted_state_dict[f"{block_prefix}proj_mlp.weight"] = torch.cat([mlp])
            converted_state_dict[f"{block_prefix}proj_mlp.bias"] = torch.cat([mlp_bias])
            # qk norm
            converted_state_dict[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(
                f"{prefix}single_blocks.{i}.norm.query_norm.scale"
            )
            converted_state_dict[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(
                f"{prefix}single_blocks.{i}.norm.key_norm.scale"
            )
            # output projections.
            converted_state_dict[f"{block_prefix}proj_out.weight"] = original_state_dict.pop(
                f"{prefix}single_blocks.{i}.linear2.weight"
            )
            converted_state_dict[f"{block_prefix}proj_out.bias"] = original_state_dict.pop(
                f"{prefix}single_blocks.{i}.linear2.bias"
            )

        converted_state_dict["proj_out.weight"] = original_state_dict.pop(f"{prefix}final_layer.linear.weight")
        converted_state_dict["proj_out.bias"] = original_state_dict.pop(f"{prefix}final_layer.linear.bias")
        converted_state_dict["norm_out.linear.weight"] = self.swap_scale_shift(
            original_state_dict.pop(f"{prefix}final_layer.adaLN_modulation.1.weight")
        )
        converted_state_dict["norm_out.linear.bias"] = self.swap_scale_shift(
            original_state_dict.pop(f"{prefix}final_layer.adaLN_modulation.1.bias")
        )

        return converted_state_dict
    
    def convert_to_diffuses(
        self
        , input_safetensor_file_path:str
        , output_directory:str = None
    ):
        original_state_dict = load_file(input_safetensor_file_path)
        has_guidance        = any("guidance" in k for k in original_state_dict)
        num_layers          = 19
        num_single_layers   = 38
        inner_dim           = 3072  
        mlp_ratio           = 4.0
        
        # check if model.diffusion_model is in the key
        is_comfyui_safetensors = False
        original_keys = original_state_dict.keys()
        for key in original_keys:
            if "model.diffusion_model." in key:
                is_comfyui_safetensors = True
                break
        
        if is_comfyui_safetensors:
            converted_transformer_state_dict    = self.convert_comfyui_transformer_checkpoint_to_diffusers(
                original_state_dict, num_layers, num_single_layers, inner_dim, mlp_ratio=mlp_ratio
            )
        else:
            converted_transformer_state_dict    = self.convert_flux_transformer_checkpoint_to_diffusers(
                original_state_dict, num_layers, num_single_layers, inner_dim, mlp_ratio=mlp_ratio
            )

        transformer = FluxTransformer2DModel(guidance_embeds=has_guidance)
        transformer.load_state_dict(converted_transformer_state_dict, strict=True)
        
        if output_directory:
            transformer_path = os.path.join(output_directory,transformer)
        else:
            transformer_path = f"{os.path.dirname(input_safetensor_file_path)}/{os.path.basename(input_safetensor_file_path).split('.')[0]}/transformer"
            
        transformer.to(self.dtype).save_pretrained(transformer_path)
        
        return transformer_path