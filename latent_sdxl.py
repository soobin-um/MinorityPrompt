from typing import Any, Optional, Tuple
import os
from safetensors.torch import load_file

import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from diffusers.models.attention_processor import (AttnProcessor2_0,
                                                  LoRAAttnProcessor2_0,
                                                  LoRAXFormersAttnProcessor,
                                                  XFormersAttnProcessor)
from tqdm import tqdm

from torch.optim.adam import Adam
import copy

####### Factory #######
__SOLVER__ = {}

def register_solver(name: str):
    def wrapper(cls):
        if __SOLVER__.get(name, None) is not None:
            raise ValueError(f"Solver {name} already registered.")
        __SOLVER__[name] = cls
        return cls
    return wrapper

def get_solver(name: str, **kwargs):
    if name not in __SOLVER__:
        raise ValueError(f"Solver {name} does not exist.")
    return __SOLVER__[name](**kwargs)

########################

class SDXL():
    def __init__(self, 
                 solver_config: dict,
                 model_key:str="stabilityai/stable-diffusion-xl-base-1.0",
                 dtype=torch.float16,
                 device='cuda'):

        self.device = device
        pipe = StableDiffusionXLPipeline.from_pretrained(model_key, torch_dtype=dtype).to(device)
        self.dtype = dtype

        # avoid overflow in float16
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype).to(device)

        self.tokenizer_1_base = copy.deepcopy(pipe.tokenizer)
        self.tokenizer_2_base = copy.deepcopy(pipe.tokenizer_2)
        self.text_enc_1_base = copy.deepcopy(pipe.text_encoder)
        self.text_enc_2_base = copy.deepcopy(pipe.text_encoder_2)
        self.unet = pipe.unet

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.default_sample_size = self.unet.config.sample_size

        # sampling parameters
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        N_ts = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
        self.skip = N_ts // solver_config.num_sampling

        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
        self.scheduler.alphas_cumprod_default = self.scheduler.alphas_cumprod
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.sample(*args, **kwargs)

    def alpha(self, t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at

    @torch.no_grad()
    def _text_embed(self, prompt, tokenizer, text_enc, clip_skip):
        text_inputs = tokenizer(
            prompt,
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt')
        text_input_ids = text_inputs.input_ids
        # print("prompt: ", prompt)
        # print("text_input_ids: ", text_input_ids)
        # also print string for the associated text tokens
        # print("text tokens: ", tokenizer.convert_ids_to_tokens(text_input_ids[0]))
        prompt_embeds = text_enc(text_input_ids.to(self.device), output_hidden_states=True)

        pool_prompt_embeds = prompt_embeds[0]
        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            # +2 because SDXL always indexes from the penultimate layer.
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
        return prompt_embeds, pool_prompt_embeds

    @torch.no_grad()
    def get_text_embed(self, null_prompt_1, prompt_1, null_prompt_2=None, prompt_2=None, clip_skip=None):
        '''
        At this time, assume that batch_size = 1.
        We should extend the code to batch_size > 1.
        '''        
        # Encode the prompts
        # if prompt_2 is None, set same as prompt_1
        prompt_1 = [prompt_1] if isinstance(prompt_1, str) else prompt_1
        null_prompt_1 = [null_prompt_1] if isinstance(null_prompt_1, str) else null_prompt_1


        prompt_embed_1, pool_prompt_embed = self._text_embed(prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)
        if prompt_2 is None:
            prompt_embed = [prompt_embed_1]
        else:
            # Comment on diffusers' source code:
            # "We are only ALWAYS interested in the pooled output of the final text encoder"
            # i.e. we overwrite the pool_prompt_embed with the new one
            prompt_embed_2, pool_prompt_embed = self._text_embed(prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
            prompt_embed = [prompt_embed_1, prompt_embed_2]
        
        null_embed_1, pool_null_embed = self._text_embed(null_prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)
        if null_prompt_2 is None:
            null_embed = [null_embed_1]
        else:
            null_embed_2, pool_null_embed = self._text_embed(null_prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
            null_embed = [null_embed_1, null_embed_2]

        # concat embeds from two encoders
        null_prompt_embeds = torch.concat(null_embed, dim=-1)
        prompt_embeds = torch.concat(prompt_embed, dim=-1)

        return null_prompt_embeds, prompt_embeds, pool_null_embed, pool_prompt_embed            
    
    def _differentiable_text_embed(self, prompt, tokenizer, text_enc, clip_skip):
        text_inputs = tokenizer(
            prompt,
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt')
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_enc(text_input_ids.to(self.device), output_hidden_states=True)

        pool_prompt_embeds = prompt_embeds[0]
        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            # +2 because SDXL always indexes from the penultimate layer.
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
        return prompt_embeds, pool_prompt_embeds

    def differentiable_get_text_embed(self, null_prompt_1, prompt_1, null_prompt_2=None, prompt_2=None, clip_skip=None, te1_nograd=False, te2_nograd=False):
        '''
        At this time, assume that batch_size = 1.
        We should extend the code to batch_size > 1.
        '''        
        # Encode the prompts
        # if prompt_2 is None, set same as prompt_1
        prompt_1 = [prompt_1] if isinstance(prompt_1, str) else prompt_1
        null_prompt_1 = [null_prompt_1] if isinstance(null_prompt_1, str) else null_prompt_1

        if te1_nograd:
            prompt_embed_1, pool_prompt_embed = self._text_embed(prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)
        else:
            prompt_embed_1, pool_prompt_embed = self._differentiable_text_embed(prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)

        if prompt_2 is None:
            prompt_embed = [prompt_embed_1]
        else:
            # Comment on diffusers' source code:
            # "We are only ALWAYS interested in the pooled output of the final text encoder"
            # i.e. we overwrite the pool_prompt_embed with the new one
            if te2_nograd:
                prompt_embed_2, pool_prompt_embed = self._text_embed(prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
            else:
                prompt_embed_2, pool_prompt_embed = self._differentiable_text_embed(prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
            prompt_embed = [prompt_embed_1, prompt_embed_2]
        
        if te1_nograd:
            null_embed_1, pool_null_embed = self._text_embed(null_prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)
        else:
            null_embed_1, pool_null_embed = self._differentiable_text_embed(null_prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip)

        if null_prompt_2 is None:
            null_embed = [null_embed_1]
        else:
            if te2_nograd:
                null_embed_2, pool_null_embed = self._text_embed(null_prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
            else:
                null_embed_2, pool_null_embed = self._differentiable_text_embed(null_prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip)
            null_embed = [null_embed_1, null_embed_2]

        # concat embeds from two encoders
        null_prompt_embeds = torch.concat(null_embed, dim=-1)
        prompt_embeds = torch.concat(prompt_embed, dim=-1)

        return null_prompt_embeds, prompt_embeds, pool_null_embed, pool_prompt_embed            

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    @torch.no_grad()
    def encode(self, x):
        return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor 

    # @torch.no_grad() 
    def decode(self, zt):
        # make sure the VAE is in float32 mode, as it overflows in float16
        # needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        # if needs_upcasting:
        #     self.upcast_vae()
        #     zt = zt.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        image = self.vae.decode(zt / self.vae.config.scaling_factor).sample.float()
        return image


    def predict_noise(self, zt, t, uc, c, added_cond_kwargs):
        t_in = t.unsqueeze(0)
        if uc is None:
            noise_c = self.unet(zt, t_in, encoder_hidden_states=c,
                                   added_cond_kwargs=added_cond_kwargs)['sample']
            noise_uc = noise_c
        elif c is None:
            noise_uc = self.unet(zt, t_in, encoder_hidden_states=uc,
                                   added_cond_kwargs=added_cond_kwargs)['sample']
            noise_c = noise_uc
        else:
            c_embed = torch.cat([uc, c], dim=0)
            z_in = torch.cat([zt] * 2)
            t_in = torch.cat([t_in] * 2)
            noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed,
                                   added_cond_kwargs=added_cond_kwargs)['sample']
            noise_uc, noise_c = noise_pred.chunk(2)

        return noise_uc, noise_c

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim):
        add_time_ids = list(original_size+crops_coords_top_left+target_size)
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        assert expected_add_embed_dim == passed_add_embed_dim, (
             f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               prompt1 = ["", ""],
               prompt2 = ["", ""],
               cfg_guidance:float=5.0,
               original_size: Optional[Tuple[int, int]]=None,
               crops_coords_top_left: Tuple[int, int]=(0, 0),
               target_size: Optional[Tuple[int, int]]=None,
               negative_original_size: Optional[Tuple[int, int]]=None,
               negative_crops_coords_top_left: Tuple[int, int]=(0, 0),
               negative_target_size: Optional[Tuple[int, int]]=None,
               clip_skip: Optional[int]=None,
               popt_kwargs: Optional[dict]=None,
               etc_kwargs: Optional[dict]=None,
               **kwargs):
        
        self.prompt1 = prompt1
        self.prompt2 = prompt2
        self.cfg_guidance = cfg_guidance
        self.original_size = original_size
        self.crops_coords_top_left = crops_coords_top_left
        self.target_size = target_size
        self.negative_original_size = negative_original_size
        self.negative_crops_coords_top_left = negative_crops_coords_top_left
        self.negative_target_size = negative_target_size
        self.clip_skip = clip_skip

        # 0. Default height and width to unet
        height = self.default_sample_size * self.vae_scale_factor
        width = self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # reset tokenizer and text_encoder
        self.tokenizer_1 = copy.deepcopy(self.tokenizer_1_base)
        self.tokenizer_2 = copy.deepcopy(self.tokenizer_2_base)
        self.text_enc_1 = copy.deepcopy(self.text_enc_1_base)
        self.text_enc_2 = copy.deepcopy(self.text_enc_2_base)

        # embedding
        (null_prompt_embeds,
         prompt_embeds,
         pool_null_embed,
         pool_prompt_embed) = self.get_text_embed(prompt1[0], prompt1[1], prompt2[0], prompt2[1], clip_skip)

        # prepare kwargs for SDXL
        add_text_embeds = pool_prompt_embed
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
            )
        else:
            negative_add_time_ids = add_time_ids
        negative_text_embeds = pool_null_embed 

        if cfg_guidance != 0.0 and cfg_guidance != 1.0:
            # do cfg
            add_text_embeds = torch.cat([negative_text_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        add_cond_kwargs = {
            'text_embeds': add_text_embeds.to(self.device),
            'time_ids': add_time_ids.to(self.device)
        }

        # reverse sampling
        zt = self.reverse_process(null_prompt_embeds, prompt_embeds, cfg_guidance, add_cond_kwargs, target_size, popt_kwargs=popt_kwargs, etc_kwargs=etc_kwargs, **kwargs)

        # decode
        with torch.no_grad():
            img = self.decode(zt)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

    def initialize_latent(self,
                          method: str='random',
                          src_img: Optional[torch.Tensor]=None,
                          add_cond_kwargs: Optional[dict]=None,
                          **kwargs):
        if method == 'ddim':
            assert src_img is not None, "src_img must be provided for inversion"
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('uc'),
                               kwargs.get('c'),
                               kwargs.get('cfg_guidance', 0.0),
                               add_cond_kwargs)
        elif method == 'npi':
            assert src_img is not None, "src_img must be provided for inversion"
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('c'),
                               kwargs.get('c'),
                               1.0,
                               add_cond_kwargs)
        elif method == 'random':
            size = kwargs.get('size', (1, 4, 128, 128))
            z = torch.randn(size).to(self.device)
        else: 
            raise NotImplementedError

        return z.requires_grad_()

    def inversion(self, z0, uc, c, cfg_guidance, add_cond_kwargs):
        # if we use cfg_guidance=0.0 or 1.0 for inversion, add_cond_kwargs must be splitted. 
        if cfg_guidance == 0.0 or cfg_guidance == 1.0:
            add_cond_kwargs['text_embeds'] = add_cond_kwargs['text_embeds'][-1].unsqueeze(0)
            add_cond_kwargs['time_ids'] = add_cond_kwargs['time_ids'][-1].unsqueeze(0)

        zt = z0.clone().to(self.device)
        pbar = tqdm(reversed(self.scheduler.timesteps), desc='DDIM inversion')
        for _, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c  = self.predict_noise(zt, t, uc, c, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at_prev).sqrt() * noise_pred) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1-at).sqrt() * noise_pred

        return zt
    
    def reverse_process(self, *args, **kwargs):
        raise NotImplementedError
    
    @torch.enable_grad()
    def prompt_opt(self, zt, t, step, placeholder_token_ids_enc1, placeholder_token_ids_enc2, null_prompt_embeds, prompt_embeds_base, add_cond_kwargs_base, cfg_guidance, popt_kwargs):
        assert cfg_guidance > 0.0
        placeholder_string = popt_kwargs['placeholder_string']
        assert "_" in placeholder_string and len(placeholder_string.split("_")) == 2
        placeholder_symbol = placeholder_string.split("_")[0]

        decay_rate = popt_kwargs['lr_decay_rate']
        num_opt_tokens = popt_kwargs['num_opt_tokens']
        
        para = list(self.text_enc_1.get_input_embeddings().parameters()) + list(self.text_enc_2.get_input_embeddings().parameters())
        optimizer = Adam(para, lr=popt_kwargs['p_opt_lr'] * (1. - step * decay_rate))

        # keep original embeddings as reference
        orig_embeds_params_enc1 = self.text_enc_1.get_input_embeddings().weight.data.clone()
        orig_embeds_params_enc2 = self.text_enc_2.get_input_embeddings().weight.data.clone()

        prompt1 = self.prompt1.copy()
        prompt2 = self.prompt2.copy()

        # add placeholder tokens only for prompt
        prompt_list_1 = [prompt1[1]]
        if popt_kwargs['placeholder_position'] == 'end':
            prompt_list_1 = [p + " " + " ".join(f"*_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompt_list_1)]
        elif popt_kwargs['placeholder_position'] == 'start':
            prompt_list_1 = [" ".join(f"*_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompt_list_1)]
        prompt1[1] = prompt_list_1[0]

        prompt_list_2 = [prompt2[1]]
        if popt_kwargs['placeholder_position'] == 'end':
            prompt_list_2 = [p + " " + " ".join(f"*_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompt_list_2)]
        elif popt_kwargs['placeholder_position'] == 'start':
            prompt_list_2 = [" ".join(f"*_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompt_list_2)]
        prompt2[1] = prompt_list_2[0]

        null_prompt_embeds, prompt_embeds, add_cond_kwargs = self.get_embed_from_prompt12(prompt1, prompt2)

        at = self.scheduler.alphas_cumprod[t]

        t_mg = int(
            len(self.scheduler.alphas_cumprod_default) * popt_kwargs['p_ratio']
        )
        at_mg = self.scheduler.alphas_cumprod_default[t_mg]
        if popt_kwargs['dynamic_pr']:
            next_t = t - self.skip + 1
            t_mg = int(
                len(self.scheduler.alphas_cumprod_default) - next_t
            )
            at_mg = self.scheduler.alphas_cumprod[t_mg]
        
        t_mg = torch.tensor(t_mg).to(t.device)

        add_cond_kwargs_base['text_embeds'] = add_cond_kwargs_base['text_embeds'][-1].unsqueeze(0).detach()
        add_cond_kwargs_base['time_ids'] = add_cond_kwargs_base['time_ids'][-1].unsqueeze(0).detach()
        
        for i in range(popt_kwargs['p_opt_iter']):
            add_cond_kwargs['text_embeds'] = add_cond_kwargs['text_embeds'][-1].unsqueeze(0)
            add_cond_kwargs['time_ids'] = add_cond_kwargs['time_ids'][-1].unsqueeze(0)

            _, noise_pred = self.predict_noise(zt, t, None, prompt_embeds, add_cond_kwargs)

            # tweedie (x0hat)
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            rand_noise = torch.randn_like(noise_pred, device=noise_pred.device)
            zs = at_mg.sqrt() * z0t + (1-at_mg).sqrt() * rand_noise
            _, noise_pred_s = self.predict_noise(zs, t_mg, None, prompt_embeds_base.detach(), add_cond_kwargs_base)
            
            # tweedie (x0doublehat)
            z0s = (zs - (1-at_mg).sqrt() * noise_pred_s) / at_mg.sqrt()

            assert z0t.shape == z0s.shape and len(z0t.shape) == 4
            term_1 = (z0t.detach() - z0s).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
            term_2 = (z0t - z0s.detach()).reshape(z0t.shape[0], -1).norm(p=2.0, dim=-1)
            ms = term_1 + popt_kwargs['sg_lambda'] * term_2
            loss = -1 * ms.sum()
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Let's make sure we don't update any embedding weights besides the newly added token
            self.restore_embedding(placeholder_token_ids_enc1, orig_embeds_params_enc1, self.tokenizer_1, self.text_enc_1)
            self.restore_embedding(placeholder_token_ids_enc2, orig_embeds_params_enc2, self.tokenizer_2, self.text_enc_2)
            
            if not i == popt_kwargs['p_opt_iter'] - 1:
                null_prompt_embeds, prompt_embeds, add_cond_kwargs = self.get_embed_from_prompt12(prompt1, prompt2)
            else:
                with torch.no_grad():
                    null_prompt_embeds, prompt_embeds, add_cond_kwargs = self.get_embed_from_prompt12(prompt1, prompt2)

        return prompt_embeds, add_cond_kwargs

    def get_embed_from_prompt12(self, prompt1, prompt2, te1_nograd=False, te2_nograd=False):
        # 0. Default height and width to unet
        height = self.default_sample_size * self.vae_scale_factor
        width = self.default_sample_size * self.vae_scale_factor

        original_size = self.original_size or (height, width)
        target_size = self.target_size or (height, width)

        # embedding
        (null_prompt_embeds,
         prompt_embeds,
         pool_null_embed,
         pool_prompt_embed) = self.differentiable_get_text_embed(prompt1[0], prompt1[1], prompt2[0], prompt2[1], self.clip_skip, te1_nograd, te2_nograd)

        # prepare kwargs for SDXL
        add_text_embeds = pool_prompt_embed
        add_time_ids = self._get_add_time_ids(
            original_size,
            self.crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
        )

        if self.negative_original_size is not None and self.negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                self.negative_original_size,
                self.negative_crops_coords_top_left,
                self.negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
            )
        else:
            negative_add_time_ids = add_time_ids
        negative_text_embeds = pool_null_embed

        if self.cfg_guidance != 0.0 and self.cfg_guidance != 1.0:
            # do cfg
            add_text_embeds = torch.cat([negative_text_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        add_cond_kwargs = {
            'text_embeds': add_text_embeds.to(self.device),
            'time_ids': add_time_ids.to(self.device)
        }
        return null_prompt_embeds, prompt_embeds, add_cond_kwargs

    def restore_embedding(self, placeholder_token_ids, orig_embeds_params, tokenizer, text_enc):
        index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
        index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

        with torch.no_grad():
            text_enc.get_input_embeddings().weight[
                index_no_updates
            ] = orig_embeds_params[index_no_updates]

    def initialize_embedding(self, tokenizer, text_enc, popt_kwargs, b_size=1):
        num_opt_tokens = popt_kwargs['num_opt_tokens'] * b_size # assignging popt_kwargs['num_opt_tokens'] tokens per each sample
        init_type = popt_kwargs['init_type']
        init_word = popt_kwargs['init_word']
        init_gau_scale = popt_kwargs['init_gau_scale']
        init_rand_vocab = popt_kwargs['init_rand_vocab']
        init_max_cs = False
        num_vocab = len(tokenizer)

        assert init_type in ['default', 'word', 'gaussian', 'gaussian_white']
        
        token_embeds_base = text_enc.get_input_embeddings().weight.data.detach().clone()

        placeholder_string = popt_kwargs['placeholder_string']
        # assert popt_kwargs['num_opt_tokens'] == 1 # for now, we only support one token
        assert "_" in placeholder_string and len(placeholder_string.split("_")) == 2 # the tokens should take the form of "*_0"
        
        placeholder_tokens = [placeholder_string]
        additional_tokens = []
        
        placeholder_symbol = placeholder_string.split("_")[0]
        for i in range(1, num_opt_tokens):
            print("Additional placeholder token: ", f"{placeholder_symbol}_{i}")
            additional_tokens.append(f"{placeholder_symbol}_{i}")
        placeholder_tokens += additional_tokens
        print("Placeholder tokens: ", placeholder_tokens)

        num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
        print("Number of tokens added to tokenizer: ", num_added_tokens)
        if num_added_tokens != num_opt_tokens:
            # print(f"The tokenizer already contains the token {placeholder_string}.")
            raise ValueError(
                f"The tokenizer already contains the token {placeholder_string}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )
        
        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        print("Placeholder token ids: ", placeholder_token_ids)

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_enc.resize_token_embeddings(len(tokenizer))
        
        if init_type == 'word':
            assert (init_rand_vocab != init_max_cs) or (init_rand_vocab == False and init_max_cs == False)
            if init_rand_vocab:
                token_embeds = text_enc.get_input_embeddings().weight.data
                with torch.no_grad():
                    for token_id in placeholder_token_ids:
                        rand_idx = torch.randint(0, num_vocab, (1,))
                        print(f"Initialize token id {token_id} as a random vocabulary of index {rand_idx}.")
                        token_embeds[token_id] = token_embeds[rand_idx].clone()
            elif init_max_cs:
                assert self.prompt1 == self.prompt2 # for now, we only support the same prompt for both encoders
                # get rid of indices of special tokens in token_embeds_base
                special_token_ids = torch.tensor(tokenizer.all_special_ids)
                token_embeds_min_cs = token_embeds_base[~torch.isin(torch.arange(token_embeds_base.shape[0]), special_token_ids)]

                prompt1 = self.prompt1.copy()
                prompt1_ids = tokenizer.encode(prompt1[1], add_special_tokens=False, return_tensors='pt').squeeze()
                prompt1_embeds = token_embeds_min_cs[prompt1_ids]
                cos_sims = torch.einsum('ij,kj->ik', token_embeds_min_cs, prompt1_embeds).sum(dim=-1)
                min_idx = cos_sims.argmax()
                # import ipdb; ipdb.set_trace()
                
                token_embeds = text_enc.get_input_embeddings().weight.data
                with torch.no_grad():    
                    for token_id in placeholder_token_ids:
                        print(f"Initialize token id {token_id} as a min-cs vocabulary of index {min_idx}.")
                        token_embeds[token_id] = token_embeds_min_cs[min_idx].clone()
            else:
                assert init_word != ""
                # Convert the initializer_token, placeholder_token to ids
                token_ids = tokenizer.encode(init_word, add_special_tokens=False)
                # Check if initializer_token is a single token or a sequence of tokens
                if len(token_ids) > 1:
                    raise ValueError("The initializer token must be a single token.")
                
                initializer_token_id = token_ids[0]
                
                token_embeds = text_enc.get_input_embeddings().weight.data

                with torch.no_grad():
                    for token_id in placeholder_token_ids:
                        print(f"Initialize token id {token_id} as the token embeddin of {init_word}.")
                        # print(f"token_embeds[{token_id}] (before replacement): ", token_embeds[token_id])
                        token_embeds[token_id] = token_embeds[initializer_token_id].clone()
                        # print(f"token_embeds[{token_id}] (after replacement): ", token_embeds[token_id])                


        elif 'gaussian' in init_type:
            embeds_mean = token_embeds_base.mean(dim=0)
            if init_type == 'gaussian_white':
                var_vector = (token_embeds_base ** 2 - embeds_mean.unsqueeze(0) ** 2).mean(dim=0)
                embeds_cov = torch.diag(var_vector) * (init_gau_scale ** 2)
            elif init_type == 'gaussian':
                embeds_cov = torch.einsum('ij,ik->jk', token_embeds_base, token_embeds_base) / token_embeds_base.shape[0]
                embeds_cov = embeds_cov.float() * (init_gau_scale ** 2)
                
            # Create a multivariate normal distribution
            
            mvn = torch.distributions.MultivariateNormal(embeds_mean, covariance_matrix=embeds_cov)

            token_embeds = text_enc.get_input_embeddings().weight.data
            with torch.no_grad():
                for token_id in placeholder_token_ids:
                    print(f"Initialize token id {token_id} as a multivariate normal distribution ({init_type}).")
                    token_embeds[token_id] = mvn.sample()
        elif init_type == 'default':
            print("Default initialization of newly-added embeddings.")


        # Freeze all parameters except for the token embeddings in text encoder
        text_enc.text_model.encoder.requires_grad_(False)
        text_enc.text_model.final_layer_norm.requires_grad_(False)
        text_enc.text_model.embeddings.position_embedding.requires_grad_(False)

        return placeholder_token_ids
                

class SDXLLightning(SDXL):
    def __init__(self, 
                 solver_config: dict,
                 base_model_key:str="stabilityai/stable-diffusion-xl-base-1.0",
                 light_model_ckpt:str="ckpt/sdxl_lightning_4step_unet.safetensors",
                 dtype=torch.float16,
                 device='cuda'):

        self.device = device

        # load the student model
        unet = UNet2DConditionModel.from_config(base_model_key, subfolder="unet").to("cuda", torch.float16)
        ext = os.path.splitext(light_model_ckpt)[1]
        if ext == ".safetensors":
            state_dict = load_file(light_model_ckpt)
        else:
            state_dict = torch.load(light_model_ckpt, map_location="cpu")
        print(unet.load_state_dict(state_dict, strict=True))
        unet.requires_grad_(False)
        self.unet = unet

        #pipe2 = StableDiffusionXLPipeline.from_single_file(light_model_ckpt, torch_dtype=dtype).to(device)
        pipe = StableDiffusionXLPipeline.from_pretrained(base_model_key, unet=self.unet, torch_dtype=dtype).to(device)
        self.dtype = dtype

        # avoid overflow in float16
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype).to(device)

        self.tokenizer_1_base = copy.deepcopy(pipe.tokenizer)
        self.tokenizer_2_base = copy.deepcopy(pipe.tokenizer_2)
        self.text_enc_1_base = copy.deepcopy(pipe.text_encoder)
        self.text_enc_2_base = copy.deepcopy(pipe.text_encoder_2)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.default_sample_size = self.unet.config.sample_size

        # sampling parameters
        self.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        self.total_alphas = self.scheduler.alphas_cumprod.clone()
        N_ts = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
        self.skip = N_ts // solver_config.num_sampling

        #self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
        self.scheduler.alphas_cumprod_default = self.scheduler.alphas_cumprod
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod]).to(device)

###########################################
# Base version
###########################################

@register_solver('ddim')
class BaseDDIM(SDXL):
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        popt_kwargs=None,
                        etc_kwargs=None,
                        **kwargs):
        #################################
        # Sample region - where to change
        #################################
        # initialize zT            
        zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))

        # initialize embedding for prompt-opt
        if popt_kwargs['prompt_opt']:
            self.text_enc_1 = self.text_enc_1.to(torch.float32)
            self.text_enc_2 = self.text_enc_2.to(torch.float32)
            placeholder_token_ids_enc1 = self.initialize_embedding(self.tokenizer_1, self.text_enc_1, popt_kwargs)
            placeholder_token_ids_enc2 = self.initialize_embedding(self.tokenizer_2, self.text_enc_2, popt_kwargs)
            self.vae.requires_grad_(False)

        prompt_embeds_base = prompt_embeds.detach().clone()
        null_prompt_embeds_base = null_prompt_embeds.detach().clone()
        add_cond_kwargs_base = add_cond_kwargs.copy()
        
        # sampling
        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
        for step, t in enumerate(pbar):
            next_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_next = self.scheduler.alphas_cumprod[next_t]

            # for prompt-opt
            if popt_kwargs['prompt_opt'] and t > popt_kwargs['t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % popt_kwargs['inter_rate'] == 0:
                prompt_embeds, add_cond_kwargs = self.prompt_opt(
                    zt,
                    t,
                    step,
                    placeholder_token_ids_enc1,
                    placeholder_token_ids_enc2,
                    null_prompt_embeds,
                    prompt_embeds_base,
                    add_cond_kwargs_base,
                    cfg_guidance,
                    popt_kwargs
                )
            else:
                if popt_kwargs['prompt_opt'] and popt_kwargs['base_prompt_after_popt']:
                    prompt_embeds = prompt_embeds_base.detach().clone()
                    add_cond_kwargs = add_cond_kwargs_base.copy()

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_pred

            if callback_fn is not None:
                callback_kwargs = { 'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last stpe, do not add noise
        return z0t


@register_solver('ddim_lightning')
class BaseDDIMLight(BaseDDIM, SDXLLightning):
    def __init__(self, **kwargs):
        SDXLLightning.__init__(self, **kwargs)
    
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        popt_kwargs=None,
                        etc_kwargs=None,
                        **kwargs):
        assert cfg_guidance == 1.0, "CFG should be turned off in the lightning version"
        return super().reverse_process(null_prompt_embeds, 
                                        prompt_embeds, 
                                        cfg_guidance, 
                                        add_cond_kwargs, 
                                        shape, 
                                        callback_fn,
                                        popt_kwargs=popt_kwargs,
                                        etc_kwargs=etc_kwargs,
                                        **kwargs)


@register_solver("ddim_edit")
class EditWardSwapDDIM(BaseDDIM):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               prompt1 = ["", "", ""],
               prompt2 = ["", "", ""],
               cfg_guidance:float=5.0,
               original_size: Optional[Tuple[int, int]]=None,
               crops_coords_top_left: Tuple[int, int]=(0, 0),
               target_size: Optional[Tuple[int, int]]=None,
               negative_original_size: Optional[Tuple[int, int]]=None,
               negative_crops_coords_top_left: Tuple[int, int]=(0, 0),
               negative_target_size: Optional[Tuple[int, int]]=None,
               clip_skip: Optional[int]=None,
               **kwargs):

        # 0. Default height and width to unet
        height = self.default_sample_size * self.vae_scale_factor
        width = self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # embedding
        (null_prompt_embeds,
         src_prompt_embeds,
         pool_null_embed,
         pool_src_prompt_embed) = self.get_text_embed(prompt1[0], prompt1[1], prompt2[0], prompt2[1], clip_skip)

        (_,
         tgt_prompt_embeds,
         _,
         pool_tgt_prompt_embed) = self.get_text_embed(prompt1[0], prompt1[2], prompt2[0], prompt2[2], clip_skip)

        # prepare kwargs for SDXL
        add_src_text_embeds = pool_src_prompt_embed
        add_tgt_text_embeds = pool_tgt_prompt_embed

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=src_prompt_embeds.dtype,
            text_encoder_projection_dim=int(pool_src_prompt_embed.shape[-1]),
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=src_prompt_embeds.dtype,
                text_encoder_projection_dim=int(pool_src_prompt_embed.shape[-1]),
            )
        else:
            negative_add_time_ids = add_time_ids
        negative_text_embeds = pool_null_embed 

        if cfg_guidance != 0.0 and cfg_guidance != 1.0:
            # do cfg
            add_src_text_embeds = torch.cat([negative_text_embeds, add_src_text_embeds], dim=0)
            add_tgt_text_embeds = torch.cat([negative_text_embeds, add_tgt_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        add_src_cond_kwargs = {
            'text_embeds': add_src_text_embeds.to(self.device),
            'time_ids': add_time_ids.to(self.device)
        }

        add_tgt_cond_kwargs = {
            'text_embeds': add_tgt_text_embeds.to(self.device),
            'time_ids': add_time_ids.to(self.device)
        }

        # reverse sampling
        zt = self.reverse_process(null_prompt_embeds,
                                  src_prompt_embeds, 
                                  tgt_prompt_embeds,
                                  cfg_guidance,
                                  add_src_cond_kwargs,
                                  add_tgt_cond_kwargs,
                                  **kwargs)

        # decode
        with torch.no_grad():
            img = self.decode(zt)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

    def reverse_process(self,
                        null_prompt_embeds,
                        src_prompt_embeds,
                        tgt_prompt_embed,
                        cfg_guidance,
                        add_src_cond_kwargs,
                        add_tgt_cond_kwargs,
                        callback_fn=None,
                        popt_kwargs=None,
                        **kwargs):
        #################################
        # Sample region - where to change
        #################################
        # initialize zT
        zt = self.initialize_latent(method='ddim',
                                    src_img=kwargs.get('src_img', None),
                                    uc=null_prompt_embeds,
                                    c=src_prompt_embeds,
                                    cfg_guidance=cfg_guidance,
                                    add_cond_kwargs=add_src_cond_kwargs)

        # sampling
        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_next = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, 
                                                       null_prompt_embeds,
                                                       tgt_prompt_embed,
                                                       add_tgt_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_pred

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last stpe, do not add noise
        return z0t


###########################################
# CFG++ version
###########################################

@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(SDXL):
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        popt_kwargs=None,
                        etc_kwargs=None,
                        **kwargs):
        #################################
        # Sample region - where to change
        #################################
        # initialize zT
        zt = self.initialize_latent(size=(1, 4, shape[1] // self.vae_scale_factor, shape[0] // self.vae_scale_factor))
        
        # initialize embedding for prompt-opt
        if popt_kwargs['prompt_opt']:
            self.text_enc_1 = self.text_enc_1.to(torch.float32)
            self.text_enc_2 = self.text_enc_2.to(torch.float32)
            placeholder_token_ids_enc1 = self.initialize_embedding(self.tokenizer_1, self.text_enc_1, popt_kwargs)
            placeholder_token_ids_enc2 = self.initialize_embedding(self.tokenizer_2, self.text_enc_2, popt_kwargs)
            self.vae.requires_grad_(False)
            # self.unet.requires_grad_(False)

        prompt_embeds_base = prompt_embeds.detach().clone()
        add_cond_kwargs_base = add_cond_kwargs.copy()

        # sampling
        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
        for step, t in enumerate(pbar):
            next_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_next = self.scheduler.alphas_cumprod[next_t]
            
            # for prompt-opt
            if popt_kwargs['prompt_opt'] and t > popt_kwargs['t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % popt_kwargs['inter_rate'] == 0:
                prompt_embeds, add_cond_kwargs = self.prompt_opt(
                    zt,
                    t,
                    step,
                    placeholder_token_ids_enc1,
                    placeholder_token_ids_enc2,
                    null_prompt_embeds,
                    prompt_embeds_base,
                    add_cond_kwargs_base,
                    cfg_guidance,
                    popt_kwargs
                ) # return optimized prompts
            else:
                if popt_kwargs['prompt_opt'] and popt_kwargs['base_prompt_after_popt']:
                    prompt_embeds = prompt_embeds_base.detach().clone()
                    add_cond_kwargs = add_cond_kwargs_base.copy()


            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = { 'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last stpe, do not add noise
        return z0t

@register_solver('ddim_cfg++_lightning')
class BaseDDIMCFGppLight(BaseDDIMCFGpp, SDXLLightning):
    def __init__(self, **kwargs):
        SDXLLightning.__init__(self, **kwargs)
    
    def reverse_process(self,
                        null_prompt_embeds,
                        prompt_embeds,
                        cfg_guidance,
                        add_cond_kwargs,
                        shape=(1024, 1024),
                        callback_fn=None,
                        popt_kwargs=None,
                        etc_kwargs=None,
                        **kwargs):
        assert cfg_guidance == 1.0, "CFG should be turned off in the lightning version"
        return super().reverse_process(null_prompt_embeds, 
                                        prompt_embeds, 
                                        cfg_guidance, 
                                        add_cond_kwargs, 
                                        shape,
                                        callback_fn,
                                        popt_kwargs=popt_kwargs,
                                        etc_kwargs=etc_kwargs,
                                        **kwargs)


@register_solver("ddim_edit_cfg++")
class EditWardSwapDDIMCFGpp(EditWardSwapDDIM):
    @torch.no_grad()
    def inversion(self, z0, uc, c, cfg_guidance, add_cond_kwargs):
        # if we use cfg_guidance=0.0 or 1.0 for inversion, add_cond_kwargs must be splitted. 
        if cfg_guidance == 0.0 or cfg_guidance == 1.0:
            add_cond_kwargs['text_embeds'] = add_cond_kwargs['text_embeds'][-1].unsqueeze(0)
            add_cond_kwargs['time_ids'] = add_cond_kwargs['time_ids'][-1].unsqueeze(0)

        zt = z0.clone().to(self.device)
        pbar = tqdm(reversed(self.scheduler.timesteps), desc='DDIM inversion')
        for _, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            noise_uc, noise_c  = self.predict_noise(zt, t, uc, c, add_cond_kwargs)
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at_prev).sqrt() * noise_uc) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1-at).sqrt() * noise_pred

        return zt

    def reverse_process(self,
                        null_prompt_embeds,
                        src_prompt_embeds,
                        tgt_prompt_embed,
                        cfg_guidance,
                        add_src_cond_kwargs,
                        add_tgt_cond_kwargs,
                        callback_fn=None,
                        **kwargs):
        #################################
        # Sample region - where to change
        #################################
        # initialize zT
        zt = self.initialize_latent(method='ddim',
                                    src_img=kwargs.get('src_img', None),
                                    uc=null_prompt_embeds,
                                    c=src_prompt_embeds,
                                    cfg_guidance=cfg_guidance,
                                    add_cond_kwargs=add_src_cond_kwargs)

        # sampling
        pbar = tqdm(self.scheduler.timesteps.int(), desc='SDXL')
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_next = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, 
                                                       null_prompt_embeds,
                                                       tgt_prompt_embed,
                                                       add_tgt_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_next.sqrt() * z0t + (1-at_next).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last stpe, do not add noise
        return z0t
#############################

if __name__ == "__main__":
    # print all list of solvers
    print(f"Possble solvers: {[x for x in __SOLVER__.keys()]}")
        
