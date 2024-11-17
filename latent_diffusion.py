"""
This module includes LDM-based inverse problem solvers.
Forward operators follow DPS and DDRM/DDNM.
"""

from typing import Any, Callable, Dict, Optional

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
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

class StableDiffusion():
    def __init__(self,
                 solver_config: Dict,
                #  model_key:str="runwayml/stable-diffusion-v1-5",
                 model_key:str="botp/stable-diffusion-v1-5",
                 device: Optional[torch.device]=None,
                 **kwargs):
        self.device = device

        self.dtype = kwargs.get("pipe_dtype", torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.dtype).to(device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.tokenizer_base = copy.deepcopy(pipe.tokenizer)
        self.text_encoder_base = copy.deepcopy(pipe.text_encoder)

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        total_timesteps = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
        self.skip = total_timesteps // solver_config.num_sampling

        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
        self.scheduler.alphas_cumprod_default = self.scheduler.alphas_cumprod
        self.scheduler.alphas_cumprod_default = self.scheduler.alphas_cumprod_default.to(device)
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod]).to(device)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.sample(*args, **kwargs)

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Solver must implement sample() method.")
    
    def alpha(self, t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at

    @torch.no_grad()
    def get_text_embed(self, null_prompt, prompt):
        """
        Get text embedding.
        args:
            null_prompt (str): null text
            prompt (str): guidance text
        """
        # null text embedding (negation)
        null_text_input = self.tokenizer(null_prompt,
                                         padding='max_length',
                                         max_length=self.tokenizer.model_max_length,
                                         return_tensors="pt",)
        null_text_embed = self.text_encoder(null_text_input.input_ids.to(self.device))[0]

        # text embedding (guidance)
        text_input = self.tokenizer(prompt,
                                    padding='max_length',
                                    max_length=self.tokenizer.model_max_length,
                                    return_tensors="pt",
                                    truncation=True)
        text_embed = self.text_encoder(text_input.input_ids.to(self.device))[0]

        return null_text_embed, text_embed
    
    def differentiable_get_text_embed(self, null_prompt, prompt):
        """
        Get text embedding.
        args:
            null_prompt (str): null text
            prompt (str): guidance text
        """
        # null text embedding (negation)
        null_text_input = self.tokenizer(null_prompt,
                                         padding='max_length',
                                         max_length=self.tokenizer.model_max_length,
                                         return_tensors="pt",)
        null_text_embed = self.text_encoder(null_text_input.input_ids.to(self.device))[0]

        # text embedding (guidance)
        text_input = self.tokenizer(prompt,
                                    padding='max_length',
                                    max_length=self.tokenizer.model_max_length,
                                    return_tensors="pt",
                                    truncation=True)
        text_embed = self.text_encoder(text_input.input_ids.to(self.device))[0]

        return null_text_embed, text_embed

    def encode(self, x):
        """
        xt -> zt
        """
        return self.vae.encode(x).latent_dist.sample() * 0.18215

    def decode(self, zt):
        """
        zt -> xt
        """
        zt = 1/0.18215 * zt
        img = self.vae.decode(zt).sample.float()
        return img

    def predict_noise(self,
                      zt: torch.Tensor,
                      t: torch.Tensor,
                      uc: torch.Tensor,
                      c: torch.Tensor):
        """
        compuate epsilon_theta for null and condition
        args:
            zt (torch.Tensor): latent features
            t (torch.Tensor): timestep
            uc (torch.Tensor): null-text embedding
            c (torch.Tensor): text embedding
        """
        t_in = t.unsqueeze(0) if len(t.shape) == 0 else t
        # print("t_in.shape: ", t_in.shape)
        if uc is None:
            noise_c = self.unet(zt, t_in, encoder_hidden_states=c)['sample']
            noise_uc = noise_c
        elif c is None:
            noise_uc = self.unet(zt, t_in, encoder_hidden_states=uc)['sample']
            noise_c = noise_uc
        else:
            c_embed = torch.cat([uc, c], dim=0)
            z_in = torch.cat([zt] * 2) 
            t_in = torch.cat([t_in] * 2)
            noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed)['sample']
            noise_uc, noise_c = noise_pred.chunk(2)

        return noise_uc, noise_c

    @torch.no_grad()
    def inversion(self,
                  z0: torch.Tensor,
                  uc: torch.Tensor,
                  c: torch.Tensor,
                  cfg_guidance: float=1.0):

        # initialize z_0
        zt = z0.clone().to(self.device)
         
        # loop
        pbar = tqdm(reversed(self.scheduler.timesteps), desc='DDIM Inversion')
        for _, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            noise_uc, noise_c = self.predict_noise(zt, t, uc, c) 
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at_prev).sqrt() * noise_pred) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1-at).sqrt() * noise_pred

        return zt
    
    def initialize_latent(self,
                          method: str='random',
                          src_img: Optional[torch.Tensor]=None,
                          b_size: int=1,
                          **kwargs):
        if method == 'ddim':
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('uc'),
                               kwargs.get('c'),
                               cfg_guidance=kwargs.get('cfg_guidance', 0.0))
        elif method == 'npi':
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('c'),
                               kwargs.get('c'),
                               cfg_guidance=1.0)
        elif method == 'random':
            size = kwargs.get('latent_dim', (b_size, 4, 64, 64))
            z = torch.randn(size).to(self.device)
        else: 
            raise NotImplementedError

        return z.requires_grad_()
    

    @torch.enable_grad()
    def prompt_opt(self, zt, t, step, placeholder_token_ids_enc, uc, c_base, cfg_guidance, popt_kwargs):
        assert cfg_guidance > 0.0
        placeholder_string = popt_kwargs['placeholder_string']
        assert "_" in placeholder_string and len(placeholder_string.split("_")) == 2
        placeholder_symbol = placeholder_string.split("_")[0]

        decay_rate = popt_kwargs['lr_decay_rate']
        num_opt_tokens = popt_kwargs['num_opt_tokens']        

        para = self.text_encoder.get_input_embeddings().parameters()
        optimizer = Adam(para, lr=popt_kwargs['p_opt_lr'] * (1. - step * decay_rate))

        # keep original embeddings as reference
        orig_embeds_params_enc = self.text_encoder.get_input_embeddings().weight.data.clone()

        prompt = self.prompt.copy()

        # add placeholder tokens only for prompt
        prompt_list = [prompt[1]]
        if popt_kwargs['placeholder_position'] == 'end':
            prompt_list = [p + " " + " ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompt_list)]
        elif popt_kwargs['placeholder_position'] == 'start':
            prompt_list = [" ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompt_list)]
        prompt[1] = prompt_list[0]

        uc, c = self.differentiable_get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        at = self.alpha(t)

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
        
        for i in range(popt_kwargs['p_opt_iter']):
            _, noise_pred = self.predict_noise(zt, t, None, c)
            
            # tweedie (x0hat)
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            rand_noise = torch.randn_like(noise_pred, device=noise_pred.device)
            zs = at_mg.sqrt() * z0t + (1-at_mg).sqrt() * rand_noise
            _, noise_pred_s = self.predict_noise(zs, t_mg, None, c_base.detach())
            
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
            self.restore_embedding(placeholder_token_ids_enc, orig_embeds_params_enc, self.tokenizer, self.text_encoder)
            if not i == popt_kwargs['p_opt_iter'] - 1:
                uc, c = self.differentiable_get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
            else:
                uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        return c
    
    
    
    @torch.enable_grad()
    def batch_prompt_opt(self, zt, ts, step, placeholder_token_ids_enc, uc, c_base, cfg_guidance, popt_kwargs):
        assert cfg_guidance > 0.0
        placeholder_string = popt_kwargs['placeholder_string']
        decay_rate = popt_kwargs['lr_decay_rate']
        num_opt_tokens = popt_kwargs['num_opt_tokens']

        para = self.text_encoder.get_input_embeddings().parameters()
        optimizer = Adam(para, lr=popt_kwargs['p_opt_lr'] * (1. - step * decay_rate))

        # keep original embeddings as reference
        orig_embeds_params_enc = self.text_encoder.get_input_embeddings().weight.data.clone()

        prompts = self.prompts.copy()
        null_prompts = self.null_prompts.copy()
        b_size = len(prompts)

        # add placeholder tokens only for prompt
        assert "_" in placeholder_string and len(placeholder_string.split("_")) == 2
        placeholder_symbol = placeholder_string.split("_")[0]
        if popt_kwargs['placeholder_position'] == 'end':
            prompts = [p + " " + " ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompts)]
        elif popt_kwargs['placeholder_position'] == 'start':
            prompts = [" ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompts)]

        _, c = self.differentiable_get_text_embed(null_prompt=null_prompts, prompt=prompts)

        at = self.scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)

        t_mg = int(
            len(self.scheduler.alphas_cumprod_default) * popt_kwargs['p_ratio']
        )
        ts_mg = torch.full((b_size,), t_mg, device=ts.device, dtype=torch.long)
        at_mg = self.scheduler.alphas_cumprod_default[ts_mg].view(b_size, 1, 1, 1)
        if popt_kwargs['dynamic_pr']:
            next_t = ts[0].item() - self.skip + 1
            t_mg = int(
                len(self.scheduler.alphas_cumprod_default) - next_t
            )
            ts_mg = torch.full((b_size,), t_mg, device=ts.device, dtype=torch.long)
            at_mg = self.scheduler.alphas_cumprod[ts_mg].view(b_size, 1, 1, 1)
        
        for i in range(popt_kwargs['p_opt_iter']):
            _, noise_pred = self.predict_noise(zt, ts, None, c)
            
            # tweedie (x0hat)
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            rand_noise = torch.randn_like(noise_pred, device=noise_pred.device)
            zs = at_mg.sqrt() * z0t + (1-at_mg).sqrt() * rand_noise
            _, noise_pred_s = self.predict_noise(zs, ts_mg, None, c_base.detach())
            
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
            self.restore_embedding(placeholder_token_ids_enc, orig_embeds_params_enc, self.tokenizer, self.text_encoder)
            
            if not i == popt_kwargs['p_opt_iter'] - 1:
                _, c = self.differentiable_get_text_embed(null_prompt=null_prompts, prompt=prompts)
            else:
                _, c = self.get_text_embed(null_prompt=null_prompts, prompt=prompts)

        return c
    
    
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
        num_vocab = len(tokenizer)

        assert init_type in ['default', 'word', 'gaussian', 'gaussian_white']
        
        if 'gaussian' in init_type:
            token_embeds_base = text_enc.get_input_embeddings().weight.data.detach().clone()

        placeholder_string = popt_kwargs['placeholder_string']
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
            raise ValueError(
                f"The tokenizer already contains the token {placeholder_string}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )
        
        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        print("Placeholder token ids: ", placeholder_token_ids)

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_enc.resize_token_embeddings(len(tokenizer))
        
        if init_type == 'word':
            if not init_rand_vocab:
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
            else:
                token_embeds = text_enc.get_input_embeddings().weight.data
                with torch.no_grad():
                    for token_id in placeholder_token_ids:
                        rand_idx = torch.randint(0, num_vocab, (1,))
                        print(f"Initialize token id {token_id} as a random vocabulary of index {rand_idx}.")
                        token_embeds[token_id] = token_embeds[rand_idx].clone()


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
    
    @torch.enable_grad()
    def popt_diverse(self, zt, ts, step, placeholder_token_ids_enc, uc, c_base, cfg_guidance, popt_kwargs):
        assert cfg_guidance > 0.0
        placeholder_string = popt_kwargs['placeholder_string']
        decay_rate = popt_kwargs['lr_decay_rate']
        num_opt_tokens = popt_kwargs['num_opt_tokens']

        para = self.text_encoder.get_input_embeddings().parameters()
        optimizer = Adam(para, lr=popt_kwargs['p_opt_lr'] * (1. - step * decay_rate))

        # keep original embeddings as reference
        orig_embeds_params_enc = self.text_encoder.get_input_embeddings().weight.data.clone()

        prompts = self.prompts.copy()
        null_prompts = self.null_prompts.copy()
        b_size = len(prompts)
        assert b_size > 1 # batch size should be larger than 1 for diverse generation

        # add placeholder tokens only for prompt
        assert "_" in placeholder_string and len(placeholder_string.split("_")) == 2
        placeholder_symbol = placeholder_string.split("_")[0]
        if popt_kwargs['placeholder_position'] == 'end':
            prompts = [p + " " + " ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) for idx, p in enumerate(prompts)]
        elif popt_kwargs['placeholder_position'] == 'start':
            prompts = [" ".join(f"{placeholder_symbol}_{num_opt_tokens*idx+i}" for i in range(num_opt_tokens)) + " " + p for idx, p in enumerate(prompts)]

        _, c = self.differentiable_get_text_embed(null_prompt=null_prompts, prompt=prompts)

        at = self.scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)
        
        for i in range(popt_kwargs['p_opt_iter']):
            _, noise_pred = self.predict_noise(zt, ts, None, c)
            
            # tweedie (x0hat)
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
            loss_per_sample = torch.zeros(b_size, device=zt.device)
            for i in range(b_size):
                for j in range(b_size):
                    if i == j:
                        continue
                    repel_from = z0t[j]
                    loss_per_sample[i] += (z0t[i] - repel_from).reshape(-1).norm(p=2.0)
            loss = -1 * loss_per_sample.sum() # encouraging z0t to be diverse
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # Let's make sure we don't update any embedding weights besides the newly added token
            self.restore_embedding(placeholder_token_ids_enc, orig_embeds_params_enc, self.tokenizer, self.text_encoder)
            
            if not i == popt_kwargs['p_opt_iter'] - 1:
                _, c = self.differentiable_get_text_embed(null_prompt=null_prompts, prompt=prompts)
            else:
                _, c = self.get_text_embed(null_prompt=null_prompts, prompt=prompts)

        return c



###########################################
# Base version
###########################################

@register_solver("ddim")
class BaseDDIM(StableDiffusion):
    """
    Basic DDIM solver for SD.
    Useful for text-to-image generation
    """

    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def sample(self,
               cfg_guidance=7.5,
               prompt=["",""],
               callback_fn=None,
               popt_kwargs=None,
               etc_kwargs=None,
               **kwargs):
        """
        Main function that defines each solver.
        This will generate samples without considering measurements.
        """

        self.prompt = prompt
        
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        
        c_base = c.detach().clone()
        uc_base = uc.detach().clone()

        # Initialize zT
        zt = self.initialize_latent()
        zt = zt.requires_grad_()

        if popt_kwargs['prompt_opt']:
            self.text_encoder = self.text_encoder.to(torch.float32)
            placeholder_token_ids_enc = self.initialize_embedding(self.tokenizer, self.text_encoder, popt_kwargs)
            self.vae.requires_grad_(False)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            # for prompt-opt
            if popt_kwargs['prompt_opt'] and t > popt_kwargs['t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % popt_kwargs['inter_rate'] == 0:
                c = self.prompt_opt(
                    zt.detach(),
                    t,
                    step,
                    placeholder_token_ids_enc,
                    uc,
                    c_base,
                    cfg_guidance,
                    popt_kwargs
                )
            else:
                if popt_kwargs['prompt_opt'] and popt_kwargs['base_prompt_after_popt']:
                    c = c_base.detach().clone()

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]
        
        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    
    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def batch_sample(self,
               cfg_guidance=7.5,
               prompts=[""],
               null_prompts=[""],
               popt_kwargs=None,
               etc_kwargs=None,
               **kwargs):
        """
        Main function that defines each solver.
        This will generate samples without considering measurements.
        """
        assert len(prompts) == len(null_prompts)
        assert isinstance(prompts, list) and isinstance(null_prompts, list)
        self.prompts = prompts
        self.null_prompts = null_prompts

        # reset tokenizer and text_encoder
        self.tokenizer = copy.deepcopy(self.tokenizer_base)
        self.text_encoder = copy.deepcopy(self.text_encoder_base)

        b_size = len(prompts)

        # Text embedding
        uc, c = self.get_text_embed(null_prompt=null_prompts, prompt=prompts)
        c_base = c.detach().clone()
        uc_base = uc.detach().clone()

        # Initialize zT
        zt = self.initialize_latent(b_size=b_size)
        zt = zt.requires_grad_()
        
        if popt_kwargs['prompt_opt'] or popt_kwargs['popt_diverse']:
            self.text_encoder = self.text_encoder.to(torch.float32)
            placeholder_token_ids_enc = self.initialize_embedding(self.tokenizer, self.text_encoder, popt_kwargs, b_size=b_size)
            self.vae.requires_grad_(False)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            ts = torch.full((b_size,), t, device=self.device, dtype=torch.long)

            at = self.scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)
            at_prev = self.scheduler.alphas_cumprod[ts - self.skip].view(b_size, 1, 1, 1)

            if popt_kwargs['prompt_opt'] and t > popt_kwargs['t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % popt_kwargs['inter_rate'] == 0:
                c = self.batch_prompt_opt(
                    zt.detach(),
                    ts,
                    step,
                    placeholder_token_ids_enc,
                    uc,
                    c_base,
                    cfg_guidance, 
                    popt_kwargs
                )
            else:
                if popt_kwargs['prompt_opt'] and popt_kwargs['base_prompt_after_popt']:
                    c = c_base.detach().clone()

            if popt_kwargs['popt_diverse'] and t > popt_kwargs['t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % popt_kwargs['inter_rate'] == 0:
                c = self.popt_diverse(
                    zt.detach(),
                    ts,
                    step,
                    placeholder_token_ids_enc,
                    uc,
                    c_base,
                    cfg_guidance, 
                    popt_kwargs
                )
            else:
                if popt_kwargs['popt_diverse'] and popt_kwargs['base_prompt_after_popt']:
                    c = c_base.detach().clone()

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, ts, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred
        
        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

@register_solver("ddim_inversion")
class InversionDDIM(BaseDDIM):
    """
    Editing via WardSwap after inversion.
    Useful for text-guided image editing.
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def sample(self,
               src_img,
               cfg_guidance=7.5,
               prompt=["","",""],
               callback_fn=None,
               **kwargs):
        
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # Initialize zT
        zt = self.initialize_latent(method='ddim',
                                    src_img=src_img,
                                    uc=uc,
                                    c=c,
                                    cfg_guidance=cfg_guidance)
        zt = zt.requires_grad_()

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]
        
        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

@register_solver("ddim_edit")
class EditWardSwapDDIM(InversionDDIM):
    """
    Editing via WardSwap after inversion.
    Useful for text-guided image editing.
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def sample(self,
               src_img,
               cfg_guidance=7.5,
               prompt=["","",""],
               callback_fn=None,
               **kwargs):
        
        # Text embedding
        uc, src_c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        _, tgt_c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[2])

        # Initialize zT
        zt = self.initialize_latent(method='ddim',
                                    src_img=src_img,
                                    uc=uc,
                                    c=src_c,
                                    cfg_guidance=cfg_guidance)
        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="DDIM-edit")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, tgt_c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred
        
            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

###########################################
# CFG++ version
###########################################

@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(StableDiffusion):
    """
    DDIM solver for SD with CFG++.
    Useful for text-to-image generation
    """
    def __init__(self,
                 solver_config: Dict,
                #  model_key:str="runwayml/stable-diffusion-v1-5",
                 model_key:str="botp/stable-diffusion-v1-5",
                 device: Optional[torch.device]=None,
                 **kwargs):
        super().__init__(solver_config, model_key, device, **kwargs)

    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def sample(self,
               cfg_guidance=7.5,
               prompt=["",""],
               callback_fn=None,
               popt_kwargs=None,
               **kwargs):
        """
        Main function that defines each solver.
        This will generate samples without considering measurements.
        """
        
        self.prompt = prompt

        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        c_base = c.detach().clone()

        # Initialize zT
        zt = self.initialize_latent()
        zt = zt.requires_grad_() # why zt is required grad?
        
        if popt_kwargs['prompt_opt']:
            self.text_encoder = self.text_encoder.to(torch.float32)
            placeholder_token_ids_enc = self.initialize_embedding(self.tokenizer, self.text_encoder, popt_kwargs)
            self.vae.requires_grad_(False)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            # for prompt-opt
            if popt_kwargs['prompt_opt'] and t > popt_kwargs['t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % popt_kwargs['inter_rate'] == 0:
                c = self.prompt_opt(
                    zt.detach(),
                    t,
                    step,
                    placeholder_token_ids_enc,
                    uc,
                    c_base,
                    cfg_guidance, 
                    popt_kwargs
                )
            else:
                if popt_kwargs['base_prompt_after_popt']:
                    c = c_base.detach().clone()

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]
        
        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
    
    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def batch_sample(self,
               cfg_guidance=7.5,
               prompts=[""],
               null_prompts=[""],
               popt_kwargs=None,
               **kwargs):
        """
        Main function that defines each solver.
        This will generate samples without considering measurements.
        """
        assert len(prompts) == len(null_prompts)
        assert isinstance(prompts, list) and isinstance(null_prompts, list)
        self.prompts = prompts
        self.null_prompts = null_prompts

        # reset tokenizer and text_encoder
        self.tokenizer = copy.deepcopy(self.tokenizer_base)
        self.text_encoder = copy.deepcopy(self.text_encoder_base)

        b_size = len(prompts)

        # Text embedding
        uc, c = self.get_text_embed(null_prompt=null_prompts, prompt=prompts)
        c_base = c.detach().clone()

        # Initialize zT
        zt = self.initialize_latent(b_size=b_size)
        zt = zt.requires_grad_()
        
        if popt_kwargs['prompt_opt']:
            self.text_encoder = self.text_encoder.to(torch.float32)
            placeholder_token_ids_enc = self.initialize_embedding(self.tokenizer, self.text_encoder, popt_kwargs, b_size=b_size)
            self.vae.requires_grad_(False)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            ts = torch.full((b_size,), t, device=self.device, dtype=torch.long)

            at = self.scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)
            at_prev = self.scheduler.alphas_cumprod[ts - self.skip].view(b_size, 1, 1, 1)

            # for prompt-opt
            if popt_kwargs['prompt_opt'] and t > popt_kwargs['t_lo'] * len(self.scheduler.alphas_cumprod_default) \
                and step % popt_kwargs['inter_rate'] == 0:
                c = self.batch_prompt_opt(
                    zt.detach(),
                    ts,
                    step,
                    placeholder_token_ids_enc,
                    uc,
                    c_base,
                    cfg_guidance, 
                    popt_kwargs
                )
            else:
                if popt_kwargs['prompt_opt'] and popt_kwargs['base_prompt_after_popt']:
                    c = c_base.detach().clone()

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, ts, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc
        
        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

@register_solver("ddim_inversion_cfg++")
class InversionDDIMCFGpp(BaseDDIMCFGpp):
    """
    Editing via WardSwap after inversion.
    Useful for text-guided image editing.
    """
    @torch.no_grad()
    def inversion(self,
                  z0: torch.Tensor,
                  uc: torch.Tensor,
                  c: torch.Tensor,
                  cfg_guidance: float=1.0):

        # initialize z_0
        zt = z0.clone().to(self.device)
         
        # loop
        pbar = tqdm(reversed(self.scheduler.timesteps), desc='DDIM Inversion')
        for _, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t-self.skip)

            noise_uc, noise_c = self.predict_noise(zt, t, uc, c) 
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at_prev).sqrt() * noise_uc) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1-at).sqrt() * noise_pred

        return zt

    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def sample(self,
               src_img,
               cfg_guidance=7.5,
               prompt=["",""],
               callback_fn=None,
               **kwargs):
        
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # Initialize zT
        zt = self.initialize_latent(method='ddim',
                                    src_img=src_img,
                                    uc=uc,
                                    c=c,
                                    cfg_guidance=cfg_guidance)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]
        
        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

@register_solver("ddim_edit_cfg++")
class EditWardSwapDDIMCFGpp(InversionDDIMCFGpp):
    """
    Editing via WardSwap after inversion.
    Useful for text-guided image editing.
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def sample(self,
               src_img,
               cfg_guidance=7.5,
               prompt=["","",""],
               callback_fn=None,
               **kwargs):
        
        # Text embedding
        uc, src_c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        _, tgt_c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[2])

        # Initialize zT
        zt = self.initialize_latent(method='ddim',
                                    src_img=src_img,
                                    uc=uc,
                                    c=src_c,
                                    cfg_guidance=cfg_guidance)
        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="DDIM-edit")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, tgt_c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc
        
            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()


#############################

if __name__ == "__main__":
    # print all list of solvers
    print(f"Possble solvers: {[x for x in __SOLVER__.keys()]}")
    
