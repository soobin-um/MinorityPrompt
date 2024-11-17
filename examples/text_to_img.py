import argparse
from pathlib import Path

from munch import munchify
from torchvision.utils import save_image

from latent_diffusion import get_solver
from latent_sdxl import get_solver as get_solver_sdxl
from utils.callback_util import ComposeCallback
from utils.log_util import create_workdir, set_seed

import os


def main():
    parser = argparse.ArgumentParser(description="Latent Diffusion")
    parser.add_argument("--workdir", type=Path, default="examples/workdir/t2i")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--null_prompt", type=str, default="")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--cfg_guidance", type=float, default=7.5)
    parser.add_argument("--method", type=str, default='ddim')
    parser.add_argument("--model", type=str, default='sd15', choices=["sd15", "sd20", "sdxl", "sdxl_lightning"])
    parser.add_argument("--NFE", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # Prompt optimization
    parser.add_argument("--prompt_opt", action='store_true')
    parser.add_argument("--p_ratio", type=float, default=0.75)
    parser.add_argument("--p_opt_iter", type=int, default=10)
    parser.add_argument("--p_opt_lr", type=float, default=1e-2)
    parser.add_argument("--t_lo", type=float, default=0.9)
    parser.add_argument("--placeholder_string", type=str, default="*_0")
    parser.add_argument("--num_opt_tokens", type=int, default=1)
    parser.add_argument("--init_type", type=str, default="default")
    parser.add_argument("--init_word", type=str, default="")
    parser.add_argument("--init_gau_scale", type=float, default=1.0)
    parser.add_argument("--dynamic_pr", action='store_true')
    parser.add_argument("--base_prompt_after_popt", action='store_true')
    parser.add_argument("--inter_rate", type=int, default=1)
    parser.add_argument("--lr_decay_rate", type=float, default=0.0)
    parser.add_argument("--init_rand_vocab", action='store_true')
    parser.add_argument("--sg_lambda", type=float, default=1.0)
    parser.add_argument("--placeholder_position", type=str, default="end", choices=["end", "start"])

    args = parser.parse_args()
    popt_kwargs = {
        "prompt_opt": args.prompt_opt,
        "p_ratio": args.p_ratio,
        "p_opt_iter": args.p_opt_iter,
        "p_opt_lr": args.p_opt_lr,
        "t_lo": args.t_lo,
        "placeholder_string": args.placeholder_string,
        "num_opt_tokens": args.num_opt_tokens,
        "init_type": args.init_type,
        "init_word": args.init_word,
        "init_gau_scale": args.init_gau_scale,
        "dynamic_pr": args.dynamic_pr,
        "base_prompt_after_popt": args.base_prompt_after_popt,
        "inter_rate": args.inter_rate,
        "lr_decay_rate": args.lr_decay_rate,
        "init_rand_vocab": args.init_rand_vocab,
        "sg_lambda": args.sg_lambda,
        "placeholder_position": args.placeholder_position,
        "popt_diverse": False,
    }
    print(popt_kwargs)

    set_seed(args.seed)
    create_workdir(args.workdir)

    solver_config = munchify({'num_sampling': args.NFE })
    callback = ComposeCallback(workdir=args.workdir,
                               frequency=1,
                               callbacks=["draw_noisy", 'draw_tweedie'])
    # callback = None

    foldername = f'{args.model}/{args.method}-N={args.NFE}-cfgw={args.cfg_guidance}/{(args.prompt).replace(" ", "_")}/seed={args.seed}'
    os.makedirs(args.workdir.joinpath(f'result/{foldername}'), exist_ok=True)
    filename = f"{foldername}/base"
    if args.prompt_opt:
        filename = f"{foldername}/popt-tlo={args.t_lo}-pr={args.p_ratio}-N={args.p_opt_iter}-lr={args.p_opt_lr}-decay={args.lr_decay_rate}-init={args.init_type}-iw={args.init_word}-ig={args.init_gau_scale}-inter-rate={args.inter_rate}-Nt={args.num_opt_tokens}-php={args.placeholder_position}"
    if args.dynamic_pr:
        filename += "-dynamic_pr"
    if args.base_prompt_after_popt:
        filename += "-bpap"
    if args.init_rand_vocab:
        filename += "-rand-vocab"
    if args.null_prompt != "":
        filename += f'-null={(args.null_prompt).replace(" ", "_")}'
        
    if os.path.exists(args.workdir.joinpath(f'result/{filename}.png')):
        print(f"File {args.workdir.joinpath(f'result/{filename}.png')} already exists, skipping...")
        return


    if args.model == "sdxl" or args.model == "sdxl_lightning":
        if args.model == 'sdxl':
            solver = get_solver_sdxl(args.method,
                                    solver_config=solver_config,
                                    device=args.device)
        else:
            light_model_ckpt = f"ckpt/sdxl_lightning_{args.NFE}step_unet.safetensors"
            print(f"Using light model checkpoint: {light_model_ckpt}")
            solver = get_solver_sdxl(args.method,
                                    solver_config=solver_config,
                                    device=args.device,
                                    light_model_ckpt=light_model_ckpt)

        result = solver.sample(prompt1=[args.null_prompt, args.prompt],
                                prompt2=[args.null_prompt, args.prompt],
                                cfg_guidance=args.cfg_guidance,
                                target_size=(1024, 1024),
                                callback_fn=callback,
                                popt_kwargs=popt_kwargs)

    else:
        # model_key = "runwayml/stable-diffusion-v1-5" if args.model == "sd15" else "stabilityai/stable-diffusion-2-base"
        model_key = "botp/stable-diffusion-v1-5" if args.model == "sd15" else "stabilityai/stable-diffusion-2-base"
        solver = get_solver(args.method,
                            solver_config=solver_config,
                            model_key=model_key,
                            device=args.device)
        result = solver.sample(prompt=[args.null_prompt, args.prompt],
                               cfg_guidance=args.cfg_guidance,
                               callback_fn=callback,
                               popt_kwargs=popt_kwargs)

    
    save_image(result, args.workdir.joinpath(f'result/{filename}.png'), normalize=True)

if __name__ == "__main__":
    main()
