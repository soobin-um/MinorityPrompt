import argparse
from pathlib import Path
import os

from munch import munchify
from torchvision.utils import save_image

from latent_diffusion import get_solver
from latent_sdxl import get_solver as get_solver_sdxl
from utils.callback_util import ComposeCallback
from utils.log_util import create_workdir, set_seed

from itertools import islice
from tqdm import tqdm

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def main():
    parser = argparse.ArgumentParser(description="Latent Diffusion")
    parser.add_argument("--workdir", type=Path, default="examples/workdir/mscoco")
    parser.add_argument('--prompt_dir', type=Path, default=Path('examples/assets/coco_v2.txt'))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--null_prompt", type=str, default="")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--cfg_guidance", type=float, default=7.5)
    parser.add_argument("--method", type=str, default='ddim')
    parser.add_argument("--model", type=str, default='sd15', choices=["sd15", "sd20", "sdxl", "sdxl_lightning"])
    parser.add_argument("--NFE", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--b_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--resume_from", type=int, default=0)

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
    parser.add_argument("--popt_diverse", action='store_true')
    
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
        "popt_diverse": args.popt_diverse,
    }
    print(popt_kwargs)

    set_seed(args.seed)
    create_workdir(args.workdir)
    
    # load prompt
    text_list = []
    with open(args.prompt_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:  # Only add non-empty lines
                text_list.append(stripped_line)
    text_list = text_list[args.resume_from : (args.resume_from + args.num_samples)] # Test for 10k MS-COCO validation
    if args.popt_diverse:
        # assert (args.model == "sd15") or (args.model == "sd20"), "Diverse prompt optimization is only available for sd15 and sd20 model that support batch generation."
        # repeat each prompt same with the number of b_size
        text_list = [text for text in text_list for _ in range(args.b_size)]
        args.num_samples = args.b_size * args.num_samples
        assert len(text_list) == args.num_samples, "Number of samples should be equal to the length of text_list."

    solver_config = munchify({'num_sampling': args.NFE })
    callback = ComposeCallback(workdir=args.workdir,
                               frequency=1,
                               callbacks=["draw_noisy", 'draw_tweedie'])
    # callback = None

    basename = f'{args.workdir}/result/{args.model}/{args.method}-N={args.NFE}-cfgw={args.cfg_guidance}/seed={args.seed}'
    foldername = f"{basename}/base"
    if args.prompt_opt:
        foldername = f"{basename}/popt-tlo={args.t_lo}-pr={args.p_ratio}-N={args.p_opt_iter}-lr={args.p_opt_lr}-decay={args.lr_decay_rate}-init={args.init_type}-iw={args.init_word}-ig={args.init_gau_scale}-inter-rate={args.inter_rate}-Nt={args.num_opt_tokens}-php={args.placeholder_position}"
    if args.popt_diverse:
        foldername = f"{basename}/popt-diverse-tlo={args.t_lo}-N={args.p_opt_iter}-lr={args.p_opt_lr}-decay={args.lr_decay_rate}-init={args.init_type}-iw={args.init_word}-ig={args.init_gau_scale}-inter-rate={args.inter_rate}-Nt={args.num_opt_tokens}-php={args.placeholder_position}"
    if args.dynamic_pr:
        foldername += "-dynamic_pr"
    if args.base_prompt_after_popt:
        foldername += "-bpap"
    if args.init_rand_vocab:
        foldername += "-rand-vocab"
    foldername += f"-NFE={args.NFE}-b_size={args.b_size}-{args.resume_from}-{args.resume_from + args.num_samples}"
    os.makedirs(foldername, exist_ok=True)

    # Count PNG files in foldername
    png_files = list(Path(foldername).glob("*.png"))
    num_png_files = len(png_files)

    # Check if number of PNG files exceeds args.num_samples
    if num_png_files >= args.num_samples:
        print("Number of PNG files exceeds num_samples. Ending the code.")
        return

    if args.model == "sdxl" or args.model == "sdxl_lightning":
        if args.model == "sdxl":
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

        img_count = args.resume_from
        for i, text in enumerate(tqdm(text_list, desc='Batch')):
            print(f'Processing {i+1}/{len(text_list)}: {text}')

            result = solver.sample(prompt1=[args.null_prompt, text],
                                    prompt2=[args.null_prompt, text],
                                    cfg_guidance=args.cfg_guidance,
                                    target_size=(1024, 1024),
                                    callback_fn=callback,
                                    popt_kwargs=popt_kwargs)
            save_image(result, f'{foldername}/{str(img_count).zfill(5)}.png', normalize=True)
            img_count += 1
    else:
        # model_key = "runwayml/stable-diffusion-v1-5" if args.model == "sd15" else "stabilityai/stable-diffusion-2-base"
        model_key = "botp/stable-diffusion-v1-5" if args.model == "sd15" else "stabilityai/stable-diffusion-2-base"
        solver = get_solver(args.method,
                            solver_config=solver_config,
                            model_key=model_key,
                            device=args.device)
        text_list_batched = list(chunk(text_list, args.b_size))
        img_count = args.resume_from
        for i, prompts in enumerate(tqdm(text_list_batched, desc='Batch')):
            prompts = list(prompts) if isinstance(prompts, tuple) else prompts
            null_prompts = len(prompts) * [args.null_prompt]
            import time
            start = time.time()
            result = solver.batch_sample(prompts=prompts,
                                         null_prompts=null_prompts,
                                         cfg_guidance=args.cfg_guidance,
                                         callback_fn=callback,
                                         popt_kwargs=popt_kwargs)
            print(f"Time taken for batch: {time.time() - start}")
            for idx, img in enumerate(result):
                save_image(img, f'{foldername}/{str(img_count).zfill(5)}.png', normalize=True)
                img_count += 1
                                        

if __name__ == "__main__":
    main()