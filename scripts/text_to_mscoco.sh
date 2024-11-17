gpu=0

model="sdxl_lightning"
method="ddim_lightning"
cfg_w=1.0
NFE=4
num_samples=5000

t_lo=0.0
N=3
lr=0.001
init_type="word"
init_word="cool"

conda activate mprompt
CUDA_VISIBLE_DEVICES=$gpu python -m examples.text_to_mscoco \
--method "$method" --cfg_guidance $cfg_w --model "$model" --NFE $NFE --num_samples $num_samples \
--prompt_opt --p_opt_lr $lr --p_opt_iter $N --t_lo $t_lo --init_type "$init_type" --init_word "$init_word" --dynamic_pr