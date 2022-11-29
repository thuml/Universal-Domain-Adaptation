

export CUDA_VISIBLE_DEVICES=2


lrs='1e-2 5e-3 1e-3 5e-4'

seeds='2134 3412 4132'


lrs='1e-2'
seeds='2134'

for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/fine_tuning.py --config configs/nlp/fine_tuning-clinc-opda.yaml --lr $lr --seed $seed
    done
done
