

export CUDA_VISIBLE_DEVICES=1


lrs='1e-2 5e-3 1e-3 5e-4'

seeds='2134 3412 4132'


lrs='1e-5'
seeds='2134'

for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/cmu.py --config configs/nlp/cmu-clinc-opda.yaml --lr $lr --seed $seed
    done
done
