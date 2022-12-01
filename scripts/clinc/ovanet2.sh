

export CUDA_VISIBLE_DEVICES=1


lrs='5e-5 1e-5 5e-6 1e-6'
lrs='5e-2 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6 5e-7 1e-7'

seeds='1234 2134 3412 4132'
seeds='3412'


for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/ovanet.py --config configs/nlp/ovanet-clinc-opda.yaml --lr $lr --seed $seed
    done
done
