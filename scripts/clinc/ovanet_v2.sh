

export CUDA_VISIBLE_DEVICES=2


lrs='1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6'

seeds='1234 2134 3412 4132'


for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/ovanet_v2.py --config configs/nlp/ovanet-clinc-opda.yaml --lr $lr --seed $seed
    done
done
