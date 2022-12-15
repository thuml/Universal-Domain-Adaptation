

export CUDA_VISIBLE_DEVICES=1


lrs='1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6'

seeds='1234 2134 3412 4132'

seeds='1234'

for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli.py --config configs/nlp/udanli-clinc-opda.yaml --lr $lr --seed $seed
    done
done
