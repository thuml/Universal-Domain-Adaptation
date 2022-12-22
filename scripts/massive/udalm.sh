

export CUDA_VISIBLE_DEVICES=3


seeds='1234 2134 3412 4132'


lrs='5e-5 1e-5 6e-6'


for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udalm.py --config configs/nlp/udalm-massive-opda.yaml --lr $lr --seed $seed
    done
done