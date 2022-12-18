

export CUDA_VISIBLE_DEVICES=3


lrs='5e-5'
seeds='1234'

for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v4.py --config configs/nlp/udanli-clinc-opda_0.1.yaml --lr $lr --seed $seed
    done
done
