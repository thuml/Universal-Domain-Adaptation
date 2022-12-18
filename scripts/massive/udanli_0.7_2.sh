

export CUDA_VISIBLE_DEVICES=1


lrs='5e-5 1e-5 5e-6 1e-6'


seeds='1234 2134 3412 4132'
seeds='1234'


for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v5.py --config configs/nlp/udanli-massive-opda_0.7.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done
