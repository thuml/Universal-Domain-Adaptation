

export CUDA_VISIBLE_DEVICES=2


lrs='1e-5 5e-6 1e-6'

seeds='1234 2134 3412 4132'




for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v11.py --config configs/nlp/udanli_v11/udanli-clinc-opda_0.8_0.01.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done

for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v11.py --config configs/nlp/udanli_v11/udanli-clinc-opda_0.8_0.03.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done


for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v11.py --config configs/nlp/udanli_v11/udanli-clinc-opda_0.8_0.05.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done



for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v11.py --config configs/nlp/udanli_v11/udanli-clinc-opda_0.8_0.1.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done



for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v11.py --config configs/nlp/udanli_v11/udanli-clinc-opda_0.8_0.2.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done