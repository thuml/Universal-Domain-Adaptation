

export CUDA_VISIBLE_DEVICES=0


lrs='1e-5'
seeds='1234'

for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9-1_plot.py --config configs/nlp/udanli/udanli-clinc-opda_0.8.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done
