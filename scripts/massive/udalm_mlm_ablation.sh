

export CUDA_VISIBLE_DEVICES=2


seeds='1234 2134 3412'

lr='1e-4'

# common class = 2
for seed in  $seeds; do
    python nlp/udalm_mlm.py --config configs/nlp/ablation/udalm-massive-2.yaml --lr $lr --seed $seed
done

# common class = 6
for seed in  $seeds; do
    python nlp/udalm_mlm.py --config configs/nlp/ablation/udalm-massive-6.yaml --lr $lr --seed $seed
done

# common class = 8
for seed in  $seeds; do
    python nlp/udalm_mlm.py --config configs/nlp/ablation/udalm-massive-8.yaml --lr $lr --seed $seed
done

