

export CUDA_VISIBLE_DEVICES=0


seeds='1234 2134 3412'

lr='1e-4'

# common class = 2
for seed in  $seeds; do
    python nlp/udalm_mlm.py --config configs/nlp/ablation/udalm-massive-2.yaml --lr $lr --seed $seed
done

# common class = 4
for seed in  $seeds; do
    python nlp/udalm_mlm.py --config configs/nlp/ablation/udalm-massive-4.yaml --lr $lr --seed $seed
done

# common class = 6
for seed in  $seeds; do
    python nlp/udalm_mlm.py --config configs/nlp/ablation/udalm-massive-6.yaml --lr $lr --seed $seed
done

# common class = 10
for seed in  $seeds; do
    python nlp/udalm_mlm.py --config configs/nlp/ablation/udalm-massive-10.yaml --lr $lr --seed $seed
done

# common class = 12
for seed in  $seeds; do
    python nlp/udalm_mlm.py --config configs/nlp/ablation/udalm-massive-12.yaml --lr $lr --seed $seed
done

# common class = 14
for seed in  $seeds; do
    python nlp/udalm_mlm.py --config configs/nlp/ablation/udalm-massive-14.yaml --lr $lr --seed $seed
done

# common class = 16
for seed in  $seeds; do
    python nlp/udalm_mlm.py --config configs/nlp/ablation/udalm-massive-16.yaml --lr $lr --seed $seed
done





