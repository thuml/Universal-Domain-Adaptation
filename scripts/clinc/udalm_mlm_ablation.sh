

export CUDA_VISIBLE_DEVICES=2


seeds='1234 2134 3412'


lr='1e-4'

# num common class = 2
for seed in  $seeds; do
    python nlp/udalm_mlm.py --config configs/nlp/ablation/udalm-clinc-2.yaml --lr $lr --seed $seed
done

# num common class = 6
for seed in  $seeds; do
    python nlp/udalm_mlm.py --config configs/nlp/ablation/udalm-clinc-6.yaml --lr $lr --seed $seed
done

# num common class = 8
for seed in  $seeds; do
    python nlp/udalm_mlm.py --config configs/nlp/ablation/udalm-clinc-8.yaml --lr $lr --seed $seed
done
