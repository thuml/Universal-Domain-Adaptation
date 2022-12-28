

export CUDA_VISIBLE_DEVICES=2


seeds='1234 2134 3412 4132'


lr='1e-4'



# CDA
for seed in  $seeds; do
    python nlp/udalm_mlm.py --config configs/nlp/udalm-trec-cda.yaml --lr $lr --seed $seed
done

