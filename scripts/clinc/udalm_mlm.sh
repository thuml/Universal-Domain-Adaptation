

export CUDA_VISIBLE_DEVICES=2


seeds='1234 2134 3412 4132'


lr='1e-4'


# # OPDA 
# for seed in  $seeds; do
#     python nlp/udalm_mlm.py --config configs/nlp/udalm-clinc-opda.yaml --lr $lr --seed $seed
# done


# # CDA 
# for seed in  $seeds; do
#     python nlp/udalm_mlm.py --config configs/nlp/udalm-clinc-cda.yaml --lr $lr --seed $seed
# done


# ODA 
for seed in  $seeds; do
    python nlp/udalm_mlm_oda.py --config configs/nlp/udalm-clinc-oda.yaml --lr $lr --seed $seed
done