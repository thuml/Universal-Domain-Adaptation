

export CUDA_VISIBLE_DEVICES=1


seeds='1234 2134 3412 4132'


lrs='5e-5 1e-5 5e-6'
# lrs='1e-3 5e-4 1e-4'


# CDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udalm.py --config configs/nlp/udalm-trec-cda.yaml --lr $lr --seed $seed
    done
done

