

export CUDA_VISIBLE_DEVICES=1


lrs='1e-4 5e-5 1e-5 5e-6 1e-6'

seeds='1234 2134 3412 4132'



# # OPDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/dann.py --config configs/nlp/dann-clinc-opda.yaml --lr $lr --seed $seed
#     done
# done

# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/dann.py --config configs/nlp/dann-clinc-cda.yaml --lr $lr --seed $seed
#     done
# done


# CDA
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/dann_oda.py --config configs/nlp/dann-clinc-oda.yaml --lr $lr --seed $seed
    done
done

sh scripts/clinc/uan.sh

