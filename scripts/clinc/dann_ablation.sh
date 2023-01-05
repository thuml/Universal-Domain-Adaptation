

export CUDA_VISIBLE_DEVICES=2


lrs='1e-4 5e-5 1e-5 5e-6 1e-6'

seeds='1234 2134 3412'

# # Common class = 2
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/dann.py --config configs/nlp/ablation/dann-clinc-2.yaml --lr $lr --seed $seed
#     done
# done

# Common class = 4 (main experiment)

# Common class = 6
for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/dann.py --config configs/nlp/ablation/dann-clinc-6.yaml --lr $lr --seed $seed
    done
done

# # Common class = 8
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/dann.py --config configs/nlp/ablation/dann-clinc-8.yaml --lr $lr --seed $seed
#     done
# done



# sh scripts/clinc/dann.sh