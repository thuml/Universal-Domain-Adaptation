

export CUDA_VISIBLE_DEVICES=0


seeds='1234 2134 3412 4132'

num_samples='2'

# OPDA
for seed in  $seeds; do
    for num_sample in $num_samples; do
        python nlp/udanli_generate_dataset.py --config configs/nlp/udanli/udanli-clinc-opda_0.1.yaml --seed $seed --num_nli_sample $num_sample
    done
done

# # CDA
# for seed in  $seeds; do
#     for num_sample in $num_samples; do
#         python nlp/udanli_generate_dataset.py --config configs/nlp/udanli-clinc-cda_0.7.yaml --seed $seed --num_nli_sample $num_sample
#     done
# done

# # ODA
# for seed in  $seeds; do
#     for num_sample in $num_samples; do
#         python nlp/udanli_generate_dataset_oda.py --config configs/nlp/udanli/udanli-clinc-oda_0.8.yaml --seed $seed --num_nli_sample $num_sample
#     done
# done
