

export CUDA_VISIBLE_DEVICES=0

## OFFICE ##
# 0 : amazon
# 1 : dslr
# 2 : webcam

# source : amazon (0)
# target : dslr  (1)

seeds='2134 3412 4132'
lrs='1e-2 5e-3 1e-3'

# # dslr -> amazon
# lr='1e-3'
# seed='3142'
# python fine_tuning.py --config configs/vision/fine_tuning-office-train-dslr-amazon.yaml --lr $lr --seed $seed


# lrs='1e-2 5e-3 1e-3'

# seed='3142'
# # dslr -> webcam
# for lr in $lrs; do
#     python fine_tuning.py --config configs/vision/fine_tuning-office-train-dslr-webcam.yaml --lr $lr --seed $seed
# done

# seeds='1234 3142'
# for seed in seeds; do
#     # webcam -> amazon
#     for lr in $lrs; do
#         python fine_tuning.py --config configs/vision/fine_tuning-office-train-webcam-amazon.yaml --lr $lr --seed $seed
#     done
# done

seeds='1234 2134'
for seed in  $seeds; do
    # webcam -> dslr
    for lr in $lrs; do
        python fine_tuning.py --config configs/vision/fine_tuning-office-train-webcam-dslr.yaml --lr $lr --seed $seed
    done
done


# for seed in  $seeds; do

    # # amazon -> dslr
    # for lr in $lrs; do
    #     python fine_tuning.py --config configs/vision/fine_tuning-office-train-amazon-dslr.yaml --lr $lr --seed $seed
    # done

    # # amazon -> webcam
    # for lr in $lrs; do
    #     python fine_tuning.py --config configs/vision/fine_tuning-office-train-amazon-webcam.yaml --lr $lr --seed $seed
    # done

    # # dslr -> amazon
    # for lr in $lrs; do
    #     python fine_tuning.py --config configs/vision/fine_tuning-office-train-dslr-amazon.yaml --lr $lr --seed $seed
    # done

    # # dslr -> webcam
    # for lr in $lrs; do
    #     python fine_tuning.py --config configs/vision/fine_tuning-office-train-dslr-webcam.yaml --lr $lr --seed $seed
    # done

    # # webcam -> amazon
    # for lr in $lrs; do
    #     python fine_tuning.py --config configs/vision/fine_tuning-office-train-webcam-amazon.yaml --lr $lr --seed $seed
    # done

    # # webcam -> dslr
    # for lr in $lrs; do
    #     python fine_tuning.py --config configs/vision/fine_tuning-office-train-webcam-dslr.yaml --lr $lr --seed $seed
    # done
# done
