

export CUDA_VISIBLE_DEVICES=1

## OFFICE ##
# 0 : amazon
# 1 : dslr
# 2 : webcam

# source : amazon (0)
# target : dslr  (1)


seeds='3412'
lrs='1e-2 5e-3 1e-3'
for seed in  $seeds; do
    # webcam -> dslr
    for lr in $lrs; do
        python fine_tuning.py --config configs/vision/fine_tuning-office-train-webcam-dslr.yaml --lr $lr --seed $seed
    done
done

