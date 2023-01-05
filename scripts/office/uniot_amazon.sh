export CUDA_VISIBLE_DEVICES=2


lrs='1e-1 5e-2 1e-2 5e-3'

## OFFICE ##
# 0 : amazon
# 1 : dslr
# 2 : webcam

seeds='1234 2134 3412'

for seed in  $seeds; do
    # amazon -> dslr
    for lr in $lrs; do
        python uniot.py --config configs/vision/uniot-office-train-amazon-dslr.yaml --lr $lr --seed $seed
    done

    # # amazon -> webcam
    # for lr in $lrs; do
    #     python uniot.py --config configs/vision/uniot-office-train-amazon-webcam.yaml --lr $lr --seed $seed
    # done
done
