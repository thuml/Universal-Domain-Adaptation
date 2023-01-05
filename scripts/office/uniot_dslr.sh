export CUDA_VISIBLE_DEVICES=3


lrs='1e-1 5e-2 1e-2 5e-3'

## OFFICE ##
# 0 : amazon
# 1 : dslr
# 2 : webcam


# seeds='1234 2134 3412 4132'
seeds='1234 2134 3412'

for seed in  $seeds; do
    # # dslr -> amazon
    # for lr in $lrs; do
    #     python uniot.py --config configs/vision/uniot-office-train-dslr-amazon.yaml --lr $lr --seed $seed
    # done

    # dslr -> webcam
    for lr in $lrs; do
        python uniot.py --config configs/vision/uniot-office-train-dslr-webcam.yaml --lr $lr --seed $seed
    done
done


sh scripts/office/uniot_webcam.sh