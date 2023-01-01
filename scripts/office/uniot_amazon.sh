export CUDA_VISIBLE_DEVICES=2


lrs='1e-2 5e-3 1e-3 5e-4'

## OFFICE ##
# 0 : amazon
# 1 : dslr
# 2 : webcam

# amazon -> dslr
for lr in $lrs; do
    python uniot.py --config configs/vision/uniot-office-train-amazon-dslr.yaml --lr $lr
done

# amazon -> webcam
for lr in $lrs; do
    python uniot.py --config configs/vision/uniot-office-train-amazon-webcam.yaml --lr $lr
done

# # dslr -> amazon
# for lr in $lrs; do
#     python uniot.py --config configs/vision/uniot-office-train-dslr-amazon.yaml --lr $lr
# done

# # dslr -> webcam
# for lr in $lrs; do
#     python uniot.py --config configs/vision/uniot-office-train-dslr-webcam.yaml --lr $lr
# done
