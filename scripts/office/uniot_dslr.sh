export CUDA_VISIBLE_DEVICES=3


lrs='1e-2 5e-3 1e-3 5e-4'
# lrs='1e-3 5e-4 1e-2 5e-3'

## OFFICE ##
# 0 : amazon
# 1 : dslr
# 2 : webcam


# dslr -> amazon
for lr in $lrs; do
    python uniot.py --config configs/vision/uniot-office-train-dslr-amazon.yaml --lr $lr
done

# dslr -> webcam
for lr in $lrs; do
    python uniot.py --config configs/vision/uniot-office-train-dslr-webcam.yaml --lr $lr
done
