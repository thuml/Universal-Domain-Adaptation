

export CUDA_VISIBLE_DEVICES=1


lrs='1e-2 5e-3 1e-3 5e-4'


for lr in $lrs; do
    python uan.py --config configs/uan-visda-train.yaml --lr $lr
done