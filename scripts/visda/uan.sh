

export CUDA_VISIBLE_DEVICES=0


lrs='5e-3 1e-3 5e-4'


for lr in $lrs; do
    python uan.py --config configs/uan-visda-train.yaml --lr $lr
done