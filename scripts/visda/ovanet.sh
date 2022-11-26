

export CUDA_VISIBLE_DEVICES=2


# lrs='1e-2 5e-3 1e-3'
lrs='5e-5 1e-5'

for lr in $lrs; do
    python ovanet.py --config configs/ovanet-visda-train.yaml --lr $lr
done