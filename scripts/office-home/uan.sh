

export CUDA_VISIBLE_DEVICES=2


lrs='1e-2 5e-3 1e-3 5e-4'

## OFFICE-HOME ##
# 0 : art
# 1 : clipart
# 2 : product
# 3 : real world

# art -> clipart
for lr in $lrs; do
    python uan.py --config configs/uan-office_home-train-art-clipart.yaml --lr $lr
done

# art -> product
for lr in $lrs; do
    python uan.py --config configs/uan-office_home-train-art-product.yaml --lr $lr
done

# art -> real world
for lr in $lrs; do
    python uan.py --config configs/uan-office_home-train-art-realworld.yaml --lr $lr
done