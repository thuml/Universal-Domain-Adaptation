

export CUDA_VISIBLE_DEVICES=1


lrs='1e-2 5e-3 1e-3'
# lrs='5e-3 1e-4 5e-5'


## OFFICE-HOME ##
# 0 : art
# 1 : clipart
# 2 : product
# 3 : real world


# art -> clipart
for lr in $lrs; do
    python ovanet.py --config configs/ovanet-office_home-train-art-clipart.yaml --lr $lr
done

# art -> product
for lr in $lrs; do
    python ovanet.py --config configs/ovanet-office_home-train-art-product.yaml --lr $lr
done

# art -> real world
for lr in $lrs; do
    python ovanet.py --config configs/ovanet-office_home-train-art-realworld.yaml --lr $lr
done
