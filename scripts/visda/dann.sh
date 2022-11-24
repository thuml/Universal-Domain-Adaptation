

export CUDA_VISIBLE_DEVICES=1


lrs='1e-2 5e-3 1e-3'
lrs='5e-3 1e-4 5e-5'

for lr in $lrs; do
    python dann.py --config configs/dann-visda-train.yaml --lr $lr --step 0.05
done

