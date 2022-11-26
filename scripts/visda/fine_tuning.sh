

export CUDA_VISIBLE_DEVICES=0


lrs='1e-2 5e-3 1e-3 5e-4'
lrs='5e-3 1e-3 5e-4'

for lr in $lrs; do
    python fine_tuning.py --config configs/fine_tuning-visda-train.yaml --lr $lr --step 0.05
done

