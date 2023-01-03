

CUDA_VISIBLE_DEVICES=1


lrs='1e-2 5e-3 1e-3 5e-4'


## OFFICE ##
# 0 : amazon
# 1 : dslr
# 2 : webcam

# # DONE
# # amazon -> dslr
# for lr in $lrs; do
#     python uan.py --config configs/vision/office-train-amazon-dslr.yaml --lr $lr
# done

seeds='1234 2134 3412 4132'

lrs='1e-2 5e-3 1e-3 5e-4'
lrs='1e-5 5e-5'

for seed in  $seeds; do
    # amazon -> dslr
    for lr in $lrs; do
        python uan.py --config configs/vision/uan-office-train-amazon-dslr.yaml --lr $lr --seed $seed
    done

    # amazon -> webcam
    for lr in $lrs; do
        python uan.py --config configs/vision/uan-office-train-amazon-webcam.yaml --lr $lr --seed $seed
    done

    # dslr -> amazon
    for lr in $lrs; do
        python uan.py --config configs/vision/uan-office-train-dslr-amazon.yaml --lr $lr --seed $seed
    done

    # dslr -> webcam
    for lr in $lrs; do
        python uan.py --config configs/vision/uan-office-train-dslr-webcam.yaml --lr $lr --seed $seed
    done

;
    # webcam -> amazon
    for lr in $lrs; do
        python uan.py --config configs/vision/uan-office-train-webcam-amazon.yaml --lr $lr --seed $seed
    done


    # webcam -> dslr
    for lr in $lrs; do
        python uan.py --config configs/vision/uan-office-train-webcam-dslr.yaml --lr $lr --seed $seed
    done
done