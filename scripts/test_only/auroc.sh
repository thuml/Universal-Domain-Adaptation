
export CUDA_VISIBLE_DEVICES=1

## OFFICE ##
# 0 : amazon
# 1 : dslr
# 2 : webcam


method='ovanet'
lr='1e-4'
python auroc.py --config configs/ovanet-office-train-amazon-webcam.yaml --lr $lr --method $method 

lr='1e-2'
python auroc.py --config configs/ovanet-office-train-dslr-amazon.yaml --lr $lr --method $method 

lr='1e-3'
python auroc.py --config configs/ovanet-office-train-dslr-webcam.yaml --lr $lr --method $method 


# method='uan'
# # amazon -> webcam
# lr='1e-3'
# threshold='-0.5'
# python auroc.py --config configs/uan-office-train-dslr-webcam.yaml --lr $lr --method $method --threshold $threshold --min_threshold -1.0 --max_threshold 1.0
