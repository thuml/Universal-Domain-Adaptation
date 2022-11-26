
export CUDA_VISIBLE_DEVICES=1



# method='ovanet'
# lr='1e-4'
# python auroc.py --config configs/ovanet-visda-train.yaml --lr $lr --method $method 



# method='fine_tuning'
# lr='5e-4'
# threshold='0.15'
# python auroc.py --config configs/fine_tuning-visda-train.yaml --lr $lr --method $method --threshold $threshold --min_threshold 0 --max_threshold 1.0 --step 0.01



method='uan'
lr='1e-3'
threshold='-0.5'
python auroc.py --config configs/uan-visda-train.yaml --lr $lr --method $method --threshold $threshold --min_threshold -1.0 --max_threshold 1.0 --step 0.01


method='dann'
lr='5e-3'
threshold='0.25'
python auroc.py --config configs/dann-visda-train.yaml --lr $lr --method $method --threshold $threshold --min_threshold 0 --max_threshold 1.0 --step 0.001

