

export CUDA_VISIBLE_DEVICES=2



# # fine-tuning
# method='fine_tuning'

# seed='1234'
# lr='1e-5'
# threshold='0.345'
# python nlp/evaluation.py --config configs/nlp/fine_tuning-massive-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='2134'
# lr='1e-4'
# threshold='0.51'
# python nlp/evaluation.py --config configs/nlp/fine_tuning-massive-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='3412'
# lr='1e-5'
# threshold='0.545'
# python nlp/evaluation.py --config configs/nlp/fine_tuning-massive-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='4132'
# lr='5e-5'
# threshold='0.59'
# python nlp/evaluation.py --config configs/nlp/fine_tuning-massive-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold




# # dann
# method='dann'

# seed='1234'
# lr='5e-5'
# threshold='0.47'
# python nlp/evaluation.py --config configs/nlp/dann-massive-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='2134'
# lr='1e-4'
# threshold='0.625'
# python nlp/evaluation.py --config configs/nlp/dann-massive-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='3412'
# lr='1e-4'
# threshold='0.81'
# python nlp/evaluation.py --config configs/nlp/dann-massive-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='4132'
# lr='5e-5'
# threshold='0.385'
# python nlp/evaluation.py --config configs/nlp/dann-massive-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold




# # uan
# method='uan'

# seed='1234'
# lr='1e-4'
# threshold='-0.08'
# python nlp/evaluation.py --config configs/nlp/uan-clinc-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='2134'
# lr='1e-4'
# threshold='0.5'
# python nlp/evaluation.py --config configs/nlp/uan-clinc-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='3412'
# lr='1e-4'
# threshold='0.2'
# python nlp/evaluation.py --config configs/nlp/uan-clinc-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='4132'
# lr='1e-5'
# threshold='0.26'
# python nlp/evaluation.py --config configs/nlp/uan-clinc-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold




# # cmu
# method='cmu'

# seed='1234'
# lr='1e-4'
# threshold='0.5'
# python nlp/evaluation.py --config configs/nlp/cmu-massive-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='2134'
# lr='5e-5'
# threshold='0.5'
# python nlp/evaluation.py --config configs/nlp/cmu-massive-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='3412'
# lr='1e-4'
# threshold='0.5'
# python nlp/evaluation.py --config configs/nlp/cmu-massive-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='4132'
# lr='5e-5'
# threshold='0.5'
# python nlp/evaluation.py --config configs/nlp/cmu-massive-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold



# # ovanet
# method='ovanet'

# seed='1234'
# lr='1e-3'
# python nlp/evaluation.py --config configs/nlp/ovanet-massive-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='2134'
# lr='5e-5'
# python nlp/evaluation.py --config configs/nlp/ovanet-massive-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='3412'
# lr='5e-6'
# python nlp/evaluation.py --config configs/nlp/ovanet-massive-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='4132'
# lr='5e-6'
# python nlp/evaluation.py --config configs/nlp/ovanet-massive-opda.yaml --method_name $method --lr $lr --seed $seed



# udalm
method='udalm'

# seed='1234'
# lr='5e-5'
# python nlp/evaluation.py --config configs/nlp/udalm-massive-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='2134'
# lr='5e-5'
# python nlp/evaluation.py --config configs/nlp/udalm-massive-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='3412'
# lr='5e-5'
# python nlp/evaluation.py --config configs/nlp/udalm-massive-opda.yaml --method_name $method --lr $lr --seed $seed

seed='4132'
lr='5e-5'
python nlp/evaluation.py --config configs/nlp/udalm-massive-opda.yaml --method_name $method --lr $lr --seed $seed

