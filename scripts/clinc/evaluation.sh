

export CUDA_VISIBLE_DEVICES=1



# # fine-tuning
# method='fine_tuning'

# seed='1234'
# lr='5e-5'
# threshold='0.53'
# python nlp/evaluation.py --config configs/nlp/fine_tuning-clinc-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='2134'
# lr='1e-6'
# threshold='0.52'
# python nlp/evaluation.py --config configs/nlp/fine_tuning-clinc-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='3412'
# lr='1e-6'
# threshold='0.48'
# python nlp/evaluation.py --config configs/nlp/fine_tuning-clinc-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='4132'
# lr='1e-4'
# threshold='0.57'
# python nlp/evaluation.py --config configs/nlp/fine_tuning-clinc-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold




# # dann
# method='dann'

# seed='1234'
# lr='5e-6'
# threshold='0.35'
# python nlp/evaluation.py --config configs/nlp/dann-clinc-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='2134'
# lr='5e-6'
# threshold='0.225'
# python nlp/evaluation.py --config configs/nlp/dann-clinc-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='3412'
# lr='5e-6'
# threshold='0.215'
# python nlp/evaluation.py --config configs/nlp/dann-clinc-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='4132'
# lr='5e-5'
# threshold='0.51'
# python nlp/evaluation.py --config configs/nlp/dann-clinc-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold




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
# lr='1e-5'
# threshold='0.5'
# python nlp/evaluation.py --config configs/nlp/cmu-clinc-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='2134'
# lr='1e-5'
# threshold='0.5'
# python nlp/evaluation.py --config configs/nlp/cmu-clinc-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='3412'
# lr='1e-5'
# threshold='0.5'
# python nlp/evaluation.py --config configs/nlp/cmu-clinc-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='4132'
# lr='1e-5'
# threshold='0.5'
# python nlp/evaluation.py --config configs/nlp/dann-clinc-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold




# # ovanet
# method='ovanet'

# seed='1234'
# lr='5e-5'
# python nlp/evaluation.py --config configs/nlp/ovanet-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

# seed='2134'
# lr='1e-6'
# python nlp/evaluation.py --config configs/nlp/ovanet-clinc-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='3412'
# lr='1e-6'
# python nlp/evaluation.py --config configs/nlp/ovanet-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

# seed='4132'
# lr='1e-5'
# python nlp/evaluation.py --config configs/nlp/ovanet-clinc-opda.yaml --method_name $method --lr $lr --seed $seed



# udalm
method='udalm'

seed='1234'
lr='5e-5'
python nlp/evaluation.py --config configs/nlp/udalm-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

seed='2134'
lr='5e-5'
python nlp/evaluation.py --config configs/nlp/udalm-clinc-opda.yaml --method_name $method --lr $lr --seed $seed

seed='3412'
lr='5e-5'
python nlp/evaluation.py --config configs/nlp/udalm-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

seed='4132'
lr='5e-5'
python nlp/evaluation.py --config configs/nlp/udalm-clinc-opda.yaml --method_name $method --lr $lr --seed $seed

