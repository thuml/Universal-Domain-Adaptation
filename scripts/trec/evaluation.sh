

export CUDA_VISIBLE_DEVICES=2



# # fine-tuning
# method='fine_tuning'

# seed='1234'
# lr='5e-5'
# threshold='0.90'
# python nlp/evaluation.py --config configs/nlp/fine_tuning-trec-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='2134'
# lr='5e-5'
# threshold='0.63'
# python nlp/evaluation.py --config configs/nlp/fine_tuning-trec-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='3412'
# lr='5e-5'
# threshold='0.69'
# python nlp/evaluation.py --config configs/nlp/fine_tuning-trec-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='4132'
# lr='5e-6'
# threshold='0.5'
# python nlp/evaluation.py --config configs/nlp/fine_tuning-trec-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold




# dann
method='dann'

seed='1234'
lr='1e-4'
threshold='0.33'
python nlp/evaluation.py --config configs/nlp/dann-trec-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

seed='2134'
lr='5e-5'
threshold='0.27'
python nlp/evaluation.py --config configs/nlp/dann-trec-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

seed='3412'
lr='1e-5'
threshold='0.38'
python nlp/evaluation.py --config configs/nlp/dann-trec-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

seed='4132'
lr='5e-6'
threshold='0.44'
python nlp/evaluation.py --config configs/nlp/dann-trec-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold




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
# python nlp/evaluation.py --config configs/nlp/cmu-trec-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='2134'
# lr='5e-5'
# threshold='0.5'
# python nlp/evaluation.py --config configs/nlp/cmu-trec-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='3412'
# lr='1e-4'
# threshold='0.5'
# python nlp/evaluation.py --config configs/nlp/cmu-trec-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold

# seed='4132'
# lr='5e-5'
# threshold='0.5'
# python nlp/evaluation.py --config configs/nlp/cmu-trec-opda.yaml --method_name $method --lr $lr --seed $seed --threshold $threshold



# # ovanet
# method='ovanet'

# seed='1234'
# lr='1e-3'
# python nlp/evaluation.py --config configs/nlp/ovanet-trec-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='2134'
# lr='5e-5'
# python nlp/evaluation.py --config configs/nlp/ovanet-trec-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='3412'
# lr='5e-6'
# python nlp/evaluation.py --config configs/nlp/ovanet-trec-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='4132'
# lr='5e-6'
# python nlp/evaluation.py --config configs/nlp/ovanet-trec-opda.yaml --method_name $method --lr $lr --seed $seed



# # udalm
# method='udalm'

# seed='1234'
# lr='5e-5'
# python nlp/evaluation.py --config configs/nlp/udalm-trec-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='2134'
# lr='5e-5'
# python nlp/evaluation.py --config configs/nlp/udalm-trec-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='3412'
# lr='5e-5'
# python nlp/evaluation.py --config configs/nlp/udalm-trec-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='4132'
# lr='5e-5'
# python nlp/evaluation.py --config configs/nlp/udalm-trec-opda.yaml --method_name $method --lr $lr --seed $seed

