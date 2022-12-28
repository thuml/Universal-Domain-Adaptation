

export CUDA_VISIBLE_DEVICES=0



# # fine-tuning
# method='fine_tuning'

# seed='1234'
# lr='5e-5'
# python nlp/evaluation_ood.py --config configs/nlp/fine_tuning-trec-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='2134'
# lr='5e-5'
# python nlp/evaluation_ood.py --config configs/nlp/fine_tuning-trec-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='3412'
# lr='5e-5'
# python nlp/evaluation_ood.py --config configs/nlp/fine_tuning-trec-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='4132'
# lr='5e-6'
# python nlp/evaluation_ood.py --config configs/nlp/fine_tuning-trec-opda.yaml --method_name $method --lr $lr --seed $seed


# dann
method='dann'

seed='1234'
lr='1e-4'
python nlp/evaluation_ood.py --config configs/nlp/dann-trec-opda.yaml --method_name $method --lr $lr --seed $seed

seed='2134'
lr='5e-5'
python nlp/evaluation_ood.py --config configs/nlp/dann-trec-opda.yaml --method_name $method --lr $lr --seed $seed

seed='3412'
lr='1e-5'
python nlp/evaluation_ood.py --config configs/nlp/dann-trec-opda.yaml --method_name $method --lr $lr --seed $seed

seed='4132'
lr='5e-6'
python nlp/evaluation_ood.py --config configs/nlp/dann-trec-opda.yaml --method_name $method --lr $lr --seed $seed


# # uan
# method='uan'

# seed='1234'
# lr='1e-3'
# python nlp/evaluation_ood.py --config configs/nlp/uan-trec-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='2134'
# lr='1e-4'
# python nlp/evaluation_ood.py --config configs/nlp/uan-trec-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='3412'
# lr='5e-5'
# python nlp/evaluation_ood.py --config configs/nlp/uan-trec-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='4132'
# lr='1e-5'
# python nlp/evaluation_ood.py --config configs/nlp/uan-trec-opda.yaml --method_name $method --lr $lr --seed $seed


# # cmu
# method='cmu'

# seed='1234'
# lr='1e-4'
# python nlp/evaluation_ood.py --config configs/nlp/cmu-trec-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='2134'
# lr='5e-5'
# python nlp/evaluation_ood.py --config configs/nlp/cmu-trec-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='3412'
# lr='1e-4'
# python nlp/evaluation_ood.py --config configs/nlp/cmu-trec-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='4132'
# lr='5e-5'
# python nlp/evaluation_ood.py --config configs/nlp/cmu-trec-opda.yaml --method_name $method --lr $lr --seed $seed


# # udalm
# method='udalm'

# seed='1234'
# lr='5e-5'
# python nlp/evaluation_ood.py --config configs/nlp/udalm-trec-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='2134'
# lr='5e-5'
# python nlp/evaluation_ood.py --config configs/nlp/udalm-trec-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='3412'
# lr='5e-5'
# python nlp/evaluation_ood.py --config configs/nlp/udalm-trec-opda.yaml --method_name $method --lr $lr --seed $seed

# seed='4132'
# lr='5e-5'
# python nlp/evaluation_ood.py --config configs/nlp/udalm-trec-opda.yaml --method_name $method --lr $lr --seed $seed
