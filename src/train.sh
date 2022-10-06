# python train_bi-encoder_mnrl.py --model_name ../../sentence-transformers/examples/training/ms_marco/output/train_bi-encoder-mnrl-distilbert-base-uncased-margin_3.0-2022-09-15_06-13-41
# python train_bi-encoder_mnrl.py --model_name  distilbert-base-uncased 
#python train_bi-encoder_mnrl.py --model_name  distilbert-base-uncased  --lambda_uni 0.0 > train_noadv_30iter_from_scratch.log 
python train_bi-encoder_mnrl.py --model_name  distilbert-base-uncased  --lambda_uni 0.1 --uni_q 0.05 >> train_adv_0.1_30iter_decay0.95_uniq0.05_from_scratch.log 