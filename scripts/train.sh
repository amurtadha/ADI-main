cd ../
 HF_DATASETS_OFFLINE=0 CUDA_VISIBLE_DEVICES=$1 python train.py --dataset $2 --train_sample $4 --pretrained_bert_name $3 --batch_size 16
