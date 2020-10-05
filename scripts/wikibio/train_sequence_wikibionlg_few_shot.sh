FOLDER='/data/wenhu/entity2text'
GPUS=$1
percent=$2

CUDA_VISIBLE_DEVICES=${GPUS} python code/run.py --batch_size 64 --dataset wikibionlg \
 --tokenizer_dir ${FOLDER}/GPT2_tokenizer/ --max_enc_len 425 --max_dec_len 72 --num_workers 8 --epochs 400 \
 --printing_steps 200 --save_every_n_epochs 200 --encoder sequence --option few_shot --percent ${percent}
