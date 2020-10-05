FOLDER=$(pwd)
GPUS=$1

CUDA_VISIBLE_DEVICES=${GPUS} python code/run.py --batch_size 112 --dataset wikidata \
 --knowledge_path ${FOLDER}/preprocess/knowledge-full.json --tokenizer_dir ${FOLDER}/GPT2_tokenizer/ \
 --max_enc_len 900 --max_dec_len 64 --epochs 20 --encoder graph --max_entity 12 --num_workers 8
