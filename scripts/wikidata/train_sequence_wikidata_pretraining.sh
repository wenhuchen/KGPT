FOLDER=$(pwd)
GPUS=$1

CUDA_VISIBLE_DEVICES=${GPUS} python code/run.py --batch_size 128  --dataset wikidata \
--tokenizer_dir ${FOLDER}/GPT2_tokenizer/ --max_enc_len 760 --max_dec_len 64 --epochs 20 \
--knowledge_path ${FOLDER}/preprocess/knowledge-full.json --encoder sequence --max_entity 12 --num_workers 8
