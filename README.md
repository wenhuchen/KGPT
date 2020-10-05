# KGPT
Code and Data for EMNLP2020 Paper "KGPT: Knowledge-Grounded Pre-Training for Data-to-Text Generation"



## Download Preprocessed Dataset
```
wget https://kgpt.s3-us-west-2.amazonaws.com/dataset.zip
unzip dataset.zip
```

## Download Pre-trained KGPT model
```
https://kgpt.s3-us-west-2.amazonaws.com/models.zip
unzip models.zip
```

## Option1: Finetune on Full Set
Finetune the model on the full downstream dataset
### Sequence Encoder
  - WebNLG
    ```
      bash scripts/webnlg/finetune_sequence_webnlg_from_wikidata.sh 0 checkpoint_wikidata/checkpoint_sequence_head8_layer6_GPT2_maxfact12/model_ep14.pt
    ```
  - E2ENLG
    ```
      bash scripts/e2enlg/finetune_sequence_e2enlg_from_wikidata.sh 0 checkpoint_wikidata/checkpoint_sequence_head8_layer6_GPT2_maxfact12/model_ep14.pt
    ```
### Graph Encoder
  - WebNLG
    ```
      bash scripts/webnlg/finetune_graph_e2enlg_from_wikidata.sh 0 checkpoint_wikidata/checkpoint_sequence_head8_layer6_GPT2_maxfact12/model_ep14.pt
    ```
  - E2ENLG
    ```
      bash scripts/e2enlg/finetune_graph_e2enlg_from_wikidata.sh 0 checkpoint_wikidata/checkpoint_graph_head8_layer6_GPT2_maxfact12/model_ep14.pt
    ```

## Option2: Finetune for Few-Shot Leanring on 1% data.
- WebNLG
  ```
    scripts/webnlg/finetune_sequence_webnlg_from_wikidata_fewshot.sh 0 checkpoint_wikidata/checkpoint_sequence_head8_layer6_GPT2_maxfact12/model_ep14.pt 0.01
  ```
- E2ENLG
  ```
    bash scripts/e2enlg/finetune_sequence_e2enlg_from_wikidata.sh 0 checkpoint_wikidata/checkpoint_sequence_head8_layer6_GPT2_maxfact12/model_ep14.pt 0.01
  ```
