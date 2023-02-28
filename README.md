# Cross-Lingual Question Answering over Knowledge Base as Reading Comprehension
Data and code for *Cross-Lingual Question Answering over Knowledge Base as Reading Comprehension*, Findings of EACL 2023

Arxiv preprint: https://arxiv.org/abs/2302.13241

## Dataset: WebQSP-zh
We present WebQSP-zh, a new dataset for cross-lingual KBQA.
It consists of 4,737 questions mannually translated from WebQSP by Chinese native speakers.

See `data/raw/webqsp-zh/README.md` for details.

## Method: xKBQA-as-MRC
We propose a novel approach for xKBQA in a reading comprehension paradigm. 
We demonstrate its effectiveness on two datasets, WebQSP-zh and QALD-M.
Here we describe how to run the code.

### Dependency
* Python >= 3.8
* See other dependency in `requirements.txt`

### Data Preparation
1. MRC datasets for finetuning

* SQuAD 1.1: Put SQuAD 1.1 in `data/external/squad` if you want to training the model from scratch.
* Existing xMRC datasets: We aggregate the existing xMRC data in each language for finetuning. Download the zip from [link](https://pkueducn-my.sharepoint.com/:u:/g/personal/zhangchen1999_pku_edu_cn/EWPlwM9lCrlDpPbNLiCKqnUBK5SG8vb0hdaFo99N-THoDw?e=NwGxQ6) and put the json files in `data/external/xmrc`.

2. KB-to-text generation results

We provide the processed subset of KBs and the results of KB-to-text Generation.
Download the zip from [link](https://pkueducn-my.sharepoint.com/:u:/g/personal/zhangchen1999_pku_edu_cn/EQBpNiyGlSNOj5CHUPZThSYB2i80RjfGFEj0G1S22qfaWA?e=Ko49C8) and put the json/pickle files in `data/processed/kb2text`.

3. Multilingual model of SentenceTransformers

Download the zip from [link](https://pkueducn-my.sharepoint.com/:u:/g/personal/zhangchen1999_pku_edu_cn/EePEZTIam3pBlaQiHtz70xoBcATST6D766VPu8y17ezx1Q) and put the folder `paraphrase-multilingual-mpnet-base-v2` in `data/external`


### Supervised Experiments on WebQSP-zh
#### Step 1: KB-to-text Generation
For KB-to-text generation, we use BART-base version of JointGT finetuned on WebNLG.
One can try other SOTA data-to-text models for better generation quality. 
We also provide the code snippet for identifying entities from the generated sentences with fuzzy matching in `src/identify_span_by_fuzzy_matching.py`.  

We provide the processed subset of Freebase and the results of KB-to-text Generation in `data/external/kb2text/kb2text-results_freebase-for-webqsp-zh.json`.

#### Step 2: Passage Construction
Run `src/obtain_passages_webqsp-zh.py` to construct a passage for each question.

```bash
cd src
python obtain_passages_webqsp-zh.py \
    --kb2text_path ../data/processed/kb2text/kb2text-results_freebase-for-webqsp-zh.json \
    --sent_trans_path  ../data/external/paraphrase-multilingual-mpnet-base-v2 \
    --train_data_path ../data/raw/webqsp-zh/train.json \
    --test_data_path ../data/raw/webqsp-zh/test.json \
    --output_path ../data/processed/webqsp-zh_xkbqa-as-mrc 
```

You can also download the output data from [link](https://pkueducn-my.sharepoint.com/:u:/g/personal/zhangchen1999_pku_edu_cn/EZ6yrF1PTwxCju_NIqJRzzQBtJxebjqkmykVk6WfIRaTqw?e=acier9) and put the json files in `data/processed/webqsp-zh_xkbqa-as-mrc`.

#### Step 3: xMRC

* Inference
  
Download our checkpoint from [link](https://pkueducn-my.sharepoint.com/:u:/g/personal/zhangchen1999_pku_edu_cn/Efj1ZZjzOFpKkMgGll-WDWMBkkGO0rYaW0KuzT1kzFJwEw?e=MwSgNO) and put it in `models`.

Run the following code:
```bash
cd src
python run_squad_choice.py \
    --model_type roberta \
    --model_name_or_path ../models/xlm-roberta-large_squad-xmrc_webqsp-zh \
    --do_eval \
    --do_lower_case \
    --data_dir ../data/processed/webqsp-zh_xkbqa-as-mrc \
    --predict_file test_squad.json \
    --cache_dir ../models/xlm-roberta-large_squad-xmrc_webqsp-zh \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ../models/xlm-roberta-large_squad-xmrc_webqsp-zh \
    --overwrite_output_dir
```
The prediction results are in `models/xlm-roberta-large_squad-xmrc_webqsp-zh/post_processed_results.json`.

* Training from scratch
  
Run the following code:
```
cd src
chmod +x train_webqsp-zh.sh
./train_webqsp-zh.sh
```


### Zero-shot Experiments on QALD-M
#### Step 1: KB-to-text Generation

We provide the processed subset of DBPedia and the results of KB-to-text Generation in `data/external/kb2text/kb2text-results_dbpedia-for-qald-m.pk`.

#### Step 2: Passage Construction

Run `src/obtain_passages_qald-m.py` to construct a passage for each question.

```bash
cd src
python obtain_passages_qald-m.py \
--kb2text_path ../data/processed/kb2text/kb2text-results_dbpedia-for-qald-m.pk \
--sent_trans_path  ../data/external/paraphrase-multilingual-mpnet-base-v2 \
--test_data_path ../data/raw/qald-m/test.json \
--output_path ../data/processed/qald-m_xkbqa-as-mrc 
```

You can also download the output data from [link](https://pkueducn-my.sharepoint.com/:u:/g/personal/zhangchen1999_pku_edu_cn/EZ6yrF1PTwxCju_NIqJRzzQBtJxebjqkmykVk6WfIRaTqw?e=acier9) and put the json files in `data/processed/qald-m_xkbqa-as-mrc`.
#### Step 3: xMRC

* Inference
  
Download our checkpoint (trained with a combination of the data in all the languages) from [link](https://pkueducn-my.sharepoint.com/:u:/g/personal/zhangchen1999_pku_edu_cn/EU7Z-ebxOXROuMoV5UgP_rQBq_TXa2jYdbNtq3vOizeS5w?e=zPJehA) and put it in `models`.


Run the following code:
```bash
cd src
python run_squad_choice.py \
--model_type roberta \
--model_name_or_path ../models/xlm-roberta-large_squad_combined-all_qald-m \
--do_eval \
--do_lower_case \
--data_dir ../data/processed/qald-m_xkbqa-as-mrc \
--predict_file test_{lang}_squad.json \
--cache_dir ../models/xlm-roberta-large_squad-combined_qald_{lang} \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir ../models/xlm-roberta-large_squad-combined_qald_{lang} \
--overwrite_output_dir
```

Replace `{lang}` with the languages in `['fa', 'de', 'ro', 'hi', 'it', 'ru', 'fr', 'nl', 'es', 'pt', 'pt_BR']`.

The prediction results are in `models/xlm-roberta-large_squad-combined_qald_{lang}/post_processed_results.json`.

* Training from scratch

If you want training a model from scratch with xMRC data, you can follow the similar procedure in `src/train_webqsp-zh.sh`: First finetune the model on SQuAD, then on the xMRC datasets of the target language (`data/external/xmrc/combined_xmrc_{lang}.json`) or a combination of all the languages (`data/external/xmrc/combined_xmrc_all.json`).


## Troubleshooting
1. Error when running transformers 3.5.1
```
Traceback (most recent call last):
  File "run_squad_choice.py", line 32, in <module>
    from transformers import (
  File "~/anaconda3/envs/xkbqa/lib/python3.8/site-packages/transformers/__init__.py", line 626, in <module>
    from .trainer import Trainer
  File "~/anaconda3/envs/xkbqa/lib/python3.8/site-packages/transformers/trainer.py", line 69, in <module>
    from .trainer_pt_utils import (
  File "~/anaconda3/envs/xkbqa/lib/python3.8/site-packages/transformers/trainer_pt_utils.py", line 40, in <module>
    from torch.optim.lr_scheduler import SAVE_STATE_WARNING
ImportError: cannot import name 'SAVE_STATE_WARNING' from 'torch.optim.lr_scheduler' (~/anaconda3/envs/xkbqa/lib/python3.8/site-packages/torch/optim/lr_scheduler.py)
```
Replace this line of code `from torch.optim.lr_scheduler import SAVE_STATE_WARNING` with `SAVE_STATE_WARNING = ""`.

2. Please use only one GPU when running the code. We did not test the code with multiple GPUs, so there could be unexpected results under the setting of multi-GPU. 
