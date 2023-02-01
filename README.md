# Cross-lingual Question Answering over Knowledge Base as Reading Comprehension
Data and code for 'Cross-lingual Question Answering over Knowledge Base as Reading Comprehension', Findings of EACL 2023

## WebQSP-zh
WebQSP-zh consists of 4,737 questions mannually translated from WebQSP by Chinese native speakers.

See `data/raw/webqsp-zh` for details.

## xKBQA-as-MRC
We propose a novel approach for xKBQA in a reading comprehension paradigm. 
We demonstrate its effectiveness on two datasets, WebQSP-zh and QALD-M.
Here we describe how to run the code.

### Dependency
* Python >= 3.8
* 

### Data Preparation
1. MRC datasets for finetuning
* SQuAD 1.1: Put SQuAD 1.1 in `data/external/squad` if you want to training the model from scratch.
* Existing xMRC datasets: We combine the existing xMRC data for each langugae for finetuning. Download the zip from **[url]** and put the json files in `data/external/xmrc`.

2. KB-to-text generation results
We provide the processed subset of KBs and the results of KB-to-text Generation.
Download the zip from **[url]** and put the json files in `data/processed/kb2text`.

3. Multilingual model of SentenceTransformers
Download the zip from **[url]** and put the folder `paraphrase-multilingual-mpnet-base-v2` in `data/external`


### Experiments on WebQSP-zh
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

You can also download the output data from **[url]**.

### Step 3: xMRC
**Inference**
Download our checkpoint from **[url]** and put it in `models`.

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

**Training from scratch**
Run the following code:
```
cd src
chmod +x train_webqsp-zh.sh
./train_webqsp-zh.sh
```


### Experiments on QALD-M
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

You can also download the output data from **[url]**.

### Step 3: xMRC
**Inference**
Download our checkpoint (trained with a combination of the data in all the languages) from **[url]** and put it in `models`.


Run the following code:
```bash
cd src
python run_squad_choice.py \
--model_type roberta \
--model_name_or_path ../models/xlm-roberta-large_squad_combined-all \
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

