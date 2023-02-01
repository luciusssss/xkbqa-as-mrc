from tqdm import tqdm
import numpy as np
import random
import codecs
import copy
import json
import os
from pprint import pprint
import sys
import argparse
import re
import pickle as pk
sys.path.append('.')
sys.path.append('..')

from utils_io import *

from sentence_transformers import SentenceTransformer, util

# filter the duplicated cvt node in the kb2text result
def deduplicate_cvt_nodes(item_list):
    ret = []
    i = 0
    while i < len(item_list):
        item = item_list[i]
        i += 1
        if 'cvt_hashing' not in item:
            ret.append(item)
            continue
        cvt_nodes = [item]
        while i < len(item_list) and 'cvt_hashing' in item_list[i] and item_list[i]['cvt_hashing'] == item['cvt_hashing']:
            cvt_nodes.append(item_list[i])
            i += 1
        cvt_nodes_and_scores = [(node, sum([ans['score'] for ans in node['answers']])) for node in cvt_nodes]
        cvt_nodes_and_scores = sorted(cvt_nodes_and_scores, key=lambda x:x[1], reverse=True)
        ret.append(cvt_nodes_and_scores[0][0])
    return ret

def check_ans_in_passages(q_item, kb2text_items, selected_sentences):
    if len(selected_sentences) == 0:
        return []
    if isinstance(selected_sentences[0], dict):
        selected_sentences = set([t['text'] for t in selected_sentences])
    
    golden_ans = set([str(ans).lower() for ans in q_item['answers']])
    ans_cand = set()

    for kb2text_item in kb2text_items:
        if kb2text_item['passage'] not in selected_sentences:
            continue
        for ans in kb2text_item['answers']:
            if ans['answer_start'] != None and ans['score'] != 0.0:
                ans_cand.add(str(ans['object'].lower()))
    
    return list(golden_ans.intersection(ans_cand))


def select_sentences(question, candidates, embedder, topk=10):
    query_embedding = embedder.encode(question, convert_to_tensor=True)
    candidate_embeddings = embedder.encode(candidates, convert_to_tensor=True)

    hits = util.semantic_search(query_embedding, candidate_embeddings, top_k=topk)
    hits = hits[0]     
    ret = []
    for hit in hits:
        ret.append({
            'text': candidates[hit['corpus_id']],
            'score': float(hit['score'])
        })

    return ret

def get_sentence_rankings_for_dataset(dataset, kb2text_data, embedder):
    ret = {}
    for item in tqdm(dataset):
        freebase_key = item['freebase_key']
        if freebase_key not in kb2text_data or len(kb2text_data[freebase_key]) == 0:
            continue

        deduplicated_nodes = deduplicate_cvt_nodes(kb2text_data[freebase_key])
        
        selected_sentences = select_sentences(item['question'], [node['passage'] for node in deduplicated_nodes], embedder, topk=len(kb2text_data[freebase_key]))

        ret[item['qid']] = selected_sentences
    
    return ret


def compose_passages(dataset, sentence_ranking_results, mode='train', max_length=750):
    output_squad_data = {
        'data': []
    }

    for q_item in tqdm(dataset):
        qid = q_item['qid']
        fb_key = q_item['freebase_key']
        if fb_key not in kb2text_data or qid not in sentence_ranking_results:
            if mode == 'train': # discard this instance
                pass
            elif mode == 'test':
                output_squad_data['data'].append({
                    'paragraphs': [{
                        "cid": qid,
                        "context": "no text", 
                        "qas": [
                            {
                                "id": qid,
                                "question": q_item['question'],
                                "answers": [],
                                "candidates": [ 
                                    {
                                        'text': "no text",
                                        'answer_start': 0
                                    }
                                ]
                            }
                        ]
                    }]
                })
            continue
        
        golden_ans = set([a.lower() for a in q_item['answers']])

        filtered_sent = []
        total_length = 0
        for sent_item in sentence_ranking_results[qid]:
            if sent_item['text'] in filtered_sent:
                continue
            sent_length = len(sent_item['text'].split())
            if total_length + sent_length > max_length:
                break
            else:
                filtered_sent.append(sent_item['text'])
                total_length += sent_length
        
        filtered_kb2text_item = {}
        used_sent = set()
        for kb2text_item in kb2text_data[fb_key]:
            if kb2text_item['passage'] not in filtered_sent:
                continue
            if kb2text_item['passage'] in used_sent:
                continue
            used_sent.add(kb2text_item['passage'])
            filtered_kb2text_item[kb2text_item['passage']] = kb2text_item

        squad_passage = ""
        answers = []
        golden_ans = set([a.lower() for a in q_item['answers']])
        ans_cand = set()
        candidates = []

        
        for sent in filtered_sent:
            kb2text_item = filtered_kb2text_item[sent]
            for ans in kb2text_item['answers']:
                if ans['answer_start'] != None and ans['score'] != 0.0:
                    ans_cand.add(str(ans['object']).lower())
                    cand = copy.deepcopy(ans)
                    cand['answer_start'] += len(squad_passage)
                    candidates.append(cand)
                    if ans['object'].lower() in golden_ans:
                        ans = copy.deepcopy(ans)
                        ans['answer_start'] += len(squad_passage)
                        answers.append(ans)
            squad_passage += kb2text_item['passage'] + ' '
        
        if mode == 'train' and len(answers) == 0:
            continue # discard this instance
        
        answers = sorted(answers, key=lambda x: x['answer_start'])
        output_squad_data['data'].append({
            'paragraphs': [{
                "cid": qid,
                "context": squad_passage,
                "qas": [
                    {
                        "id": qid,
                        "question": q_item['question'],
                        "answers": answers,
                        "candidates": candidates
                    }
                ]
            }]
        })

    return output_squad_data

# =======   =======   =======   =======   =======   =======

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kb2text_path",
        default=None,
        type=str,
        required=True,
        help="Path to the file of KB-to-text results",
    )
    parser.add_argument(
        "--sent_trans_path",
        default=None,
        type=str,
        required=True,
        help="Path to the sentence-transformer model",
    )
    parser.add_argument(
        "--train_data_path",
        default=None,
        type=str,
        required=False,
        help="Path to the training data",
    )
    parser.add_argument(
        "--test_data_path",
        default=None,
        type=str,
        required=False,
        help="Path to the test data",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=True,
        help="Path to the outputs",
    )

    parser.add_argument(
        "--max_passage_length",
        default=750,
        type=int,
        required=False,
        help="Maximum length of the output passages",
    )


    args = parser.parse_args()


    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    
    kb2text_data = json_load(args.kb2text_path)
    print("kb2text_data loaded")
    embedder = SentenceTransformer(args.sent_trans_path)
    print("SentenceTransformer loaded")

    if args.train_data_path:
        train_data = json_load(args.train_data_path)
    if args.test_data_path:
        test_data = json_load(args.test_data_path)
    
    if args.train_data_path:
        print("start to construct passages for the training set")

        # rank all the candidate sentence
        sentence_ranking_results = get_sentence_rankings_for_dataset(train_data, kb2text_data, embedder)

        # construct a passage for each instance
        output_data = compose_passages(train_data, sentence_ranking_results, mode='train', max_length=args.max_passage_length)
        
        json_dump(output_data, os.path.join(args.output_path, 'train_squad.json'))
    
    if args.test_data_path:
        print("start to construct passages for the test set")

        # rank all the candidate sentence
        sentence_ranking_results = get_sentence_rankings_for_dataset(test_data, kb2text_data, embedder)
        output_data = compose_passages(test_data, sentence_ranking_results, mode='test', max_length=args.max_passage_length)

        # construct a passage for each instance
        json_dump(output_data, os.path.join(args.output_path, 'test_squad.json'))

