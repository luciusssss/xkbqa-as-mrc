
import time
import datetime
from collections import defaultdict
from collections import Counter
import collections
import functools
from tqdm import tqdm
import numpy as np
import random
import codecs
import copy
import json
import re
import os
from pprint import pprint
import sys
sys.path.append('..')
from tqdm import tqdm
import pickle as pk
import argparse

from sentence_transformers import SentenceTransformer, util

from utils_io import *

def filter_sentence(question, candidates, bert_embedder, topk=10):
    ret = []

    if len(question) == 0 or len(candidates) == 0:
        return ret
    
    query_embedding = bert_embedder.encode(question, convert_to_tensor=True)
    candidate_embeddings = bert_embedder.encode(candidates, convert_to_tensor=True)

    hits = util.semantic_search(query_embedding, candidate_embeddings, top_k=topk)
    hits = hits[0]      #Get the hits for the first query
    for hit in hits:
        ret.append({
            'text': candidates[hit['corpus_id']],
            'match_score': float(hit['score'])
        })
        # print(candidates[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))

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

def remove_underscore(s):
    s = [t for t in s.split('_') if len(t)]
    return ' '.join(s)

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
        "--test_data_path",
        default=None,
        type=str,
        required=True,
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


    kb2text_data = pk.load(open(args.kb2text_path, 'rb'))
    print('kb2text loaded')

    bert_embedder = SentenceTransformer(args.sent_trans_path)
    print('SentenceTransformer loaded')

    qald_data = json_load(args.test_data_path) 

    max_length = args.max_passage_length


    for lang in ['fa', 'de', 'ro', 'hi', 'it', 'ru', 'fr', 'nl', 'es', 'pt', 'pt_BR']:
        squad_data = {
                'version': lang,
                'data': []
            }
        print("start", lang)

        sentence_ranking_path = os.path.join(args.output_path, 'all_selected_sentences_%s.json' % (lang))
        if os.path.exists(sentence_ranking_path):
            all_selected_sentences_for_item = json_load(sentence_ranking_path)
        else:
            all_selected_sentences_for_item = {}

        for item in tqdm(qald_data):
            if lang not in item['question']:
                continue
            # sentence filtering
            if item['qid'] in all_selected_sentences_for_item:
                all_selected_sentences = all_selected_sentences_for_item[item['qid']]
            else:
                all_sentences = []
                for topic_ent in item['topic_entities']:
                    if topic_ent not in kb2text_data:
                        # print(topic_ent, " not in kb2text data")
                        continue
                    for sent_item in kb2text_data[topic_ent]:
                        if sent_item['answers'][0]['text'] == None:
                            continue
                        all_sentences.append(sent_item['passage'])

                
                selected_sentences = filter_sentence(item['question'][lang], all_sentences, bert_embedder, topk=len(all_sentences))

                all_selected_sentences = []
                for sent_item in selected_sentences:
                    sent_item['el_key'] = topic_ent
                    all_selected_sentences.append(sent_item)
                
                all_selected_sentences_for_item[item['qid']] = all_selected_sentences

            # passage composition
            filtered_sent = []
            total_length = 0
            for sent_item in all_selected_sentences:
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
            for topic_ent in item['topic_entities']:
                if topic_ent not in kb2text_data:
                    # print(topic_ent, " not in kb2text data")
                    continue
                for kb2text_item in kb2text_data[topic_ent]:   
                    if kb2text_item['passage'] not in filtered_sent:
                        continue
                    if kb2text_item['passage'] in used_sent:
                        continue
                    used_sent.add(kb2text_item['passage'])
                    filtered_kb2text_item[kb2text_item['passage']] = kb2text_item


            squad_passage = ""
            answers = []
            golden_ans = set([a.lower() for a in item['answers']])
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
            
            answers = sorted(answers, key=lambda x: x['answer_start'])
            squad_data['data'].append({
                'paragraphs': [{
                    "cid": item['qid'],
                    "context": squad_passage,
                    "qas": [
                        {
                            "id": item['qid'],
                            "question": item['question'][lang],
                            "answers": answers,
                            "candidates": candidates
                        }
                    ]
                }]
            })
        
        if not os.path.exists(sentence_ranking_path):
            json_dump(all_selected_sentences_for_item, sentence_ranking_path)
        json_dump(squad_data, os.path.join(args.output_path, f"test_{lang}_squad.json"))
 
