import re
import datefinder
import datetime
from nltk.tokenize import sent_tokenize
from thefuzz import fuzz, process

def find_all_span_candidates(passage, span_length, threshold):
    all_candidates = []
    tokenized_sents = sent_tokenize(passage)

    for sent in tokenized_sents:
        sent_tokens = sent.split()
        min_span_len = max(1, span_length - threshold)
        max_span_len = min(len(sent_tokens), span_length + threshold)

        for span_len in range(min_span_len, max_span_len+1):
            for start_pos in range(len(sent_tokens) - span_len + 1):
                cand_item = {
                    'start_pos': start_pos,
                    'span_len': span_len,
                    'text': ' '.join(sent_tokens[start_pos:start_pos+span_len])
                }
                if cand_item['text'][-2:] in [' ,']:
                    cand_item['span_len'] -= 2
                    cand_item['text'] = cand_item['text'][:-2]
                if cand_item['text'][:2] in [', ']:
                    cand_item['start_pos'] += 2
                    cand_item['span_len'] -= 2
                    cand_item['text'] = cand_item['text'][2:]
                if cand_item['text'][-2:] in [' .']:
                    cand_item['span_len'] -= 2
                    cand_item['text'] = cand_item['text'][:-2]
                if cand_item['text'][:2] in ['. ']:
                    cand_item['start_pos'] += 2
                    cand_item['span_len'] -= 2
                    cand_item['text'] = cand_item['text'][2:]
                    
                all_candidates.append(cand_item)
                
    return all_candidates


def fuzz_match(triple_item, passage):
    obj = triple_item['object']

    # exact match
    if obj.lower() in passage.lower():
        span_pos = passage.lower().rfind(obj.lower())
        span_text = passage[span_pos:span_pos+len(obj)]
        triple_item['text'] = span_text
        triple_item['answer_start'] = span_pos
        triple_item['score'] = 1.0
        return triple_item

    # if ans.lower() not in passage.lower():
    # deal with date first:
    if re.match(r'^\d{4}-\d{2}-\d{2}', obj):
        matches = datefinder.find_dates(passage, source=True)
        span_text = None
        span_pos = None
        for match in matches:
            if obj.startswith(datetime.date.isoformat(match[0])):
                span_text = match[1]
                span_pos = passage.lower().rfind(span_text.lower())
                span_text = passage[span_pos:span_pos+len(span_text)]
                break
        if span_text != None:
            triple_item['text'] = span_text
            triple_item['answer_start'] = span_pos
            triple_item['score'] = 1.0
            return triple_item


    all_span_candidates = find_all_span_candidates(passage, len(obj.split()), max(2, int(len(obj.split())/2)))
    # fuzz matching with thefuzz
    extract_res = process.extract(obj, [span['text'] for span in all_span_candidates], limit=5) 

    plausible_span_found = False
    top_span = None
    for matched_span in extract_res:
        obj_no_punc = re.sub(r'[^\w\s]', '', obj)
        span_no_punc = re.sub(r'[^\w\s]', '', matched_span[0])
        if len(set(obj_no_punc.split()).intersection(set(span_no_punc.split()))) == 0 or matched_span[1] <= 50 or matched_span[1] != extract_res[0][1]:
            continue
        else:
            plausible_span_found = True
            top_span = matched_span
            break
    
    if not plausible_span_found:
        triple_item['text'] = None
        triple_item['answer_start'] = None
        triple_item['score'] = 0.0
        return triple_item

    span_text = top_span[0]
    span_pos = passage.lower().rfind(span_text.lower())
    span_text = passage[span_pos:span_pos+len(span_text)]
    triple_item['text'] = span_text
    triple_item['answer_start'] = span_pos
    triple_item['score'] = top_span[1]/100
    
    return triple_item

if __name__ == '__main__':
    # here we provide an example of finding a entity
    # from the passage by fuzz matching

    passage = "Claude Debussy 's musical genre is piano and he has also contributed to Ariettes oubliées : V. Green ."
    triple_item = {
        "subject": "Claude Debussy",
        "object": "Ariettes oubliées: V. Green",
        "relation": "/music/artist/track_contributions /music/track_contribution/track",
        "text": "Ariettes oubliées : V. Green",
        "answer_start": 72,
        "score": 0.98
    }

            
    triple_item = fuzz_match(triple_item, passage)

    print(triple_item)
    '''
    output:
        triple_item = {
        "subject": "Claude Debussy",
        "object": "Ariettes oubliées: V. Green",
        "relation": "/music/artist/track_contributions /music/track_contribution/track",
        "text": "Ariettes oubliées : V. Green",
        "answer_start": 72,
        "score": 0.98
    }
    '''