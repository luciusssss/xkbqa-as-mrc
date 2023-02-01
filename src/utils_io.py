import codecs
import json

def ave(x): return sum(x)/len(x)
def codecs_out(x): return codecs.open(x, 'w', 'utf-8')
def codecs_in(x): return codecs.open(x, 'r', 'utf-8')


def json_load(x): return json.load(codecs.open(x, 'r', 'utf-8'))
def json_loadl(x): return [json.loads(t) for t in codecs_in(x)]
def json_dump(d, p): return json.dump(d, codecs.open(
    p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def json_dumps(d): return json.dumps(d, indent=2, ensure_ascii=False)