import nltk
from nltk.corpus import stopwords
import json
import sys
import copy
from Database import MyDatabase
import os

db = MyDatabase('wikidata.db', connect_each=False)
stopwords = set(stopwords.words('english'))
examples = []
with open('examples-v3-intermediate.json', 'r') as f:
    for i, line in enumerate(f):
        entry = json.loads(line.strip())
        examples.append(entry)
        sys.stdout.write('finish {} lines \r'.format(i))
print("Finish loading the intermediate json file")

if os.path.exists('knowledge-v3.json'):
    with open('knowledge-v3.json', 'r') as f:
        cache = json.load(f)
    print("Loading existing knowledge graph")
else:
    cache = {}
    for d in examples:
        kb_ids = d['kblinks']
        kb_ids += [d['title_kb_id']]
        
        for kb_id in kb_ids:
            if kb_id and kb_id not in cache:
                tmp = db.fetch('Entities', 'id', kb_id)
                if len(tmp) > 0:
                    cache[kb_id] = [tmp[0][1], tmp[0][2]]
                    relations = db.fetch('Relations', 'head_id', kb_id)
                    lexicalized = []
                    for _, r, v in relations:
                        r = db.fetch('Properties', 'property_id', r)[0][1]
                        if 'coordinate' not in r and 'image' not in r:
                            v = json.loads(v)
                            if v['type'] == 'monolingualtext':
                                v = v['value']['text']
                            elif v['type'] == 'string':
                                v = v['value']
                            elif v['type'] == 'time':
                                v = v['value']['time'].split('T')[0]
                            elif v['type'] == 'quantity':
                                v = v['value']['amount']
                            elif v['type'] == 'wikibase-entityid':
                                tail_kb_id = v['value']['id']
                                tmp = db.fetch('Entities', 'id', tail_kb_id)
                                if len(tmp) > 0:
                                    v = tmp[0][1]
                                else:
                                    #print(tail_kb_id)
                                    v = None
                            else:
                                #print(r, v['type'])
                                v = None
                                raise NotImplementedError

                            if v:
                                lexicalized.append((r, v))

                    cache[kb_id].append(lexicalized)
                else:
                    cache[kb_id] = ['', '', []]

    db.close()

    with open('knowledge-v3.json', 'w') as f:
        json.dump(cache, f, indent=2)

scored_examples = []
for i, line in enumerate(examples):
    if len(line['kblinks']) != len(line['text']):
        print("drop an example", line)
        continue
    kb_links = [_ for _ in line['kblinks'] if _ is not None]
    mapping = {_:j for j, _ in enumerate(line['kblinks'])}
    whole_sent = []

    if len(kb_links) <= 0.70 * len(line['text']) and len(kb_links) <=8:
        if line['title_kb_id'] in cache:
            kbs = cache[line['title_kb_id']]
            sent = '{} [SEP] {}'.format(kbs[0], kbs[1])
            tmps = []
            for _ in kbs[-1]:
                tmps.append('[PRED] {} [OBJ] {}'.format(_[0], _[1]))
            tmps = ' '.join(tmps)
            sent = sent + ' ' + tmps
            whole_sent.append(sent)
            if line['text'][0] in ['It', 'He', 'She']:
                line['text'][0] = kbs[0]

        for entity in kb_links:
            kbs = cache[entity]
            sent = '{} [SEP] {}'.format(kbs[0], kbs[1])
            tmps = []
            for _ in kbs[-1]:
                tmps.append('[PRED] {} [OBJ] {}'.format(_[0], _[1]))
            tmps = ' '.join(tmps)
            sent = sent + ' ' + tmps
            whole_sent.append(sent)
        
        intersections = []
        for k, entity in enumerate(kb_links):
            tmp_text = copy.copy(line['text'])
            del tmp_text[mapping[entity]]
            tmp_text = ' '.join(tmp_text)
            
            tmp_whole_sent = whole_sent[k + 1]

            text = set(tmp_text.lower().split(' ')) - stopwords
            knowledge_backend = set(tmp_whole_sent.lower().split(' ')) - stopwords
        
            intersect = text & knowledge_backend
            intersections.append(len(intersect))
        
        full_text = set((' '.join(line['text'])).split(' ')) - stopwords
        
        normalized_score = sum(intersections) / len(full_text)
        del line['hyperlinks']
        line['score'] = normalized_score
        scored_examples.append(line)
        sys.stdout.write('finished {}/{} \r'.format(i, len(examples)))

scored_examples = sorted(scored_examples, key=lambda x:x['score'], reverse=True)

with open('examples-v3.json', 'w') as f:
    json.dump(scored_examples, f)
