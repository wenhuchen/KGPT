import json
import gzip
import multiprocessing
from multiprocessing import Pool
import os

def process_claims(claim):
    results = []
    for k, vs in claim.items():
        for v in vs:
            d = v['mainsnak']
            if 'id' not in d['datatype'] and 'datavalue' in d:
                d = d['datavalue']
                if isinstance(d['value'], dict) and 'language' in d['value']:
                    if d['value']['language'] == 'en':
                        results.append((k, d))
                else:
                        results.append((k, d))                    
    return results

def func(income_file):
    fw = open('summarized.txt', 'w')
    with gzip.open(income_file, 'rt') as f:
        for i, line in enumerate(f):
            if line.startswith('{'):
                line = line.strip().rstrip(',')
                data = json.loads(line)
                if 'en' in data['labels']:
                    if 'en' in data['descriptions']:
                        description = data['descriptions']['en']['value']
                    else:
                        description = ''
                        
                    if 'en' in data['aliases']:
                        alias = [_['value'] for _ in data['aliases']['en']]
                    else:
                        alias = []
                    
                    content = json.dumps((data['id'], data['labels']['en']['value'], description, 
                                          alias, process_claims(data['claims'])))
                    
                    fw.write(content)
                    fw.write('\n')

                else:
                    pass
                    
func('latest-all.json.gz')
