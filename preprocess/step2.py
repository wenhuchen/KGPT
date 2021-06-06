import json
from wikidata import queries
import urllib.parse
import sys
import time
import os
import subprocess

assert len(sys.argv) == 3

input_file = sys.argv[1]
resume = sys.argv[2]
assert resume in ["true", "false"]
if resume == 'false':
    mode = 'w+'
else:
    mode = 'a+'

with open('wiki-to-kbid.json', 'r') as f:
    mapping = json.load(f)
    
with open('wiki-to-kbid-request-cache.json', 'r') as f:
    cache = json.load(f)

done = set()
if resume == 'true':
    with open(input_file + '.intermediate.json', 'r') as f:
        for line in f:
            title = json.loads(line.strip())['title']
            done.add(title)
    print("already accomplishsed {} lines".format(len(done)))

fw = open(input_file + '.intermediate.json', mode)
string = subprocess.check_output(['wc', '-l', input_file])
total_line = string.decode('utf8').split(' ')[0]

no_request, request = 0, 0
succ, fail = 0, 0
start_time = time.time()
idx = 0
with open(input_file, 'r') as f:
    for line_id, line in enumerate(f):
        data = json.loads(line.strip())
        if data['title'] in done:
            continue
        
        if data['title'].startswith('List of') or data['title'].startswith('Category') :
            continue
            
        wiki_id = data['title'].replace(' ', '_')
        wiki_id = urllib.parse.quote(wiki_id)
        title_kb_id = mapping.get(wiki_id, None)
        
        for text, hyperlinks in data['text']:
            num_entity = len([_ for _ in hyperlinks if _ is not None])
            if num_entity < 2:
                continue
            
            if text[-1] != '.':
                continue
            
            kb_ids = []
            at_least_1 = False
            for hyperlink in hyperlinks:
                if hyperlink is not None:
                    if hyperlink in cache:
                        kb_id = cache[hyperlink]
                    else:
                        modified = urllib.parse.unquote(hyperlink)
                        modified = [_.capitalize() for _ in modified.split(' ')]
                        modified = '_'.join(modified)
                        modified = urllib.parse.quote(modified)
                        if modified in mapping:
                            no_request += 1
                            kb_id = mapping[modified]
                        else:
                            request += 1
                            kb_id = queries.map_wikipedia_id(urllib.parse.unquote(hyperlink))
                        
                        cache[hyperlink] = kb_id
                    
                    if kb_id is not None:
                        succ += 1
                        at_least_1 = True
                    else:
                        fail += 1
                    
                    kb_ids.append(kb_id)
                else:
                    kb_ids.append(None)
            
            if title_kb_id is not None and at_least_1:
                entry = json.dumps({'id': idx, 'url': data['url'], 'title': data['title'], 
                                    'title_kb_id': title_kb_id, 'text': text,
                                    'hyperlinks': hyperlinks, 'kblinks': kb_ids})
                fw.write(entry + '\n')
                idx += 1
            
        sys.stdout.write("finished {}/{}; request/no_request = {}/{} ;succ/fail = {}/{} used time = {} \r"\
                         .format(line_id, total_line, request, no_request, succ, fail, time.time() - start_time))
        
with open('wiki-to-kbid-request-cache.json', 'w') as f:
    json.dump(cache, f)
