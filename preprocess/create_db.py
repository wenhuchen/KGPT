import json
import sys
from Database import MyDatabase

with open('properties.json', 'r') as f:
    properties = json.load(f)

db = MyDatabase('wikidata.db', connect_each=False)
db.create(True, 'Entities', [('id', 'TEXT'), ('label', 'TEXT'), ('description', 'TEXT')])
db.create(True, 'Properties', [('property_id', 'TEXT'), ('label', 'TEXT'), ('description', 'TEXT')])
db.create(True, 'Relations', [('head_id', 'TEXT'), ('property', 'TEXT'), ('target', 'TEXT')])

with open('summarized.txt', 'r') as f:
    for i, line in enumerate(f):
        data = json.loads(line.strip())
        if data[0].startswith('Q'):
            db.insert('Entities', [(data[0], data[1], data[2])])
        elif data[0].startswith('P'):
            db.insert('Properties', [(data[0], data[1], data[2])])
        else:
            print(data[0])
            
        for r, val in data[4]:
            if r in properties and data[0].startswith('Q'):
                db.insert('Relations', [(data[0], r, json.dumps(val))])
            
        sys.stdout.write('finished {}/{} \r'.format(i, 70900911))

db.commit()
db.close()
