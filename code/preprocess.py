import torch
from torch import nn
from transformers import BertModel,  BertTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, XLNetTokenizer, CTRLTokenizer
import sys
import json
from types import SimpleNamespace

if sys.argv[1] == 'build-bert':
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    tokenizer.add_tokens(['[ENT]', '[PRED]', '[SUB]', '[TRIPLE]', '[EOS]', '[SOS]'])
    tokenizer.save_pretrained('tokenizer/')

    model = BertModel.from_pretrained('bert-base-cased')
    model.resize_token_embeddings(len(tokenizer))
    embedding = model.embeddings.word_embeddings.weight
    torch.save(embedding, 'tokenizer/embedding.bin')

    knowledge_config = {'max_entity_embeddings': 8, 'max_triple_embeddings': 20, 'max_position_embeddings': 1024, 'vocab_size': len(tokenizer), 
                        'pad_token_id': tokenizer.pad_token_id, 'hidden_size': model.config.n_embd, 'hidden_dropout_prob': 0.1, 'layer_norm_eps': 1e-12, 
                        'sos_token_id': tokenizer.encode('[SOS]')[0], 'eos_token_id': tokenizer.encode('[EOS]')[0]}
    #model.save_pretrained('./')
    with open('tokenizer/knowledge_config.json', 'w') as f:
        json.dump(knowledge_config, f, indent=2)

if sys.argv[1] == 'build-gpt2':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    tokenizer.add_tokens(['[ENT]', '[PRED]', '[SUB]', '[TRIPLE]'])
    tokenizer.add_special_tokens({'eos_token':'[EOS]', 'bos_token': '[SOS]', 'pad_token': '[PAD]'})
    tokenizer.save_pretrained('../GPT2_tokenizer/')

    model = GPT2LMHeadModel.from_pretrained('gpt2')

    model.resize_token_embeddings(len(tokenizer))
    #embedding = model.transformer.wte.weight
    #torch.save(embedding, 'gpt_tokenizer_new/embedding.bin')

    #knowledge_config = {'max_entity_embeddings': 8, 'max_triple_embeddings': 20, 'max_position_embeddings': 1024, 'vocab_size': len(tokenizer), 
    #                   'pad_token_id': tokenizer.pad_token_id, 'hidden_size': model.config.n_embd, 'hidden_dropout_prob': 0.1, 'layer_norm_eps': 1e-12, 
    #                   'sos_token_id': tokenizer.encode('[SOS]')[0], 'eos_token_id': tokenizer.encode('[EOS]')[0]}
    knowledge_config = {'vocab_size': len(tokenizer), 'pad_token_id': tokenizer.pad_token_id, 'hidden_dropout_prob': 0.1, 'max_entity_embeddings': 30,\
            'max_triple_embeddings': 20, 'max_position_embeddings': 1024, 'layer_norm_eps': 1e-12, 'sos_token_id': tokenizer.encode('[SOS]')[0],\
            'eos_token_id': tokenizer.encode('[EOS]')[0], 'hidden_size': model.config.n_embd}
    
    with open('../GPT2_tokenizer/knowledge_config.json', 'w') as f:
        json.dump(knowledge_config, f, indent=2)

if sys.argv[1] == 'build-xlnet':
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

    tokenizer.add_tokens(['[ENT]', '[PRED]', '[SUB]', '[TRIPLE]', '[EOS]', '[SOS]'])
    tokenizer.save_pretrained('XLNET_tokenizer/')

    model = BertModel.from_pretrained('bert-base-cased')
    model.resize_token_embeddings(len(tokenizer))
    embedding = model.embeddings.word_embeddings.weight
    torch.save(embedding, 'XLNET_tokenizer/embedding.bin')

    knowledge_config = {'max_entity_embeddings': 8, 'max_triple_embeddings': 20, 'max_position_embeddings': 1024, 'vocab_size': len(tokenizer), 
                        'pad_token_id': tokenizer.pad_token_id, 'hidden_size': model.config.hidden_size, 'hidden_dropout_prob': 0.1, 'layer_norm_eps': 1e-12, 
                        'sos_token_id': tokenizer.encode('[SOS]')[0], 'eos_token_id': tokenizer.encode('[EOS]')[0]}
    #model.save_pretrained('./')
    with open('XLNET_tokenizer/knowledge_config.json', 'w') as f:
        json.dump(knowledge_config, f, indent=2)


if sys.argv[1] == 'build-ctrl':
    tokenizer = CTRLTokenizer.from_pretrained('ctrl')

    tokenizer.add_tokens(['[ENT]', '[PRED]', '[SUB]', '[TRIPLE]', '[EOS]', '[SOS]'])
    tokenizer.save_pretrained('CTRL_tokenizer/')

    model = BertModel.from_pretrained('bert-base-cased')
    model.resize_token_embeddings(len(tokenizer))
    embedding = model.embeddings.word_embeddings.weight
    torch.save(embedding, 'CTRL_tokenizer/embedding.bin')

    knowledge_config = {'max_entity_embeddings': 8, 'max_triple_embeddings': 20, 'max_position_embeddings': 1024, 'vocab_size': len(tokenizer), 
                        'pad_token_id': tokenizer.pad_token_id, 'hidden_size': model.config.hidden_size, 'hidden_dropout_prob': 0.1, 'layer_norm_eps': 1e-12, 
                        'sos_token_id': tokenizer.encode('[SOS]')[0], 'eos_token_id': tokenizer.encode('[EOS]')[0]}
    #model.save_pretrained('./')
    with open('CTRL_tokenizer/knowledge_config.json', 'w') as f:
        json.dump(knowledge_config, f, indent=2)


elif sys.argv[1] == 'trial':
    tokenizer = BertTokenizer.from_pretrained('tokenizer/')
    input_string = '[ENT] [PRED] label [SUB] Taylor [TRIPLE] [PRED] description [SUB] a popular single [TRIPLE] [PRED] is a subclass of [SUB] universe [TRIPLE] [PRED] has surname [SUB] Kyle [TRIPLE]'
    intermediate = tokenizer.tokenize(input_string)
    print(intermediate)
    #tokens = torch.LongTensor(tokenizer.encode())
    intermediate = tokenizer.encode(input_string, add_special_tokens=False)
    print(intermediate)

    embedding_weight = torch.load('tokenizer/embedding.bin')
    emb = nn.Embedding(len(tokenizer), 768)
    emb.weight.data = embedding_weight
    #y = emb(tokens)
    #print(y.shape)

elif sys.argv[1] == 'test':
    tokenizer = BertTokenizer.from_pretrained('tokenizer/')

    entities = ['Q31', 'Q22', 'Q35', 'Q245']

    with open('../knowledge.json', 'r') as f:
        mapping = json.load(f)

    strings = []
    index_ids = []
    entity_ids = []
    triple_ids = []
    for i, entity in enumerate(entities):
        string = ['[ENT]']
        triple_id = [0]

        entity = mapping[entity]
        label = entity[0]

        words = tokenizer.tokenize('[PRED] label [SUB] {} [TRIPLE]'.format(label))
        string += words
        triple_id += [triple_id[-1] + 1] * len(words)

        description = entity[1]
        words = tokenizer.tokenize('[PRED] description [SUB] {} [TRIPLE]'.format(description))
        string += words
        triple_id += [triple_id[-1] + 1] * len(words)

        added = set()
        relations = entity[2]
        for rel in relations:
            if rel[0] not in added:
                word = tokenizer.tokenize('[PRED] {} [SUB] {} [TRIPLE]'.format(rel[0], rel[1]))
                string += word
                triple_id += [triple_id[-1] + 1] * len(words)
                added.add(rel[0])
            if len(added) >= 8:
                break

        strings += string
        entity_ids += [i] * len(string)
        triple_ids += triple_id

    index_ids = list(range(len(strings)))

    print(strings)
    print(entity_ids)
    print(triple_ids)
    print(index_ids)

elif sys.argv[1] == 'embedding':
    from Model import KnowledgeEmbeddings

    with open('tokenizer/knowledge_config.json') as f:
        knowledge_config = SimpleNamespace(**json.load(f))
    print(knowledge_config)

    embedding = KnowledgeEmbeddings(knowledge_config)

    tokenizer = BertTokenizer.from_pretrained('tokenizer/')

    entities = ['Q31', 'Q22', 'Q35', 'Q245']

    with open('../knowledge.json', 'r') as f:
        mapping = json.load(f)

    strings = []
    index_ids = []
    entity_ids = []
    triple_ids = []
    for i, entity_id in enumerate(entities):
        string = tokenizer.encode('[ENT]', add_special_tokens=False)
        triple_id = [0]

        entity = mapping[entity_id]

        words = tokenizer.encode('[PRED] label [SUB] {} [TRIPLE]'.format(entity[0]), add_special_tokens=False)

        string += words
        triple_id += [triple_id[-1] + 1] * len(words)

        words = tokenizer.encode('[PRED] description [SUB] {} [TRIPLE]'.format(entity[1]), add_special_tokens=False)
        string += words
        triple_id += [triple_id[-1] + 1] * len(words)

        added = set()
        relations = entity[2]
        for rel in relations:
            if rel[0] not in added:
                words = tokenizer.encode('[PRED] {} [SUB] {} [TRIPLE]'.format(rel[0], rel[1]), add_special_tokens=False)
                string += words
                triple_id += [triple_id[-1] + 1] * len(words)               
                
                added.add(rel[0])
            if len(added) >= 8:
                break

        strings += string
        entity_ids += [i] * len(string)
        triple_ids += triple_id

    index_ids = list(range(len(strings)))

    input_ids = torch.LongTensor(strings)[None, :]
    entity_ids = torch.LongTensor(entity_ids)[None, :]
    triple_ids = torch.LongTensor(triple_ids)[None, :]
    position_ids = torch.LongTensor(index_ids)[None, :]
    #print(input_ids, entity_ids, triple_ids, position_ids)

    output = embedding(input_ids, entity_ids, triple_ids, position_ids)
    print(output)

elif sys.argv[1] == 'split':
    with open('../examples.json', 'r') as f:
        examples = json.load(f)
    import random
    random.shuffle(examples)

    test_size = int(0.02 * len(examples))
    train_examples = examples[:-2 * test_size]
    dev_examples = examples[-2 * test_size: -1 * test_size]
    test_examples = examples[-1 * test_size]

    with open('../train.json', 'w') as f:
        json.dump(train_examples, f)

    with open('../val.json', 'w') as f:
        json.dump(dev_examples, f)

    with open('../test.json', 'w') as f:
        json.dump(test_examples, f)

elif sys.argv[1] == 'test_tokenizer':
    from transformers import BertTokenizer#GPT2Tokenizer, XLNetTokenizer, RobertaTokenizer 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print(len(tokenizer))
    x = tokenizer.encode('190.0', add_special_tokens=False)
    z = tokenizer.decode(x)
    print(z)
    x = tokenizer.encode('MC.C.', add_special_tokens=False)
    z = tokenizer.decode(x)
    print(z)
