import torch
import os
import numpy as np
import json
import torch
import random
import pandas
import re
import copy
from torch.utils.data import Dataset

def safe_setting(matrix, x_start, x_end, y_start, y_end):
    if x_start >= matrix.shape[0] or y_start >= matrix.shape[0]:
        return

    matrix[x_start:min(matrix.shape[0], x_end), y_start:min(matrix.shape[1], y_end)] = 1
    return

class KBDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_entity, max_fact, forbid_duplicate_relation=True, percent=1.0):
        super(KBDataset, self).__init__()
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        
        if percent > 1:
            self.data = self.data[:int(percent)]
        else:
            selected_size = int(len(self.data) * percent)
            self.data = self.data[:selected_size]

        self.tokenizer = tokenizer
        self.max_entity = max_entity
        self.max_fact = max_fact
        self.forbid_duplicate_relation = forbid_duplicate_relation

    def linearize_v2(self, entity, lower_case=False):
        if lower_case:
            string = self.tokenizer.encode('[ENT] {}'.format(entity[0].lower()), add_special_tokens=False)
        else:
            string = self.tokenizer.encode('[ENT] {}'.format(entity[0]), add_special_tokens=False)
        triple_id = [1] * len(string)

        if lower_case:
            words = self.tokenizer.encode('[TRIPLE] [PRED] description [SUB] {} [TRIPLE]'.format(entity[1].lower()), add_special_tokens=False)
        else:
            words = self.tokenizer.encode('[TRIPLE] [PRED] description [SUB] {} [TRIPLE]'.format(entity[1]), add_special_tokens=False)
            
        string += words
        triple_id += [triple_id[-1] + 1] * len(words)

        added = set()
        for rel in entity[2]:
            if self.forbid_duplicate_relation and rel[0] in added:
                pass
            else:
                if lower_case:
                    words = self.tokenizer.encode('[PRED] {} [SUB] {} [TRIPLE]'.format(rel[0].lower(), rel[1].lower()), add_special_tokens=False)
                else:
                    words = self.tokenizer.encode('[PRED] {} [SUB] {} [TRIPLE]'.format(rel[0], rel[1]), add_special_tokens=False)                    
                string += words
                triple_id += [triple_id[-1] + 1] * len(words)
                added.add(rel[0])
            
            if len(added) >= self.max_fact:
                break
        
        return string, triple_id


    def __len__(self):
        return len(self.data)

    def __getitem__(self):
        raise NotImplementedError


class WikiDataDataset(KBDataset):
    def __init__(self, file_path, knowledge_path, tokenizer, max_entity, max_fact=12, max_enc_len=1024, max_dec_len=50, encoder=None, lower_case=False):
        super(WikiDataDataset, self).__init__(file_path, tokenizer, max_entity, max_fact, True)
        with open(knowledge_path, 'r') as f:
            self.knowledge = json.load(f)

        print("Total samples = {}; Total entities = {}".format(len(self.data), len(self.knowledge)))
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.lower_case = lower_case
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.pad_token_id
        self.encoder = encoder

    def get_reference(self, idx, lower_case=False):
        if self.lower_case:
            return [[_.lower() for _ in self.data[idx]['text']]]
        else:
            return [self.data[idx]['text']]

    def get_entities(self, idx):
        entry = self.data[idx]
        entities = []
        for _ in entry['kblinks']:
            if _ is not None and _ in self.knowledge and _ not in entities:
                entities.append(_)
        if 'title' in entry:
            entities.insert(0, "TITLE:::" + entry['title_kb_id'])
        return entities

    def __getitem__(self, idx):
        entry = self.data[idx]

        sentence = ' '.join(entry['text'])
        if self.lower_case:
            sentence = sentence.lower()

        entities = []
        for _ in entry['kblinks']:
            if _ is not None and _ in self.knowledge and _ not in entities:
                entities.append(_)

        if self.encoder == 'sequence':
            strings = []
            entity_ids = []
            triple_ids = []

            if 'title' in entry:
                entity = self.knowledge[entry['title_kb_id']]
                
                string, triple_id = self.linearize_v2(entity, self.lower_case)
                
                strings += string
                entity_ids += [0] * len(string)
                triple_ids += triple_id

            for i, entity_id in enumerate(entities):
                if i + 1 >= self.max_entity:
                    break

                entity = self.knowledge[entity_id]
                
                string, triple_id = self.linearize_v2(entity, self.lower_case)

                strings += string
                entity_ids += [i + 1] * len(string)
                triple_ids += triple_id

            position_ids = list(range(len(strings)))
            assert len(strings) == len(entity_ids) == len(triple_ids) == len(position_ids)

            if len(strings) >= self.max_enc_len:
                input_ids = torch.LongTensor(strings[:self.max_enc_len])
                entity_ids = torch.LongTensor(entity_ids[:self.max_enc_len])
                triple_ids = torch.LongTensor(triple_ids[:self.max_enc_len])
                position_ids = torch.LongTensor(position_ids[:self.max_enc_len])
            else:
                input_ids = torch.LongTensor(strings + [self.pad_idx] * (self.max_enc_len - len(strings)))
                entity_ids = torch.LongTensor(entity_ids + [0] * (self.max_enc_len - len(strings)))
                triple_ids = torch.LongTensor(triple_ids + [0] * (self.max_enc_len - len(strings)))
                position_ids = torch.LongTensor(position_ids + [0] * (self.max_enc_len - len(strings)))

            sentence = self.tokenizer.encode('[SOS] {} [EOS]'.format(sentence), add_special_tokens=False)
            if len(sentence) >= self.max_dec_len:
                output_ids = torch.LongTensor(sentence[:self.max_dec_len])
            else:
                output_ids = torch.LongTensor(sentence[:self.max_dec_len] + [self.pad_idx] * (self.max_dec_len - len(sentence)))

            return input_ids, entity_ids, triple_ids, position_ids, output_ids[:-1], output_ids[1:]

        elif self.encoder == 'graph':
            strings = []
            
            term_level_mask = torch.eye(self.max_enc_len).bool()
            triple_level_mask = torch.eye(self.max_enc_len).bool()
            entity_level_mask = torch.eye(self.max_enc_len).bool()
            fact_level_mask = torch.eye(self.max_enc_len).bool()
            gather_level_mask = torch.eye(self.max_enc_len).bool()

            offset = 0
            entity_positions = []
            if 'title' in entry:
                entities.insert(0, entry['title_kb_id'])

            for i, entity_id in enumerate(entities):                
                if i >= self.max_entity:
                    break

                entity = self.knowledge[entity_id]

                string = self.tokenizer.encode('[ENT]')
                entity_positions.append(offset)
                offset += 1

                entity[2].insert(0, ['description', entity[1]])
                triple_positions = []
                added = set()        
                for rel in entity[2]:
                    if offset >= self.max_enc_len - 1:
                        break
                    elif self.forbid_duplicate_relation and rel[0] in added:
                        continue
                    else:             
                        words = self.tokenizer.encode('[TRIPLE]', add_special_tokens=False)
                        triple_positions.append(offset)
                        string += words
                        offset += len(words)

                        words = self.tokenizer.encode('{} [PRED] {} [SUB] {}'.format(entity[0], rel[0], rel[1]), add_special_tokens=False)
                        string += words
                        safe_setting(term_level_mask, offset, offset + len(words), offset, offset + len(words))
                        offset += len(words)
                        safe_setting(triple_level_mask, triple_positions[-1], triple_positions[-1] + 1, triple_positions[-1] + 1, offset)

                        added.add(rel[0])

                    if len(added) >= self.max_fact:
                        break

                entity_level_mask[entity_positions[-1], triple_positions] = 1
                strings += string

                # Reserve at least 4 tokens for the next entity
                if offset >= self.max_enc_len - 5:
                    break

            for entity_position in entity_positions:
                fact_level_mask[entity_position, entity_positions] = 1

            for i in range(len(entity_positions) - 1):
                gather_level_mask[entity_positions[i] + 1 : entity_positions[i + 1], entity_positions] = 1
            gather_level_mask[entity_positions[-1] + 1:offset, entity_positions] = 1

            if len(strings) >= self.max_enc_len:
                input_ids = torch.LongTensor(strings[:self.max_enc_len])
            else:
                input_ids = torch.LongTensor(strings + [self.pad_idx] * (self.max_enc_len - len(strings)))
                
            sentence = self.tokenizer.encode('[SOS] {} [EOS]'.format(sentence), add_special_tokens=False)
            if len(sentence) >= self.max_dec_len:
                output_ids = torch.LongTensor(sentence[:self.max_dec_len])
            else:
                output_ids = torch.LongTensor(sentence[:self.max_dec_len] + [self.pad_idx] * (self.max_dec_len - len(sentence)))
                
            term_level_mask = ~term_level_mask
            triple_level_mask = ~triple_level_mask
            entity_level_mask = ~entity_level_mask
            fact_level_mask = ~fact_level_mask
            gather_level_mask = ~gather_level_mask

            return input_ids, term_level_mask, triple_level_mask, entity_level_mask, fact_level_mask, gather_level_mask, output_ids[:-1], output_ids[1:]
        
        else:
            raise NotImplementedError        


class DownStreamDataset(KBDataset):
    def __init__(self, file_path, tokenizer, max_entity, max_fact=12, max_enc_len=128, max_dec_len=30, encoder=None, forbid_duplicate_relation=True, percent=1.0):
        super(DownStreamDataset, self).__init__(file_path, tokenizer, max_entity, max_fact, forbid_duplicate_relation, percent)

        print("Total samples = {}".format(len(self.data)))
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len

        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.pad_token_id
        self.encoder = encoder

    def get_reference(self, idx, do_lower_case=False):
        if do_lower_case:
            return [_.lower().split(' ') for _ in self.data[idx]['text']]
        else:
            return [_.split(' ') for _ in self.data[idx]['text']]

    def get_entities(self, idx):
        return list(self.data[idx]['kbs'].keys())

    def __getitem__(self, idx):
        entry = self.data[idx]
        sentence = random.choice(entry['text'])
        KBs = entry['kbs']

        if self.encoder == 'sequence':
            strings = []
            entity_ids = []
            triple_ids = []

            for i, entity_label in enumerate(KBs):
                if i + 1 >= self.max_entity:
                    break

                entity = KBs[entity_label]

                string, triple_id = self.linearize_v2(entity)

                strings += string
                entity_ids += [i + 1] * len(string)
                triple_ids += triple_id

            position_ids = list(range(len(strings)))
            assert len(strings) == len(entity_ids) == len(triple_ids) == len(position_ids)

            if len(strings) >= self.max_enc_len:
                input_ids = torch.LongTensor(strings[:self.max_enc_len])
                entity_ids = torch.LongTensor(entity_ids[:self.max_enc_len])
                triple_ids = torch.LongTensor(triple_ids[:self.max_enc_len])
                position_ids = torch.LongTensor(position_ids[:self.max_enc_len])
            else:
                input_ids = torch.LongTensor(strings + [self.pad_idx] * (self.max_enc_len - len(strings)))
                entity_ids = torch.LongTensor(entity_ids + [0] * (self.max_enc_len - len(strings)))
                triple_ids = torch.LongTensor(triple_ids + [0] * (self.max_enc_len - len(strings)))
                position_ids = torch.LongTensor(position_ids + [0] * (self.max_enc_len - len(strings)))

            sentence = self.tokenizer.encode('[SOS] {} [EOS]'.format(sentence), add_special_tokens=False)
            if len(sentence) >= self.max_dec_len:
                output_ids = torch.LongTensor(sentence[:self.max_dec_len])
            else:
                output_ids = torch.LongTensor(sentence[:self.max_dec_len] + [self.pad_idx] * (self.max_dec_len - len(sentence)))

            return input_ids, entity_ids, triple_ids, position_ids, output_ids[:-1], output_ids[1:]

        elif self.encoder == 'graph':
            strings = []
            
            term_level_mask = torch.eye(self.max_enc_len).bool()
            triple_level_mask = torch.eye(self.max_enc_len).bool()
            entity_level_mask = torch.eye(self.max_enc_len).bool()
            fact_level_mask = torch.eye(self.max_enc_len).bool()
            gather_level_mask = torch.eye(self.max_enc_len).bool()

            offset = 0
            entity_positions = []
            for i, entity_label in enumerate(KBs):
                if i >= self.max_entity:
                    break

                entity = KBs[entity_label]

                string = self.tokenizer.encode('[ENT]')
                entity_positions.append(offset)
                offset += 1

                triple_positions = []
                entity[2].insert(0, ['description', entity[1]])
                added = set()
                for rel in entity[2]:
                    if offset >= self.max_enc_len - 1:
                        break
                    elif self.forbid_duplicate_relation and rel[0] in added:
                        continue
                    else:             
                        words = self.tokenizer.encode('[TRIPLE]', add_special_tokens=False)
                        triple_positions.append(offset)
                        string += words
                        offset += len(words)

                        words = self.tokenizer.encode('{} [PRED] {} [SUB] {}'.format(entity[0], rel[0], rel[1]), add_special_tokens=False)
                        string += words
                        safe_setting(term_level_mask, offset, offset + len(words), offset, offset + len(words))
                        offset += len(words)
                        safe_setting(triple_level_mask, triple_positions[-1], triple_positions[-1] + 1, triple_positions[-1] + 1, offset)

                        added.add(rel[0])

                    if len(added) >= self.max_fact:
                        break

                entity_level_mask[entity_positions[-1], triple_positions] = 1
                strings += string

                # Reserve at least 4 tokens for the next entity
                if offset >= self.max_enc_len - 5:
                    break

            for entity_position in entity_positions:
                fact_level_mask[entity_position, entity_positions] = 1

            for i in range(len(entity_positions) - 1):
                gather_level_mask[entity_positions[i] + 1 : entity_positions[i + 1], entity_positions] = 1
            gather_level_mask[entity_positions[-1] + 1:offset, entity_positions] = 1

            if len(strings) >= self.max_enc_len:
                input_ids = torch.LongTensor(strings[:self.max_enc_len])
            else:
                input_ids = torch.LongTensor(strings + [self.pad_idx] * (self.max_enc_len - len(strings)))
                
            sentence = self.tokenizer.encode('[SOS] {} [EOS]'.format(sentence), add_special_tokens=False)
            if len(sentence) >= self.max_dec_len:
                output_ids = torch.LongTensor(sentence[:self.max_dec_len])
            else:
                output_ids = torch.LongTensor(sentence[:self.max_dec_len] + [self.pad_idx] * (self.max_dec_len - len(sentence)))

            term_level_mask = ~term_level_mask
            triple_level_mask = ~triple_level_mask
            entity_level_mask = ~entity_level_mask
            fact_level_mask = ~fact_level_mask
            gather_level_mask = ~gather_level_mask
            
            return input_ids, term_level_mask, triple_level_mask, entity_level_mask, fact_level_mask, gather_level_mask, output_ids[:-1], output_ids[1:]
        
        else:
            raise NotImplementedError

class WebNLGDataset(DownStreamDataset):
    def __init__(self, file_path, tokenizer, max_entity, max_fact=12, max_enc_len=128, max_dec_len=30, encoder=False, percent=1.0):
        super(WebNLGDataset, self).__init__(file_path, tokenizer, max_entity, max_fact, max_enc_len, max_dec_len, encoder, True, percent)

class WebNLGChallengeDataset(DownStreamDataset):
    def __init__(self, file_path, tokenizer, max_entity, max_fact=12, max_enc_len=128, max_dec_len=30, encoder=False, percent=1.0):
        super(WebNLGChallengeDataset, self).__init__(file_path, tokenizer, max_entity, max_fact, max_enc_len, max_dec_len, encoder, True, percent)

class E2ENLGDataset(DownStreamDataset):
    def __init__(self, file_path, tokenizer, max_entity, max_fact=12, max_enc_len=128, max_dec_len=30, encoder=False, percent=1.0):
        super(E2ENLGDataset, self).__init__(file_path, tokenizer, max_entity, max_fact, max_enc_len, max_dec_len, encoder, True, percent)

class LogicNLGDataset(DownStreamDataset):
    def __init__(self, file_path, tokenizer, max_entity, max_fact=12, max_enc_len=128, max_dec_len=30, encoder=False, percent=1.0):
        super(LogicNLGDataset, self).__init__(file_path, tokenizer, max_entity, max_fact, max_enc_len, max_dec_len, encoder, False, percent)

class WikiBioNLGDataset(DownStreamDataset):
    def __init__(self, file_path, tokenizer, max_entity, max_fact=12, max_enc_len=128, max_dec_len=30, encoder=False, percent=1.0):
        super(WikiBioNLGDataset, self).__init__(file_path, tokenizer, max_entity, max_fact, max_enc_len, max_dec_len, encoder, True, percent)

class GPTDataset(KBDataset):
    def __init__(self, file_path, tokenizer, max_entity, max_fact=12, max_enc_len=128, max_dec_len=30, percent=1.0):
        super(GPTDataset, self).__init__(file_path, tokenizer, max_entity, max_fact, percent=percent)
        print("Total samples = {}".format(len(self.data)))
        self.pad_idx = tokenizer.pad_token_id
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
    
    def get_reference(self, idx):
        return [_.split(' ') for _ in self.data[idx]['text']]

    def get_entities(self, idx):
        return list(self.data[idx]['kbs'].keys())

    def __getitem__(self, idx):
        entry = self.data[idx]

        sentence = random.choice(entry['text'])

        KBs = entry['kbs']

        strings = ''
        for i, entity_label in enumerate(KBs):
            entity = KBs[entity_label]

            name = entity[0]
            
            for rel in entity[-1]:
                strings += ' {} {} {} . '.format(name, rel[0], rel[1])

        KB_ids = self.tokenizer.encode(strings, add_special_tokens=False)
        
        if len(KB_ids) < self.max_enc_len:
            KB_ids = [self.pad_idx] * (self.max_enc_len - len(KB_ids)) + KB_ids
        else:
            KB_ids = KB_ids[:self.max_enc_len]

        inputs = torch.LongTensor(KB_ids)

        sentence = self.tokenizer.encode('[SOS] {} [EOS]'.format(sentence), add_special_tokens=False)
        if len(sentence) >= self.max_dec_len:
            output_ids = torch.LongTensor(sentence[:self.max_dec_len])
        else:
            output_ids = torch.LongTensor(sentence[:self.max_dec_len] + [self.pad_idx] * (self.max_dec_len - len(sentence)))

        return inputs, output_ids[:-1], output_ids[1:]
