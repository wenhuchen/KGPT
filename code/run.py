import torch
import math
import os
import nltk
import sys
import json
from tqdm import trange, tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from Model import TransformerDecoder, GatedTransformerDecoder, GraphGatedTransformerDecoder
import argparse
from DataLoader import *
from transformers import BertTokenizer, GPT2Tokenizer, XLNetTokenizer
from types import SimpleNamespace
from torch.autograd import Variable
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from nltk.tokenize import word_tokenize
from torch.nn import Parameter

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, default="train",
                        help="whether to train or test the model", choices=['train', 'test', 'challenge', 'visualize', 'LM', 'compute_bleu', 'few_shot'])
    parser.add_argument('--batch_size', type=int, default=64, help="the embedding dimension")
    parser.add_argument('--train_path', type=str, default="", help="the embedding dimension")
    parser.add_argument('--test_path', type=str, default="", help="the embedding dimension")
    parser.add_argument('--challenge_path', type=str, default="", help="the embedding dimension")
    parser.add_argument('--knowledge_path', type=str, default="", help="the embedding dimension")
    parser.add_argument('--epochs', type=int, default=10, help="the embedding dimension")
    parser.add_argument('--save_every_n_epochs', type=int, default=1, help="the embedding dimension")
    parser.add_argument('--n_head', type=int, default=8, help="the embedding dimension")
    parser.add_argument('--tokenizer_dir', type=str, default='GPT2_tokenizer/', help="tokenizer loaded from")
    parser.add_argument('--n_layers', type=int, default=6, help="the embedding dimension")
    parser.add_argument('--max_len', type=int, default=8, help="the embedding dimension")
    parser.add_argument('--dataset', type=str, default='', help="the embedding dimension")
    parser.add_argument('--config', type=str, default='', help="config file for the embedding layer")
    parser.add_argument('--embedding_path', type=str, default='', help="pretrained BERT embedding weight")
    parser.add_argument('--max_enc_len', type=int, default=640, help="maximum length of the encoding part")
    parser.add_argument('--max_dec_len', type=int, default=72, help="maximum length of the decoding part")
    parser.add_argument('--output_dir', type=str, default='checkpoint', help="maximum length of the decoding part")
    parser.add_argument('--logging_steps', type=int, default=20, help="maximum length of the decoding part")
    parser.add_argument('--printing_steps', type=int, default=500, help="maximum length of the decoding part")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="maximum length of the decoding part")
    parser.add_argument('--max_fact', type=int, default=12, help="maximum length of the decoding part")
    parser.add_argument('--max_entity', type=int, default=8, help="maximum length of the decoding part")
    parser.add_argument('--starting_epoch', type=int, default=0, help="maximum length of the decoding part")
    parser.add_argument('--load_from', type=str, default='', help="maximum length of the decoding part")
    parser.add_argument('--num_workers', type=int, default=8, help="maximum length of the decoding part")
    parser.add_argument('--beam_size', type=int, default=2, help="the embedding dimension")    
    parser.add_argument('--bleu', type=int, default=4, help="the embedding dimension")    
    parser.add_argument('--hidden_size', type=int, default=None, help="the embedding dimension")    
    parser.add_argument('--finetune', action='store_true', help='Which experiment you are doing')
    parser.add_argument('--additional', type=str, default="", help='Which experiment you are doing')
    parser.add_argument('--unforbid_duplicate', default=False, action='store_true', help='Which experiment you are doing')
    parser.add_argument('--encoder', type=str, required=True, choices=['sequence', 'graph', 'graph_finegrained'], help='Which experiment you are doing')
    parser.add_argument('--lower_case', default=False, action='store_true', help='Which experiment you are doing')
    parser.add_argument('--copy_loss', default=False, action='store_true', help='Which experiment you are doing')
    parser.add_argument('--percent', default=1.0, type=float, help='Which experiment you are doing')
    args = parser.parse_args()

    return args

def load_my_state_dict(self, state_dict):
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if 'post_word_emb' in name:
            print("Sin/Cos embedding does not need to be reloaded")
            continue
        if 'entity_embeddings' in name or 'triple_embeddings' in name:
            if param.shape != own_state[name].shape:
                print("Reinitializing the weight for {}".format(name))
                continue
            else:
                pass
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

if __name__ == '__main__':
    args = parse_opt()
    device = torch.device('cuda')
    args.device = device
    if 'bert' in args.tokenizer_dir.lower():
        tokenization = 'BERT'
    elif 'gpt2' in args.tokenizer_dir.lower():
        tokenization = 'GPT2'
    elif 'xlnet' in args.tokenizer_dir.lower():
        tokenization = "XLNET"
    else:
        raise NotImplementedError("Not supported with {}".format(args.tokenizer_dir))

    folder = 'checkpoint_{}'.format(args.dataset)    
    if args.finetune:
        if args.option == 'few_shot':
            args.output_dir = '{}/{}_finetune_{}_head{}_layer{}_{}_maxfact{}{}_fewshot{}'.format(folder, 
                args.output_dir, args.encoder, args.n_head, args.n_layers, tokenization, args.max_fact, args.additional, args.percent)
        else:
            args.output_dir = '{}/{}_finetune_{}_head{}_layer{}_{}_maxfact{}{}'.format(folder, 
                args.output_dir, args.encoder, args.n_head, args.n_layers, tokenization, args.max_fact, args.additional)    
    else:
        if args.option == 'few_shot':
            args.output_dir = '{}/{}_{}_head{}_layer{}_{}_maxfact{}_fewshot{}{}'.format(folder, 
                args.output_dir, args.encoder, args.n_head, args.n_layers, tokenization, args.max_fact, args.additional, args.percent)
        else:
            args.output_dir = '{}/{}_{}_head{}_layer{}_{}_maxfact{}{}'.format(folder, 
                args.output_dir, args.encoder, args.n_head, args.n_layers, tokenization, args.max_fact, args.additional)

    if args.train_path == '':
        args.train_path = os.path.join('dataset', args.dataset, 'train.json')
    if args.test_path == '':
        args.test_path = os.path.join('dataset', args.dataset, 'val.json')
    if args.challenge_path == '':
        args.challenge_path = os.path.join('dataset', args.dataset, 'test.json')

    args.embedding_path = os.path.join(args.tokenizer_dir, 'embedding.bin')
    args.config = os.path.join(args.tokenizer_dir, 'knowledge_config.json')
    print(args)

    if tokenization == 'BERT':
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_dir)
        banwords = tokenizer.convert_tokens_to_ids(['It', 'She', 'They', 'He', 'it', 'she', 'he', 'they'])
    elif tokenization == 'GPT2':
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_dir)
        banwords = tokenizer.convert_tokens_to_ids(['It', 'She', 'They', 'He', 'it', 'she', 'he', 'they'])        
    elif tokenization == 'XLNET':
        tokenizer = XLNetTokenizer.from_pretrained(args.tokenizer_dir)
        banwords = tokenizer.convert_tokens_to_ids(['It', 'She', 'They', 'He', 'it', 'she', 'he', 'they'])      
    else:
        raise NotImplementedError

    with open(args.config, 'r') as f:
        knowledge_config = json.load(f)
        config = SimpleNamespace(**knowledge_config)
    print(config)

    if args.option == 'compute_bleu':
        with open('decoded_results.json', 'r') as f:
            results = json.load(f)
        with open(args.test_path, 'r') as f:
            references = json.load(f)

        list_of_references = []
        list_of_hypothesis = []
        for r, ref in zip(results, references):
            r = word_tokenize(r.strip())
            ref = ref['text']
            refs = [_.split(' ') for _ in ref]
            #bleus.append()
            list_of_references.append(refs)
            list_of_hypothesis.append(r)
            bleu = nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypothesis)
            sys.stdout.write("BLEU score = {} \r".format(bleu))            
        
        print("Overall BLEU score = {} \r".format(bleu))
        exit()

    if args.encoder == 'sequence':
        model = GatedTransformerDecoder(config, args.n_head, args.n_layers)
        print("Running Transformer with copy gate")
    elif args.encoder == 'graph':
        model = GraphGatedTransformerDecoder(config, args.n_head, args.n_layers)
        print("Running Transformer with fine-grained graph architecture")       
    else:
        model = TransformerDecoder(config, args.n_head, args.n_layers)
        print("Running vanilla Transformer")

    model = torch.nn.DataParallel(model)

    if args.load_from != '':
        reloaded = torch.load(args.load_from)
        load_my_state_dict(model, reloaded)
        #model.load_state_dict(reloaded)
        print("Loading model from {}".format(args.load_from))
    
    model.to(device)

    if args.option == 'train':
        if 'wikidata' in args.dataset.lower():
            train_data = WikiDataDataset(args.train_path, args.knowledge_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder, args.lower_case)
            print("Using WikiData generation dataset")
        elif args.dataset.lower() == 'webnlg':
            train_data = WebNLGDataset(args.train_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder)
            print("Using WebNLG generation dataset")      
        elif args.dataset.lower() == 'webnlg_challenge':
            train_data = WebNLGChallengeDataset(args.train_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder)
            print("Using WebNLG challenge dataset")        
        elif args.dataset.lower() == 'e2enlg':
            train_data = E2ENLGDataset(args.train_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder)
            print("Using E2E generation dataset")
        elif args.dataset.lower() == 'logicnlg':
            train_data = LogicNLGDataset(args.train_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder)
            print("Using LogicNLG generation dataset")        
        elif args.dataset.lower() == 'wikibionlg':
            train_data = WikiBioNLGDataset(args.train_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder)
            print("Using Wikibio generation dataset")            
        else:
            raise NotImplementedError("This dataset is not yet supported")

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

        t_total = len(train_dataloader) * args.epochs
        warmup_steps = int(t_total * 0.2)
        print("Warming up for {} steps, and then start linearly decresing learning rate".format(warmup_steps))

        optimizer = optim.Adam(model.parameters(), args.learning_rate)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        loss_func.to(args.device)

        model.train()
        
        logging_loss = 0
        tr_loss = 0
        global_step = args.starting_epoch * len(train_dataloader)
        
        tb_writer = SummaryWriter(log_dir=args.output_dir)
        for epoch in trange(args.starting_epoch, args.epochs, desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                data = tuple(Variable(t).to(args.device) for t in batch)
                target = data[-1]

                model.zero_grad()
                optimizer.zero_grad()

                logits = model(*data[:-1])

                loss = loss_func(logits.view(-1, logits.shape[-1]), target.view(-1))
                if math.isnan(loss):
                    import pdb
                    pdb.set_trace()

                tr_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()

                global_step += 1

                if step > 0 and global_step % args.logging_steps == 0:
                    avg_loss = (tr_loss - logging_loss) / args.logging_steps
                    tb_writer.add_scalar("loss", avg_loss, global_step)
                    logging_loss = tr_loss

                if step > 0 and global_step % args.printing_steps == 0:
                    sub_data = [_[:2] for _ in data[:-2]]
                    decoded_sentence = model.module.greedy_decode(*sub_data,  banwords=banwords, max_token_seq_len=args.max_dec_len)
                    sentence = tokenizer.decode(decoded_sentence[0])
                    if '[EOS]' in sentence:
                        print(sentence[:sentence.index('[EOS]')])
                    else:
                        print(sentence)
                    groundtruth = tokenizer.decode(data[-1][0])
                    if '[EOS]' in groundtruth:
                        print("GROUND TRUTH: ", groundtruth[:groundtruth.index('[EOS]')])
                    else:
                        print("GROUND TRUTH: ", groundtruth)

            if epoch % args.save_every_n_epochs == 0 and epoch > 0:
                torch.save(model.state_dict(), '{}/model_ep{}.pt'.format(args.output_dir, epoch))

        torch.save(model.state_dict(), '{}/model_ep{}.pt'.format(args.output_dir, epoch))        
        tb_writer.close()

    elif args.option == 'few_shot':
        print("Few-shot learning with {} data".format(args.percent))
        if args.dataset.lower() == 'webnlg':
            train_data = WebNLGDataset(args.train_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder, args.percent)
            print("Using WebNLG generation dataset")
        elif args.dataset.lower() == 'e2enlg':
            train_data = E2ENLGDataset(args.train_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder, args.percent)
            print("Using E2E generation dataset")       
        elif args.dataset.lower() == 'wikibionlg':
            train_data = WikiBioNLGDataset(args.train_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder, args.percent)
            print("Using Wikibio generation dataset")            
        else:
            raise NotImplementedError("This dataset is not yet supported")

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

        t_total = len(train_dataloader) * args.epochs
        warmup_steps = int(t_total * 0.2)
        print("Warming up for {} steps, and then start linearly decresing learning rate".format(warmup_steps))

        optimizer = optim.Adam(model.parameters(), args.learning_rate)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        loss_func.to(args.device)

        model.train()
        
        logging_loss = 0
        tr_loss = 0
        global_step = args.starting_epoch * len(train_dataloader)
        
        tb_writer = SummaryWriter(log_dir=args.output_dir)
        for epoch in trange(args.starting_epoch, args.epochs, desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                data = tuple(Variable(t).to(args.device) for t in batch)
                target = data[-1]

                model.zero_grad()
                optimizer.zero_grad()

                if args.copy_loss:
                    logits, positionwise_copy_prob = model(*data[:-1], positionwise_copy_prob=True)
                else:
                    logits = model(*data[:-1])

                if args.copy_loss:
                    mask = (target != tokenizer.pad_token_id).unsqueeze(-1).float()
                    hit = (target.unsqueeze(-1) == data[0].unsqueeze(1).repeat(1, target.shape[1], 1)).float()
                    hit = (hit / (hit.sum(-1).unsqueeze(-1) + 1e-8)) * mask
                    copy_loss = (hit * (1 - positionwise_copy_prob)) # batch x length x src_length
                    copy_loss = copy_loss.sum(-1) #/ mask.squeeze().sum(-1).unsqueeze(-1)
                    copy_loss = copy_loss.mean()

                loss = loss_func(logits.view(-1, logits.shape[-1]), target.view(-1))
                
                if args.copy_loss:
                    loss = loss + 0.7 * copy_loss

                if math.isnan(loss):
                    print("NAN BATCH --------------------------------------------- DROP IT")
                    continue

                tr_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()

                global_step += 1

                if step > 0 and global_step % args.logging_steps == 0:
                    avg_loss = (tr_loss - logging_loss) / args.logging_steps
                    tb_writer.add_scalar("loss", avg_loss, global_step)
                    logging_loss = tr_loss

                if step > 0 and global_step % args.printing_steps == 0:
                    sub_data = [_[:2] for _ in data[:-2]]
                    decoded_sentence = model.module.greedy_decode(*sub_data,  banwords=banwords, max_token_seq_len=args.max_dec_len)
                    sentence = tokenizer.decode(decoded_sentence[0])
                    if '[EOS]' in sentence:
                        print(sentence[:sentence.index('[EOS]')])
                    else:
                        print(sentence)
                    groundtruth = tokenizer.decode(data[-1][0])
                    if '[EOS]' in groundtruth:
                        print("GROUND TRUTH: ", groundtruth[:groundtruth.index('[EOS]')])
                    else:
                        print("GROUND TRUTH: ", groundtruth)

            if epoch % args.save_every_n_epochs == 0 and epoch > 0:
                torch.save(model.state_dict(), '{}/model_ep{}.pt'.format(args.output_dir, epoch))

        torch.save(model.state_dict(), '{}/model_ep{}.pt'.format(args.output_dir, epoch))        
        tb_writer.close()


    elif args.option == 'visualize':
        if 'wikidata' in args.dataset.lower():
            eval_data = WikiDataDataset(args.test_path, args.knowledge_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder, args.lower_case)
        elif args.dataset.lower() == 'webnlg':
            eval_data = WebNLGDataset(args.test_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder)
        elif args.dataset.lower() == 'webnlg':
            eval_data = WebNLGConstrainedDataset(args.test_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder)        
        elif args.dataset.lower() == 'webnlg_challenge':
            eval_data = WebNLGChallengeDataset(args.test_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder)        
        elif args.dataset.lower() == 'e2enlg':
            eval_data = E2ENLGDataset(args.test_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder)
        elif args.dataset.lower() == 'logicnlg':
            eval_data = LogicNLGDataset(args.test_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder)
        elif args.dataset.lower() == 'wikibionlg':
            eval_data = WikiBioNLGDataset(args.test_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder)
        else:
            raise NotImplementedError("This dataset is not yet supported")
        
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

        model.eval()

        results = []
        bleu = []
        for step, batch in enumerate(eval_dataloader):
            data = tuple(Variable(t).to(args.device) for t in batch)
            if args.beam_size == 1:
                result = model.module.greedy_decode(*data[:-2], banwords=banwords, max_token_seq_len=args.max_dec_len)
            else:
                result = model.module.beam_search(data[:-2], tokenizer, n_bm=args.beam_size, banwords=banwords, max_token_seq_len=args.max_dec_len)

            for offset, r in enumerate(result):
                sent = tokenizer.decode(r, clean_up_tokenization_spaces=True)
                if '[EOS]' in sent:
                    sent = sent[:sent.index('[EOS]')]
                results.append(sent)

                idx = step * args.batch_size + offset
                references = eval_data.get_reference(idx)

                entities = eval_data.get_entities(idx)

                print("ENTITIES |||||", entities)
                print("DECODED |||||", sent)
                print("REFERENCE |||||", " ".join(references[0]))

    elif args.option in ['test', 'challenge']:
        if 'wikidata' in args.dataset.lower():
            eval_data = WikiDataDataset(args.test_path if args.option == 'test' else args.challenge_path, 
                                        args.knowledge_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder, args.lower_case)
        elif args.dataset.lower() == 'webnlg':
            eval_data = WebNLGDataset(args.test_path if args.option == 'test' else args.challenge_path, 
                                      tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder)
        elif args.dataset.lower() == 'webnlg_challenge':
            eval_data = WebNLGChallengeDataset(args.test_path if args.option == 'test' else args.challenge_path, 
                                      tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder)        
        elif args.dataset.lower() == 'e2enlg':
            eval_data = E2ENLGDataset(args.test_path if args.option == 'test' else args.challenge_path, 
                                      tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder)
        elif args.dataset.lower() == 'logicnlg':
            eval_data = LogicNLGDataset(args.test_path if args.option == 'test' else args.challenge_path,
                                      tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder)
        elif args.dataset.lower() == 'wikibionlg':
            eval_data = WikiBioNLGDataset(args.test_path if args.option == 'test' else args.challenge_path,
                                      tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder)
        else:
            raise NotImplementedError("This dataset is not yet supported")

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

        model.eval()

        results = []
        list_of_references = []
        list_of_hypothesis = []
        for step, batch in enumerate(eval_dataloader):
            data = tuple(Variable(t).to(args.device) for t in batch)
            
            if args.beam_size == 1:
                result = model.module.greedy_decode(*data[:-2], banwords=banwords, max_token_seq_len=args.max_dec_len)
            else:
                result = model.module.beam_search(data[:-2], tokenizer, n_bm=args.beam_size, banwords=banwords, max_token_seq_len=args.max_dec_len)

            for offset, r in enumerate(result):
                sent = tokenizer.decode(r, clean_up_tokenization_spaces=True)
                if '[EOS]' in sent:
                    sent = sent[:sent.index('[EOS]')].strip()

                idx = step * args.batch_size + offset
                references = eval_data.get_reference(idx)
                
                tok_sent = word_tokenize(sent)
                results.append(sent)
                list_of_hypothesis.append(tok_sent)
                list_of_references.append(references)

            if args.option == 'test':
                bleu = nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypothesis)
                sys.stdout.write('finished {}/{}; BLEU{} {} \r'.format(step, len(eval_dataloader), args.bleu, bleu))
            else:
                sys.stdout.write('finished {}/{} \r'.format(step, len(eval_dataloader)))

        if args.option == 'test':
            bleu = nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypothesis)
            print('finished {}/{}; BLEU{} {}'.format(step, len(eval_dataloader), args.bleu, bleu))
            with open('decoded_results.txt', 'w') as f:
                for _ in results:
                    f.write(_ + '\n')
        
        if args.option == 'challenge':
            with open(args.load_from.replace('.pt', '.txt'), 'w') as f:
                for _ in results:
                    f.write(_ + '\n')

    elif args.option == 'LM':
        if 'wikidata' in args.dataset.lower():
            eval_data = WikiDataDataset(args.test_path, args.knowledge_path, tokenizer, args.max_entity, 
                                        args.max_fact, args.max_enc_len, args.max_dec_len, args.encoder, args.lower_case)
        else:            
            raise NotImplementedError("This dataset is not yet supported")

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

        model.eval()

        loss_func = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
        loss_func.to(args.device)

        PPLs = []
        for step, batch in enumerate(eval_dataloader):
            data = tuple(Variable(t).to(args.device) for t in batch)
            target = data[-1]

            logits = model(*data[:-1])
            loss = loss_func(logits.view(-1, logits.shape[-1]), target.view(-1))

            mask = (target.view(-1) != tokenizer.pad_token_id).float()
            loss_instance = (loss * mask).sum(-1) / mask.sum(-1)

            PPL = torch.exp(loss_instance).mean().item()
            PPLs.append(PPL)

            sys.stdout.write("finished {}/{}; PPL {} \r".format(step, len(eval_dataloader), sum(PPLs) / len(PPLs)))

        print("Overall PPL = {}".format(sum(PPLs) / len(PPLs)))
    
    else:
        raise NotImplementedError("This option is not yet supported")
