import torch
from tqdm import trange, tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from Model import TransformerDecoder, GatedTransformerDecoder, GPT2Wrapper
import argparse
from DataLoader import *
from transformers import BertTokenizer, GPT2Tokenizer, XLNetTokenizer
from transformers import GPT2LMHeadModel
import json
from types import SimpleNamespace
from torch.autograd import Variable
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
import math
import os
import nltk
import sys
from transformers import get_linear_schedule_with_warmup
from nltk.tokenize import word_tokenize


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, default="train",
                        help="whether to train or test the model", choices=['train', 'test', 'visualize', 'LM', 'compute_bleu', 'few_shot'])
    parser.add_argument('--batch_size', type=int, default=64, help="the embedding dimension")
    parser.add_argument('--train_path', type=str, default="", help="the embedding dimension")
    parser.add_argument('--test_path', type=str, default="", help="the embedding dimension")
    parser.add_argument('--challenge_path', type=str, default="", help="the embedding dimension")    
    parser.add_argument('--knowledge_path', type=str, default="../knowledge.json", help="the embedding dimension")
    parser.add_argument('--epochs', type=int, default=10, help="the embedding dimension")
    parser.add_argument('--save_every_n_epochs', type=int, default=1, help="the embedding dimension")
    parser.add_argument('--n_head', type=int, default=8, help="the embedding dimension")
    parser.add_argument('--tokenizer_dir', type=str, default='GPT2_tokenizer', help="tokenizer loaded from")
    parser.add_argument('--n_layers', type=int, default=5, help="the embedding dimension")
    parser.add_argument('--dataset', type=str, default='', help="the embedding dimension")
    parser.add_argument('--config', type=str, default='', help="config file for the embedding layer")
    parser.add_argument('--scratch_embedding', default=False, action='store_true', help="whether to load bert embedding")    
    parser.add_argument('--embedding_path', type=str, default='', help="pretrained BERT embedding weight")
    parser.add_argument('--output_dir', type=str, default='checkpoint_template_gpt2', help="maximum length of the decoding part")
    parser.add_argument('--logging_steps', type=int, default=20, help="maximum length of the decoding part")
    parser.add_argument('--printing_steps', type=int, default=500, help="maximum length of the decoding part")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="maximum length of the decoding part")
    parser.add_argument('--max_enc_len', type=int, default=640, help="maximum length of the encoding part")
    parser.add_argument('--max_dec_len', type=int, default=72, help="maximum length of the decoding part")
    parser.add_argument('--max_entity', type=int, default=8, help="maximum length of the decoding part")    
    parser.add_argument('--max_fact', type=int, default=12, help="maximum length of the decoding part")
    parser.add_argument('--starting_epoch', type=int, default=0, help="maximum length of the decoding part")
    parser.add_argument('--load_from', type=str, default='', help="maximum length of the decoding part")
    parser.add_argument('--num_workers', type=int, default=8, help="maximum length of the decoding part")
    parser.add_argument('--beam_size', type=int, default=2, help="the embedding dimension")    
    parser.add_argument('--bleu', type=int, default=4, help="the embedding dimension")    
    parser.add_argument('--hidden_size', type=int, default=None, help="the embedding dimension")    
    parser.add_argument('--percent', default=1.0, type=float, help='Which experiment you are doing')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_opt()
    device = torch.device('cuda')
    args.device = device

    if args.train_path == '':
        args.train_path = os.path.join('dataset', args.dataset, 'train.json')
    if args.test_path == '':
        args.test_path = os.path.join('dataset', args.dataset, 'val.json')
    if args.challenge_path == '':
        args.challenge_path = os.path.join('dataset', args.dataset, 'test.json')

    args.embedding_path = os.path.join(args.tokenizer_dir, 'embedding.bin')
    print(args)

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_dir)
    banwords = tokenizer.convert_tokens_to_ids(['It', 'She', 'They', 'He', 'it', 'she', 'he', 'they'])

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    model = torch.nn.DataParallel(model)
    model.to(device)
    wrapper_model = GPT2Wrapper(model)

    if args.load_from != '':
        reloaded = torch.load(args.load_from)
        model.load_state_dict(reloaded)
        print("Loading model from {}".format(args.load_from))

    folder = 'checkpoint_{}'.format(args.dataset)    
    if args.option == 'few_shot':
        args.output_dir = '{}/template_gpt2_{}_fewshot{}'.format(folder, args.dataset, args.percent)
    else:
        args.output_dir = '{}/template_gpt2_{}'.format(folder, args.dataset)

    if args.option in ['train', 'few_shot']:
        train_data = GPTDataset(args.train_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.percent)

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
                table_inputs, target_inputs, target = data

                model.zero_grad()
                optimizer.zero_grad()
                logits = wrapper_model.forward(table_inputs, target_inputs)
                
                logits = logits[:, -target.shape[1]:, :].contiguous()
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
                    decoded_sentence =  wrapper_model.greedy_search(table_inputs[:1, :], tokenizer, 30)
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

    elif args.option in ['test', 'challenge']:
        if args.option == 'test':
            eval_data = GPTDataset(args.test_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.percent)
        else:
            eval_data = GPTDataset(args.challenge_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.percent)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size, num_workers=args.num_workers)
        model.eval()

        results = []
        list_of_references = []
        list_of_hypothesis = []
        for step, batch in enumerate(eval_dataloader):
            data = tuple(Variable(t).to(args.device) for t in batch)
            table_inputs, _, _ = data
            
            if args.beam_size == 1:
                result = wrapper_model.greedy_search(table_inputs, tokenizer, args.max_dec_len)
            else:
                result = wrapper_model.beam_search([table_inputs], tokenizer, args.beam_size, args.max_dec_len)

            for offset, r in enumerate(result):
                sent = tokenizer.decode(r, clean_up_tokenization_spaces=True)
                if '[EOS]' in sent:
                    sent = sent[:sent.index('[EOS]')].strip()

                idx = step * args.batch_size + offset
                references = eval_data.get_reference(idx)

                if not isinstance(references[0], list):
                    references = [references]

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

    elif args.option == 'visualize':
        eval_data = GPTDataset(args.test_path, tokenizer, args.max_entity, args.max_fact, args.max_enc_len, args.max_dec_len, args.percent)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size, num_workers=args.num_workers)
        model.eval()
        
        for step, batch in enumerate(eval_dataloader):
            data = tuple(Variable(t).to(args.device) for t in batch)
            table_inputs, _, _ = data
            
            if args.beam_size == 1:
                result = wrapper_model.greedy_search(table_inputs, tokenizer, args.max_dec_len)
            else:
                result = wrapper_model.beam_search([table_inputs], tokenizer, args.beam_size, args.max_dec_len)

            for offset, r in enumerate(result):
                sent = tokenizer.decode(r, clean_up_tokenization_spaces=True)
                if '[EOS]' in sent:
                    sent = sent[:sent.index('[EOS]')]

                idx = step * args.batch_size + offset
                references = eval_data.get_reference(idx)

                entities = eval_data.get_entities(idx)

                print("ENTITIES |||||", entities)
                print("DECODED |||||", sent)
                print("REFERENCE |||||", " ".join(references[0]))
    else:
        raise NotImplementedError("This option is not yet supported")
