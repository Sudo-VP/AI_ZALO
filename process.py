import json,time
from bs4 import BeautifulSoup
import re
import numpy as np
from vncorenlp import VnCoreNLP
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaForSequenceClassification, RobertaConfig, AdamW
from transformers import (TFBertForSequenceClassification, 
                          BertTokenizer,
                          TFRobertaForSequenceClassification, 
                          RobertaTokenizer)
import tensorflow as tf
import random
from tqdm import tqdm_notebook
# phobert = AutoModel.from_pretrained("vinai/phobert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
MAX_LEN = 125
annotator = VnCoreNLP(address="http://127.0.0.1", port=9000) 



def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

data_ = []

with open('train/train.jsonl',encoding="utf8") as train_file:
    jsonl_content = train_file.readlines()
    result = [json.loads(jline) for jline in jsonl_content]
    # print(result[0]['html_annotation'])
    for i,data in enumerate(result):

        dict_ = []
        list_ = []
        for html in data["html_annotation"]:
            soup = BeautifulSoup(html)

            events = soup.find_all("span", {"class": "tag"})
            
            for e in events:
                dict_.append({"event_type": e["data"],"event_id":e["event_id"],"text": e.text})
        temp = [k['text'].replace('.','') for k in dict_]
        for sent in [u['text'] for u in data['original_doc']['_source']['body']]:
            try:
                if sent.replace('.','') in temp:
                    t=list(filter(lambda x: (x['text'].replace('.','') == sent.replace('.','')), dict_))[0]
                    t_={'event_type': t['event_type'], 'event_id':t['event_id'] , 'text': ' '.join([' '.join(u) for u in annotator.tokenize(t['text'])]),'event_label':[0]*6}
                    if 'goal_info' in t['event_type']:
                        t_['event_label'][1]=1
                    if 'match_info' in t['event_type']:
                        t_['event_label'][2]=1
                    if 'match_result' in t['event_type']:
                        t_['event_label'][3]=1
                    if 'card_info' in t['event_type']:
                        t_['event_label'][4]=1
                    if 'substitution' in t['event_type']:
                        t_['event_label'][5]=1
                    list_.append(t_)
                else:
                    list_.append({'event_type': 'Other','event_label':[1,0,0,0,0,0], 'event_id':None , 'text':' '.join([' '.join(u) for u in annotator.tokenize(sent)])})
            except Exception as e:
                print(e)
                break
        data_.extend(list_)

DATA_X = [d['text'] for d in data_]
DATA_Y = [d['event_label'] for d in data_]
train_sents, val_sents, train_labels, val_labels = train_test_split(DATA_X, DATA_Y, test_size=0.1)
train_ids = []
for sent in train_sents:
    encoded_sent = tokenizer.encode(sent)
    train_ids.append(encoded_sent)

val_ids = []
for sent in val_sents:
    encoded_sent = tokenizer.encode(sent)
    val_ids.append(encoded_sent)

train_ids = pad_sequences(train_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
val_ids = pad_sequences(val_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
        # for text in re.split(': | . ',important_data):
        #     print(text)

train_masks = []
for sent in train_ids:
    mask = [int(token_id > 0) for token_id in sent]
    train_masks.append(mask)

val_masks = []
for sent in val_ids:
    mask = [int(token_id > 0) for token_id in sent]
    val_masks.append(mask)
    
train_inputs = torch.tensor(train_ids)
val_inputs = torch.tensor(val_ids)
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)
train_masks = torch.tensor(train_masks)
val_masks = torch.tensor(val_masks)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = SequentialSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=32)   
config = RobertaConfig.from_pretrained("vinai/phobert-base", from_tf=True, num_labels = 6, output_hidden_states=False,)
BERT_SA = TFBertForSequenceClassification.from_pretrained("vinai/phobert-base",config=config)

config = RobertaConfig.from_pretrained(
    "vinai/phobert-base", from_tf=False, num_labels = 6, output_hidden_states=False,
)
BERT_SA = RobertaForSequenceClassification.from_pretrained(
    "vinai/phobert-base",
    config=config
)
# BERT_SA.cuda()
import random
from tqdm import tqdm_notebook
device = 'cpu'
epochs = 10

param_optimizer = list(BERT_SA.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, correct_bias=False)


for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    total_loss = 0
    BERT_SA.train()
    train_accuracy = 0
    nb_train_steps = 0
    train_f1 = 0
    
    for step, batch in tqdm_notebook(enumerate(train_dataloader)):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        print(b_labels)
        BERT_SA.zero_grad()
        outputs = BERT_SA(b_input_ids, 
            token_type_ids=None, 
            attention_mask=b_input_mask, 
            labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_train_accuracy, tmp_train_f1 = flat_accuracy(logits, label_ids)
        train_accuracy += tmp_train_accuracy
        train_f1 += tmp_train_f1
        nb_train_steps += 1
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(BERT_SA.parameters(), 1.0)
        optimizer.step()
        
    avg_train_loss = total_loss / len(train_dataloader)
    print(" Accuracy: {0:.4f}".format(train_accuracy/nb_train_steps))
    print(" F1 score: {0:.4f}".format(train_f1/nb_train_steps))
    print(" Average training loss: {0:.4f}".format(avg_train_loss))

    print("Running Validation...")
    BERT_SA.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    eval_f1 = 0
    for batch in tqdm_notebook(val_dataloader):

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = BERT_SA(b_input_ids, 
            token_type_ids=None, 
            attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy, tmp_eval_f1 = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            eval_f1 += tmp_eval_f1
            nb_eval_steps += 1
    print(" Accuracy: {0:.4f}".format(eval_accuracy/nb_eval_steps))
    print(" F1 score: {0:.4f}".format(eval_f1/nb_eval_steps))
print("Training complete!")