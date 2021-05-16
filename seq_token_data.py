import json
from transformers import AutoTokenizer, BertModel
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import os  

tokenzer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')


def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def get_bert_input(input_sender):
    inputs = tokenzer(input_sender, return_tensors ='pt', padding='max_length', truncation=True, max_length=256)
    output = model(**inputs)
    return output[0].detach().numpy()

def read_data(path):
    sender_list = list()
    input_dict = dict()
    temp = dict()
    with open(path, encoding='utf-8') as f:  
            data = json.load(f)
    for info in data:
        str_temp = ''
        _id = info['id']
        for text in info['turns']: #擷取每個客戶以及客服的訊息
            for value in text['utterances']: 
                # if len(text['utterances']) > 1:
                str_temp += value
            sender_list.append(get_bert_input(str_temp).tolist()[0])
            str_temp = ''
        for pad in range(len(sender_list), 7):
            sender_list.append(get_bert_input('').tolist()[0])
        temp['sequence'] = sender_list.copy()
        input_dict[_id] = temp.copy()
        sender_list.clear()
        temp.clear()
    return input_dict

def read_data_y(path):
    data_a = dict()
    data_s = dict()
    data_e = dict()
    A_temp = list()
    S_temp = list()
    E_temp = list()
    with open(path, encoding='utf-8') as f:  
            data = json.load(f)
    for info in data:
        id_ = info['id']
        for score in info['annotations']: #將每個id的19個ASE評分分別出來後取中間值 nugget則取出現最多次的
            A_temp.append(score["quality"]['A'])
            S_temp.append(score["quality"]['S'])
            E_temp.append(score["quality"]['E'])
        A = [A_temp.count(-2)/19, A_temp.count(-1)/19, A_temp.count(0)/19, A_temp.count(1)/19, A_temp.count(2)/19]
        S = [S_temp.count(-2)/19, S_temp.count(-1)/19, S_temp.count(0)/19, S_temp.count(1)/19, S_temp.count(2)/19]
        E = [E_temp.count(-2)/19, E_temp.count(-1)/19, E_temp.count(0)/19, E_temp.count(1)/19, E_temp.count(2)/19]
        data_a[id_] = A.copy()
        data_s[id_] = S.copy()
        data_e[id_] = E.copy()
        A.clear()
        S.clear()
        E.clear()
        A_temp.clear()
        S_temp.clear() 
        E_temp.clear()
    return data_a, data_s, data_e


def read_dev_data_x(path):
    sender_list = list()
    input_list = list()
    len_ = list()

    with open(path, encoding='utf-8') as f:  
            data = json.load(f)
    for i in data:
        str_temp = ''
        len_.append(len(i['turns']))
        for text in i['turns']: #擷取每個客戶以及客服的訊息
            for value in text['utterances']: 
                str_temp += value
            sender_list.append(get_bert_input(str_temp).tolist()[0])
            str_temp = '' 
        for pad in range(len(sender_list), 7):
            sender_list.append(get_bert_input('').tolist()[0])
        input_list.append(sender_list.copy())
        sender_list.clear()
    return input_list, len_

def read_dev_data_y(path):
    data_a = dict()
    data_s = dict()
    data_e = dict()
    A_temp = list()
    S_temp = list()
    E_temp = list()
    id_ = list()
    with open(path, encoding='utf-8') as f:  
            data = json.load(f)
    for info in data:
        id_.append(info['id'])
        for score in info['annotations']: #將每個id的19個ASE評分分別出來後取中間值 nugget則取出現最多次的
            A_temp.append(score["quality"]['A'])
            S_temp.append(score["quality"]['S'])
            E_temp.append(score["quality"]['E'])
        A = [A_temp.count(-2)/19, A_temp.count(-1)/19, A_temp.count(0)/19, A_temp.count(1)/19, A_temp.count(2)/19]
        S = [S_temp.count(-2)/19, S_temp.count(-1)/19, S_temp.count(0)/19, S_temp.count(1)/19, S_temp.count(2)/19]
        E = [E_temp.count(-2)/19, E_temp.count(-1)/19, E_temp.count(0)/19, E_temp.count(1)/19, E_temp.count(2)/19]
        data_a[id_] = A.copy()
        data_s[id_] = S.copy()
        data_e[id_] = E.copy()
        A.clear()
        S.clear()
        E.clear()
        A_temp.clear()
        S_temp.clear() 
        E_temp.clear()
    return data_a, data_s, data_e, id_


def select_data(key):
    with open('./select_{}_id.json'.format(key)) as f:
        id_data = json.load(f)
    if key == 'a':
        sam_neg_two = [i for i in range(301)]
        sam_neg_one = [i for i in range(274)]
        sam_zero = [i for i in range(2400)]
        sam_one = [i for i in range(457)]
        sam_two = [i for i in range(268)]
        neg_two = random.sample(sam_neg_two, 268)
        neg_one = random.sample(sam_neg_one, 268)
        zero = random.sample(sam_zero, 268)
        one = random.sample(sam_one, 268)
        two = random.sample(sam_two, 268)
    elif key == 's':
        sam_neg_two = [i for i in range(363)]
        sam_neg_one = [i for i in range(341)]
        sam_zero = [i for i in range(2425)]
        sam_one = [i for i in range(375)]
        sam_two = [i for i in range(196)]
        neg_two = random.sample(sam_neg_two, 196)
        neg_one = random.sample(sam_neg_one, 196)
        zero = random.sample(sam_zero, 196)
        one = random.sample(sam_one, 196)
        two = random.sample(sam_two, 196)
    elif key == 'e':
        sam_neg_two = [i for i in range(434)]
        sam_neg_one = [i for i in range(261)]
        sam_zero = [i for i in range(1402)]
        sam_one = [i for i in range(1363)]
        sam_two = [i for i in range(240)]
        neg_two = random.sample(sam_neg_two, 240)
        neg_one = random.sample(sam_neg_one, 240)
        zero = random.sample(sam_zero, 240)
        one = random.sample(sam_one, 240)
        two = random.sample(sam_two, 240)
    total = list()
    for i in neg_two:
        total.append(id_data['0'][i])
    for i in neg_one:
        total.append(id_data['1'][i])
    for i in zero:
        total.append(id_data['2'][i])
    for i in one:
        total.append(id_data['3'][i])
    for i in two:
        total.append(id_data['4'][i])

    random.shuffle(total)
    return total


def data(select_id, x_temp, y_temp):
    x = list()
    for _id in select_id:
        x_temp[_id]['label'] = torch.tensor(y_temp[_id].copy(), dtype = torch.float)
        x.append(x_temp[_id].copy())
    return x

# 這個函式的輸入 `samples` 是一個 list，裡頭的每個 element 都是
# 剛剛定義的 `FakeNewsDataset` 回傳的一個樣本，每個樣本都包含 3 tensors：
# - tokens_tensor
# - segments_tensor
# - label_tensor
# 它會對前兩個 tensors 作 zero padding，並產生前面說明過的 masks_tensors
def create_mini_batch(samples):
    sequence_tensors = [s['sequence'] for s in samples]
    # 測試集有 labels
    if samples[0]['label'] is not None:
        label_ids = torch.stack([s['label'] for s in samples])
    else:
        label_ids = None
    
    # zero pad 到同一序列長度
    sequence_tensors = pad_sequence(sequence_tensors, 
                                  batch_first=True)

    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    
    return sequence_tensors, label_ids



x_temp = read_data('./dataset/train_cn.json')
data_a, data_s, data_e = read_data_y('./dataset/train_cn.json')
select_a = select_data('a')
select_s = select_data('s')
select_e = select_data('e')
x_a = data(select_a, x_temp, data_a)
x_s = data(select_s, x_temp, data_s)
x_e = data(select_e, x_temp, data_e)
BATCH_SIZE = 16
trainloader = DataLoader(x_a, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch)

data = next(iter(trainloader))
sequence_tensors, label_ids = data
pass