import torch
import json
import math
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, BertModel

tokenzer = AutoTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese')
device = torch.device('cuda:0')
select_a = list()
select_s = list()
select_e = list()
class BiLSTM(torch.nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.label = 5
        self.bilstm = torch.nn.GRU(input_size = 768, hidden_size =256, batch_first= True, bidirectional= True)
        self.DropL = torch.nn.Dropout(0.3)
        self.bilstm2 = torch.nn.GRU(input_size = 512, hidden_size =128, batch_first= True, bidirectional= True)
        self.DropL2 = torch.nn.Dropout(0.3)
        self.dense = torch.nn.Linear(256, 1024)
        self.DropD = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(1024, self.label)

    def forward(self, x, y=None):
        layout1 = torch.nn.functional.tanh(self.bilstm(x)[0])
        layout1 = self.DropL(layout1)
        layout2 = torch.nn.functional.tanh(self.bilstm2(layout1)[0][0][-1])
        layout2 = self.DropL2(layout2)
        layout3 = torch.nn.functional.relu(self.dense(layout2))
        layout3 = self.DropD(layout3)
        pred = torch.nn.functional.softmax(self.classifier(layout3))
        
        # if y != None:
        #     # ow_ = 0
        #     # ow = 0
        #     # pi = [i for i in range(len(y)) if y[i] > 0]
        #     # pi_ = [i for i in range(len(pred)) if pred[i] > 0]
        #     # for i in pi:
        #     #     for j in range(5):
        #     #         ow += abs(i-j)*math.pow((pred[j] - y[j]), 2)
        #     # for i in pi_:
        #     #     for j in range(5):
        #     #         ow_ += abs(i-j)*math.pow((pred[j] - y[j]), 2)
        #     # ow_ = ow_ / len(pi_)
        #     # ow = ow / len(pi_)
        #     # sod = (ow_ + ow) / 2
        #     # rsnod = math.sqrt((sod/4))
        #     loss_fn = torch.nn.MSELoss()
        #     rsnod = loss_fn(pred, y)
        #     return rsnod
        # else:
        return pred

def get_bert_input(input_sender):
    inputs = tokenzer(input_sender, return_tensors ='pt', padding='max_length', truncation=True, max_length=256)
    output = bert_model(**inputs)
    return output[1].detach().numpy().tolist()

#將資料輸入給bert取輸出
def read_train_data(select, case):
    with open('./dataset/train_cn.json', encoding='utf-8') as f:  
        train_data = json.load(f)
    sender_list = list()
    total_x = list()
    y_temp = list()
    total_y = list()
    for info in train_data:
        if info['id'] not in select:
            for text in info['turns']: #擷取每個客戶以及客服的訊息
                str_temp = ''
                for value in text['utterances']: 
                    str_temp += value
                sender_list.append(get_bert_input(str_temp)[0])
            for pad in range(len(sender_list), 7):
                sender_list.append(get_bert_input('')[0])
            total_x.append(sender_list.copy())

            for score in info['annotations']: #將每個id的19個ASE評分分別出來後取中間值 nugget則取出現最多次的
                y_temp.append(score["quality"][case])    
            total_y.append([y_temp.count(-2)/19, y_temp.count(-1)/19, y_temp.count(0)/19, y_temp.count(1)/19, y_temp.count(2)/19])
            sender_list.clear()
            y_temp.clear()

    x = torch.tensor(total_x, dtype=torch.float, device=device)
    y = torch.tensor(total_y, dtype=torch.float, device=device)
    data = TensorDataset(x, y)
    return DataLoader(dataset = data, batch_size = 32, shuffle = True)

def read_dev_data(path):
    sender_list = list()
    input_list = list()
    len_ = list()
    _id = list()
    with open(path, encoding='utf-8') as f:  
            data = json.load(f)
    for i in data:
        str_temp = ''
        _id.append(i['id'])
        len_.append(len(i['turns']))
        for text in i['turns']: #擷取每個客戶以及客服的訊息
            for value in text['utterances']: 
                str_temp += value
            sender_list.append(get_bert_input(str_temp)[0])
            str_temp = '' 
        for pad in range(len(sender_list), 7):
            sender_list.append(get_bert_input('')[0])
        input_list.append(sender_list.copy())
        sender_list.clear()
    test_data = torch.tensor(input_list, dtype=torch.float, device=device)
    return test_data, len_, _id


# 資料平均擷取以及打亂
def shuffle_data(key):
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

def training_and_predictions(x, test, quality):                
    pred = list()
    step_pred = list()
    models = BiLSTM()
    models.to(device)
    models.train()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(models.parameters(), lr=0.0001)
    for epoch in range(150):
        optimizer.zero_grad()
        for step, (batch_x, batch_y) in enumerate(x):
            for batch in range(batch_x.shape[0]):
                inputs = batch_x[batch].unsqueeze(0)
                step_pred.append(models(inputs, batch_y[batch]).tolist())
            batch_pred = torch.tensor(step_pred, dtype=torch.float, device=device).requires_grad_()
            loss = loss_fn(batch_pred, batch_y)
            loss.backward()
            optimizer.step()
            step_pred.clear()
        print(f'epoch:{epoch} | epoch_rnsod:{loss}')
    with torch.no_grad():
        for test_x in test:
            tests = test_x.unsqueeze(0)
            pred.append(models(tests).tolist())
    torch.save(models.state_dict(), f=f'./model{quality}.pth')
    return pred


select_a = shuffle_data('a')
select_s = shuffle_data('s')
select_e = shuffle_data('e')

data_a = read_train_data(select_a, 'A')
data_s = read_train_data(select_s, 'S')
data_e = read_train_data(select_e, 'E')


test, test_len, test_id = read_dev_data('./dataset/dev_cn.json')
result_a = training_and_predictions(data_a, test, 'A')
result_s = training_and_predictions(data_s, test, 'S')
result_e = training_and_predictions(data_e, test, 'E')

temp = dict()
total_list = list()

for i in range(390):
    total_list.append(dict())
    total_list[i]['id'] = test_id[i]
    total_list[i]['quality'] = dict()
    total_list[i]['quality']['A'] = dict()
    total_list[i]['quality']['S'] = dict()
    total_list[i]['quality']['E'] = dict() 
    total_list[i]['quality']['A']['-2'] = result_a[i][0]
    total_list[i]['quality']['A']['-1'] = result_a[i][1]
    total_list[i]['quality']['A']['0'] = result_a[i][2]
    total_list[i]['quality']['A']['1'] = result_a[i][3]
    total_list[i]['quality']['A']['2'] = result_a[i][4]
    total_list[i]['quality']['E']['-2'] = result_e[i][0]
    total_list[i]['quality']['E']['-1'] = result_e[i][1]
    total_list[i]['quality']['E']['0'] = result_e[i][2]
    total_list[i]['quality']['E']['1'] = result_e[i][3]
    total_list[i]['quality']['E']['2'] = result_e[i][4]
    total_list[i]['quality']['S']['-2'] = result_s[i][0]
    total_list[i]['quality']['S']['-1'] = result_s[i][1]
    total_list[i]['quality']['S']['0'] = result_s[i][2]
    total_list[i]['quality']['S']['1'] = result_s[i][3]
    total_list[i]['quality']['S']['2'] = result_s[i][4]


with open('result_ptgru.json', 'w+', encoding = 'utf-8') as f:
    f.write(json.dumps(total_list, indent = 2))
