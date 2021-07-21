import torch
import json
from transformers import AutoTokenizer, BertModel

tokenzer = AutoTokenizer.from_pretrained('bert-base-chinese')
bert_model = BertModel.from_pretrained('bert-base-chinese')
device = torch.device('cuda:0')

class BiLSTM(torch.nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.label = 5
        self.bilstm = torch.nn.GRU(input_size = 768, hidden_size =256, batch_first= True, bidirectional= True)
        self.DropL = torch.nn.Dropout(0.3)
        self.bilstm2 = torch.nn.GRU(input_size = 512, hidden_size =256, batch_first= True, bidirectional= True)
        self.muti_ha = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.DropL2 = torch.nn.Dropout(0.3)
        self.dense = torch.nn.Linear(512, 1024)
        self.DropD = torch.nn.Dropout(0.3)
        
        self.classifier = torch.nn.Linear(256, self.label)

    def forward(self, x, y=None):
        layout1 = self.bilstm(x)[0]
        layout1 = self.DropL(layout1)
        layout2 = self.bilstm2(layout1)[0]
        layout2 = self.DropL2(layout2)
        # layout3 = torch.nn.functional.relu(self.dense(layout2), inplace=True)
        # layout3 = self.DropD(layout3)
        layout = layout2.reshape(7, -1, 512)
        attoutput, attn_output_weights = self.muti_ha(layout, layout, layout)
        attoutput = attoutput.reshape(-1, 7, 256)
        pred = torch.nn.functional.softmax(self.classifier(attoutput), -1)
        y_pred = pred[:, -1]
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
        return y_pred


def get_bert_input(input_sender):
    inputs = tokenzer(input_sender, return_tensors ='pt', padding='max_length', truncation=True, max_length=256)
    output = bert_model(**inputs)
    return output[1].detach().numpy().tolist()

def read_dev_data(path):
    sender_list = list()
    input_list = list()
    len_ = list()
    _id = list()
    with open(path, encoding='utf-8') as f:  
            data = json.load(f)
    for i in data:
        _id.append(i['id'])
        len_.append(len(i['turns']))
        for text in i['turns']: #擷取每個客戶以及客服的訊息
            str_temp = '' 
            for value in text['utterances']: 
                str_temp += value
            sender_list.append(get_bert_input(str_temp)[0])
        for pad in range(len(sender_list), 7):
            sender_list.append(get_bert_input('')[0])
        input_list.append(sender_list.copy())
        sender_list.clear()
    test_data = torch.tensor(input_list, dtype=torch.float, device=device)
    return test_data, len_, _id

def load_and_pred(key, test):
    models = BiLSTM()
    models.to(device) 
    models.load_state_dict(torch.load(f'model{key}.pth'))
    models.eval()
    with torch.no_grad():
        test_output = models(test)
    return test_output.tolist()

test, test_len, test_id = read_dev_data('./dataset/dev_cn.json')
a = test.view(7, -1, 768)
result_a = load_and_pred('A', test)
# result_s = load_and_pred('S', test)
# result_e = load_and_pred('E', test)
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
    # total_list[i]['quality']['E']['-2'] = result_e[i][0]
    # total_list[i]['quality']['E']['-1'] = result_e[i][1]
    # total_list[i]['quality']['E']['0'] = result_e[i][2]
    # total_list[i]['quality']['E']['1'] = result_e[i][3]
    # total_list[i]['quality']['E']['2'] = result_e[i][4]
    # total_list[i]['quality']['S']['-2'] = result_s[i][0]
    # total_list[i]['quality']['S']['-1'] = result_s[i][1]
    # total_list[i]['quality']['S']['0'] = result_s[i][2]
    # total_list[i]['quality']['S']['1'] = result_s[i][3]
    # total_list[i]['quality']['S']['2'] = result_s[i][4]


with open('result_grurs.json', 'w+', encoding = 'utf-8') as f:
    f.write(json.dumps(total_list, indent = 2))