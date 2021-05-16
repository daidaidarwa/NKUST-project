import json
from transformers import AutoTokenizer, BertModel
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
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
    return output[1].detach().numpy()

def read_data(path):
    sender_list = list()
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
    train = np.array(sender_list, dtype = 'float')
    return train

def read_data_y(path):
    data_a = list()
    data_s = list()
    data_e = list()
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
        data_a.append(A.copy())
        data_s.append(S.copy())
        data_e.append(E.copy())
        A.clear()
        S.clear()
        E.clear()
        A_temp.clear()
        S_temp.clear() 
        E_temp.clear()
    a = np.array(data_a, dtype = 'float')
    s = np.array(data_s, dtype = 'float')
    e = np.array(data_e, dtype = 'float') 
    return a, s, e


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
        input_list = np.array(sender_list, dtype = 'float')
    return input_list, len_

def read_dev_data_y(path):
    list_a = list()
    list_s = list()
    list_e = list()
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
        list_a.append(A.copy())
        list_s.append(S.copy())
        list_e.append(E.copy())
        A.clear()
        S.clear()
        E.clear()
        A_temp.clear()
        S_temp.clear() 
        E_temp.clear()
    np_a = np.array(list_a, dtype='float')
    np_s = np.array(list_s, dtype='float')
    np_e = np.array(list_e, dtype='float')
    return np_a, np_s, np_e, id_

def QualityA_model(x, y, input_test):
    keras_model = Sequential()
    keras_model.add(Dense(768, input_dim = 768, activation = 'relu'))
    keras_model.add(Dropout(0.25))
    keras_model.add(Dense(1536, activation = 'relu'))
    keras_model.add(Dropout(0.25))
    keras_model.add(Dense(5, activation = 'softmax'))
    print(keras_model.summary())
    keras_model.compile(optimizer=optimizers.SGD(learning_rate=0.05), loss='mse', metrics=['accuracy'])    
    a_process = keras_model.fit(x, y, epochs = 150, batch_size = 125, validation_split=0.2)
    # show_train_history(a_process, 'loss', 'val_loss')
#     show_train_history(a_process, 'accuracy', 'val_accuracy')
    result = keras_model.predict(input_test)
    return result

train_data = read_data('./dataset/train_cn.json')
A_train, S_train, E_train = read_data_y('./dataset/train_cn.json')
test, len_ = read_dev_data_x('./dataset/dev_cn.json')
import os  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

result_s = QualityA_model(train_data, S_train, test)
result_a = QualityA_model(train_data, A_train, test)
result_e = QualityA_model(train_data, E_train, test)
A_, S_, E_, test_id = read_dev_data_y('./dataset/dev_cn.json')
temp = dict()
total_list = list()

for i in range(390):
    total_list.append(dict())
    total_list[i]['id'] = test_id[i]
    total_list[i]['quality'] = dict()
    total_list[i]['quality']['A'] = dict()
    total_list[i]['quality']['S'] = dict()
    total_list[i]['quality']['E'] = dict() 
    total_list[i]['quality']['A']['-2'] = result_a[i].tolist()[0]
    total_list[i]['quality']['A']['-1'] = result_a[i].tolist()[1]
    total_list[i]['quality']['A']['0'] = result_a[i].tolist()[2]
    total_list[i]['quality']['A']['1'] = result_a[i].tolist()[3]
    total_list[i]['quality']['A']['2'] = result_a[i].tolist()[4]
    total_list[i]['quality']['E']['-2'] = result_e[i].tolist()[0]
    total_list[i]['quality']['E']['-1'] = result_e[i].tolist()[1]
    total_list[i]['quality']['E']['0'] = result_e[i].tolist()[2]
    total_list[i]['quality']['E']['1'] = result_e[i].tolist()[3]
    total_list[i]['quality']['E']['2'] = result_e[i].tolist()[4]
    total_list[i]['quality']['S']['-2'] = result_s[i].tolist()[0]
    total_list[i]['quality']['S']['-1'] = result_s[i].tolist()[1]
    total_list[i]['quality']['S']['0'] = result_s[i].tolist()[2]
    total_list[i]['quality']['S']['1'] = result_s[i].tolist()[3]
    total_list[i]['quality']['S']['2'] = result_s[i].tolist()[4]

with open('quality_dq.json', 'w', encoding = 'utf-8') as f:
    f.write(json.dumps(total_list, indent = 2))