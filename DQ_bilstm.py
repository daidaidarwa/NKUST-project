import json
from transformers import AutoTokenizer, BertModel
import tensorflow as tf
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
    return output[1].detach().numpy().tolist()

def read_data(path):
    sender_list = list()
    input_dict = dict()
    temp = list()
    with open(path, encoding='utf-8') as f:  
            data = json.load(f)
    for info in data:
        str_temp = ''
        _id = info['id']
        for text in info['turns']: #擷取每個客戶以及客服的訊息
            for value in text['utterances']: 
                # if len(text['utterances']) > 1:
                str_temp += value
            sender_list.append(get_bert_input(str_temp)[0])
            str_temp = ''
        for pad in range(len(sender_list), 7):
            sender_list.append(get_bert_input('')[0])
        input_dict[_id] = sender_list.copy()
        sender_list.clear()
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
            sender_list.append(get_bert_input(str_temp)[0])
            str_temp = '' 
        for pad in range(len(sender_list), 7):
            sender_list.append(get_bert_input('')[0])
        input_list.append(sender_list.copy())
        sender_list.clear()
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
    y = list()
    for _id in select_id:
        x.append(x_temp[_id])
        y.append(y_temp[_id])
    npx = np.array(x, dtype='float')
    npy = np.array(y, dtype='float')
    return npx, npy

def my_losses(y_true, y_pred):
    print(y_pred[1])
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)

def Score_a_val(x, y, test_x, batch, epoch):    
    keras_model = tf.keras.Sequential()
    # keras_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(768), input_shape=(x.shape[1], x.shape[2])))
    # keras_model.add(tf.keras.layers.Dropout(0.3))
    keras_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences = True), input_shape=(x.shape[1], x.shape[2])))
    keras_model.add(tf.keras.layers.Dropout(0.3))
    # keras_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences = True)))
    # keras_model.add(tf.keras.layers.Dropout(0.3))
    keras_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128)))
    keras_model.add(tf.keras.layers.Dropout(0.3))
    keras_model.add(tf.keras.layers.Dense(1024, activation = 'relu'))
    keras_model.add(tf.keras.layers.Dropout(0.2))
    keras_model.add(tf.keras.layers.Dense(5, activation = 'softmax'))
    print(keras_model.summary())
    keras_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=my_losses, metrics=[tf.keras.metrics.MeanSquaredError()])    
    a_process = keras_model.fit(x, y, epochs = epoch, batch_size = batch, validation_split=0.2)
    # show_train_history(a_process, 'loss', 'val_loss')
#     show_train_history(a_process, 'accuracy', 'val_accuracy')
    result = keras_model.predict(test_x)
    return result

x_temp = read_data('./dataset/train_cn.json')
data_a, data_s, data_e = read_data_y('./dataset/train_cn.json')
select_a = select_data('a')
select_s = select_data('s')
select_e = select_data('e')

x_a, y_a = data(select_a, x_temp, data_a)
x_s, y_s = data(select_s, x_temp, data_s)
x_e, y_e = data(select_e, x_temp, data_e)


test_X, test_len = read_dev_data_x('./dataset/dev_cn.json')
test_a, test_s, test_e, test_id = read_dev_data_y('./dataset/dev_cn.json')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
test = np.array(test_X, dtype = 'float')

result_a = Score_a_val(x_a, y_a, test, 32, 150)
result_s = Score_a_val(x_s, y_s, test, 32, 150)
result_e = Score_a_val(x_e, y_e, test, 32, 150)

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


with open('result_qualityCLSGRU.json', 'w', encoding = 'utf-8') as f:
    f.write(json.dumps(total_list, indent = 2))
