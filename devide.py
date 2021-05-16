import json

dev_id = list()
cls_token = list()
seq_token = list()
combine = dict()
result = list()
count = 0
with open('./dev_id.txt', 'r', encoding='utf-8') as f:
    for i in f.readlines():
        dev_id.append(i.strip('\n')) 

with open('./result_qualityCLSGRU.json', 'r', encoding='utf-8') as f:
    cls_token = json.load(f)

with open('./result_qualitySEQGRU.json', 'r', encoding='utf-8') as f:
    seq_token = json.load(f)

for idx in dev_id:
    result.append(dict())
    result[count]['id'] = ''
    result[count]['quality'] = dict()
    result[count]['quality']['A'] = dict()
    result[count]['quality']['S'] = dict()
    result[count]['quality']['E'] = dict()
    result[count]['id'] = idx
    result[count]['quality']['A']['-2'] = (cls_token[dev_id.index(idx)]['quality']['A']['-2'] + seq_token[dev_id.index(idx)]['quality']['A']['-2']) / 2
    result[count]['quality']['A']['-1'] = (cls_token[dev_id.index(idx)]['quality']['A']['-1'] + seq_token[dev_id.index(idx)]['quality']['A']['-1']) / 2
    result[count]['quality']['A']['0'] = (cls_token[dev_id.index(idx)]['quality']['A']['0'] + seq_token[dev_id.index(idx)]['quality']['A']['0']) / 2
    result[count]['quality']['A']['1'] = (cls_token[dev_id.index(idx)]['quality']['A']['1'] + seq_token[dev_id.index(idx)]['quality']['A']['1']) / 2
    result[count]['quality']['A']['2'] = (cls_token[dev_id.index(idx)]['quality']['A']['2'] + seq_token[dev_id.index(idx)]['quality']['A']['2']) / 2
    result[count]['quality']['E']['-2'] = (cls_token[dev_id.index(idx)]['quality']['E']['-2'] + seq_token[dev_id.index(idx)]['quality']['E']['-2']) / 2
    result[count]['quality']['E']['-1'] = (cls_token[dev_id.index(idx)]['quality']['E']['-1'] + seq_token[dev_id.index(idx)]['quality']['E']['-1']) / 2
    result[count]['quality']['E']['0'] = (cls_token[dev_id.index(idx)]['quality']['E']['0'] + seq_token[dev_id.index(idx)]['quality']['E']['0']) / 2
    result[count]['quality']['E']['1'] = (cls_token[dev_id.index(idx)]['quality']['E']['1'] + seq_token[dev_id.index(idx)]['quality']['E']['1']) / 2 
    result[count]['quality']['E']['2'] = (cls_token[dev_id.index(idx)]['quality']['E']['2'] + seq_token[dev_id.index(idx)]['quality']['E']['2']) / 2
    result[count]['quality']['S']['-2'] = (cls_token[dev_id.index(idx)]['quality']['S']['-2'] + seq_token[dev_id.index(idx)]['quality']['S']['-2']) / 2
    result[count]['quality']['S']['-1'] = (cls_token[dev_id.index(idx)]['quality']['S']['-1'] + seq_token[dev_id.index(idx)]['quality']['S']['-1']) / 2
    result[count]['quality']['S']['0'] = (cls_token[dev_id.index(idx)]['quality']['S']['0'] + seq_token[dev_id.index(idx)]['quality']['S']['0']) / 2
    result[count]['quality']['S']['1'] = (cls_token[dev_id.index(idx)]['quality']['S']['1'] + seq_token[dev_id.index(idx)]['quality']['S']['1']) / 2
    result[count]['quality']['S']['2'] = (cls_token[dev_id.index(idx)]['quality']['S']['2'] + seq_token[dev_id.index(idx)]['quality']['S']['2']) / 2
    count += 1

with open('./result_GRU.json', 'w+', encoding='utf-8') as f:
    f.write(json.dumps(result, indent = 2))