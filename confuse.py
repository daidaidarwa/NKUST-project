import json

bin_ = ['-2', '-1', '0', '1', '2']
A_temp = list()
S_temp = list()
E_temp = list()
A = list()
S = list()
E = list()
dq_preA = list()
dq_preS = list()
dq_preE = list()

cls_preA = list()
cls_preS = list()
cls_preE = list()

seq_preA = list()
seq_preS = list()
seq_preE = list()
with open('./quality_dq.json', 'r', encoding='utf-8') as f:
    dq_data = json.load(f)

with open('./result_qualityRSNOD.json', 'r', encoding='utf-8') as f:
    cls_data = json.load(f)

with open('./result_qualitySEQGRU.json', 'r', encoding='utf-8') as f:
    seq_data = json.load(f)

for dq_bin in dq_data:
    dq_preA.append(max(dq_bin['quality']['A'], key=dq_bin['quality']['A'].get))
    dq_preS.append(max(dq_bin['quality']['S'], key=dq_bin['quality']['S'].get))
    dq_preE.append(max(dq_bin['quality']['E'], key=dq_bin['quality']['E'].get))
dq_data.clear()

for cls_bin in cls_data:
    cls_preA.append(max(cls_bin['quality']['A'], key=cls_bin['quality']['A'].get))
    cls_preS.append(max(cls_bin['quality']['S'], key=cls_bin['quality']['S'].get))
    cls_preE.append(max(cls_bin['quality']['E'], key=cls_bin['quality']['E'].get))
cls_data.clear()

for seq_bin in seq_data:
    seq_preA.append(max(seq_bin['quality']['A'], key=seq_bin['quality']['A'].get))
    seq_preS.append(max(seq_bin['quality']['S'], key=seq_bin['quality']['S'].get))
    seq_preE.append(max(seq_bin['quality']['E'], key=seq_bin['quality']['E'].get))
seq_data.clear()


with open('./dataset/dev_cn.json', encoding='utf-8') as f:  
            data = json.load(f)
for info in data:
    for score in info['annotations']: #將每個id的19個ASE評分分別出來後取中間值 nugget則取出現最多次的
        A_temp.append(score["quality"]['A'])
        S_temp.append(score["quality"]['S'])
        E_temp.append(score["quality"]['E'])
    totalA = [A_temp.count(-2)/19, A_temp.count(-1)/19, A_temp.count(0)/19, A_temp.count(1)/19, A_temp.count(2)/19]
    totalS = [S_temp.count(-2)/19, S_temp.count(-1)/19, S_temp.count(0)/19, S_temp.count(1)/19, S_temp.count(2)/19]
    totalE = [E_temp.count(-2)/19, E_temp.count(-1)/19, E_temp.count(0)/19, E_temp.count(1)/19, E_temp.count(2)/19]
    A.append(bin_[totalA.index(max(totalA))])
    S.append(bin_[totalS.index(max(totalS))])
    E.append(bin_[totalE.index(max(totalE))])
    A_temp.clear()
    S_temp.clear() 
    E_temp.clear()

def show(confuse):

    print('\t-2\t-1\t0\t1\t2')
    print('-2\t{}\t{}\t{}\t{}\t{}'.format(confuse['-2'][0], confuse['-1'][0], confuse['0'][0], confuse['1'][0], confuse['2'][0]))
    print('-1\t{}\t{}\t{}\t{}\t{}'.format(confuse['-2'][1], confuse['-1'][1], confuse['0'][1], confuse['1'][1], confuse['2'][1]))
    print('0\t{}\t{}\t{}\t{}\t{}'.format(confuse['-2'][2], confuse['-1'][2], confuse['0'][2], confuse['1'][2], confuse['2'][2]))
    print('1\t{}\t{}\t{}\t{}\t{}'.format(confuse['-2'][3], confuse['-1'][3], confuse['0'][3], confuse['1'][3], confuse['2'][3]))
    print('2\t{}\t{}\t{}\t{}\t{}'.format(confuse['-2'][4], confuse['-1'][4], confuse['0'][4], confuse['1'][4], confuse['2'][4]))

def draw(trueA, preA, trueS, preS, trueE, preE):

    confuse_a = {'-2' : [0, 0, 0, 0, 0], '-1' : [0, 0, 0, 0, 0], '0' : [0, 0, 0, 0, 0], '1' : [0, 0, 0, 0, 0], '2' : [0, 0, 0, 0, 0]}
    for p, p_ in zip(trueA, preA):
        confuse_a[p][bin_.index(p_)] += 1

    confuse_e = {'-2' : [0, 0, 0, 0, 0], '-1' : [0, 0, 0, 0, 0], '0' : [0, 0, 0, 0, 0], '1' : [0, 0, 0, 0, 0], '2' : [0, 0, 0, 0, 0]}
    for p, p_ in  zip(trueE, preE):
        confuse_e[p][bin_.index(p_)] += 1

    confuse_s = {'-2' : [0, 0, 0, 0, 0], '-1' : [0, 0, 0, 0, 0], '0' : [0, 0, 0, 0, 0], '1' : [0, 0, 0, 0, 0], '2' : [0, 0, 0, 0, 0]}
    for p, p_ in  zip(trueS, preS):
        confuse_s[p][bin_.index(p_)] += 1
    
    show(confuse_a)
    show(confuse_s)
    show(confuse_e)

# draw(A, dq_preA, S, dq_preS, E, dq_preE)
# draw(A, seq_preA, S, seq_preS, E, seq_preE)
draw(A, cls_preA, S, cls_preS, E, cls_preE)
