import json
import random
from pyltp import SentenceSplitter

_len=0
_num=0
_aver = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

sl=0
sll=0
for sid in range(3):
    data = []
    for subtask in ["d", "m"]:
        with open("data/c3-" + subtask + "-" + ["train.json", "dev.json", "test.json"][sid], "r", encoding="utf8") as f:
            data += json.load(f)
    if sid == 0:
        random.shuffle(data)

    for i in range(len(data)):
        for j in range(len(data[i][1])):
            pas = '\n'.join(data[i][0]).lower()
            sll += 1
            sl += len(pas)
            sentence = SentenceSplitter.split(pas)
            le = len(sentence)

            if le>20:
                _aver[20]+=1
            else:
                _aver[le]+=1
            _len+=le
            _num+=1
            d = [pas, data[i][1][j]["question"].lower()]
            for k in range(len(data[i][1][j]["choice"])):
                d += [data[i][1][j]["choice"][k].lower()]
            for k in range(len(data[i][1][j]["choice"]), 4):
                d += ['']
            d += [data[i][1][j]["answer"].lower()]
            # print(d)
print(_num)
print(_len/_num)
print(_aver)
print(sll)
print(sl/sll)