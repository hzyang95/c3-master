import logging
import os
import subprocess
import time

import logging
t = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    filename=t+'.log',
                    level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
# 设置格式
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
# 告诉handler使用这个格式
console.setFormatter(formatter)
# add the handler to the root logger
# 为root logger添加handler
logging.getLogger('').addHandler(console)

logging.info('begin')




def run_cmd(command: str):
    logging.info(command)
    subprocess.check_call(command, shell=True)


def wait_for_file(file: str, minute: int = 1):
    if not os.path.exists(file):
        logging.info(f'Could not find file {file}. Waiting...')
        minute_cnt = 0
        while not os.path.exists(file):
            print(f'The {minute_cnt}th minute...')
            time.sleep(60)
            minute_cnt += 1
        print(f'Have found file. Wait for writing for {minute} extra minutes...')
        time.sleep(60 * minute)
        logging.info(f'Find file {file} after waiting for {minute_cnt + minute} minutes')


# model
# bert_base_model = "~/bert-base-uncased.tar.gz"
# bert_base_vocab = "~/bert-base-uncased-vocab.txt"
# bert_large_model = "../BERT/bert-large-uncased.tar.gz"
# bert_large_vocab = "../BERT/bert-large-uncased-vocab.txt"

bert_base_model = "/users8/hzyang/proj/c3-master/pretrained/bert-base-uncased/bert-base-uncased.tar.gz"
bert_base_vocab = "/users8/hzyang/proj/c3-master/pretrained/bert-base-uncased//bert-base-uncased-vocab.txt"

# train_file = 'data/RACE/train-high-ini.json'
# dev_file = 'data/RACE/dev-high.json'
# test_file = 'data/RACE/test-high.json'

train_file = '/users8/hzyang/proj/c3-master/Self-Training-MRC-master/data/RACE/train-high-ini.json'
dev_file = '/users8/hzyang/proj/c3-master/Self-Training-MRC-master/data/RACE/dev-high.json'
test_file = '/users8/hzyang/proj/c3-master/Self-Training-MRC-master/data/RACE/test-high.json'



task_name = 'race'
reader_name = 'multiple-race'
bert_name = 'pool-race'
learning_rate = 4e-5
num_train_epochs = 3

metric = 'accuracy'

output_dir = f'/users8/hzyang/proj/c3-master/Self-Training-MRC-master/experiments/race/topk-evidence/high/pool/v1.0'

cmd = f'/users8/hzyang/miniconda3/envs/python36/bin/python /users8/hzyang/proj/c3-master/Self-Training-MRC-master/main_multi_choice_top_k_evidence.py --bert_model bert-base-uncased ' \
    f'--vocab_file {bert_base_vocab} --model_file {bert_base_model} --output_dir {output_dir} --predict_dir {output_dir} ' \
    f'--train_file {train_file} --predict_file {dev_file} --test_file {test_file} ' \
    f'--max_seq_length 380 --train_batch_size 20 --predict_batch_size 5 ' \
    f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs} ' \
    f' --gradient_accumulation_steps 5 --per_eval_step 3000 ' \
    f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
    f'--metric {metric} '

cmd += '--do_predict '

run_cmd(cmd)