import logging
import os
import subprocess
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def run_cmd(command: str):
    logger.info(command)
    subprocess.check_call(command, shell=True)


def wait_for_file(file: str, minute: int = 1):
    if not os.path.exists(file):
        logger.info(f'Could not find file {file}. Waiting...')
        minute_cnt = 0
        while not os.path.exists(file):
            print(f'The {minute_cnt}th minute...')
            time.sleep(60)
            minute_cnt += 1
        print(f'Have found file. Wait for writing for {minute} extra minutes...')
        time.sleep(60 * minute)
        logger.info(f'Find file {file} after waiting for {minute_cnt + minute} minutes')


# model
# bert_base_model = "../BERT/bert-base-uncased.tar.gz"
# bert_base_vocab = "../BERT/bert-base-uncased-vocab.txt"
# bert_large_model = "../BERT/bert-large-uncased.tar.gz"
# bert_large_vocab = "../BERT/bert-large-uncased-vocab.txt"

# train_file = '/home/jiaofangkai/RACE/RACE/train-combine.json'
# dev_file = '/home/jiaofangkai/RACE/RACE/dev-combine.json'
# test_file = '/home/jiaofangkai/RACE/RACE/test-combine.json'


#
# task_name = 'race'
# reader_name = 'multiple-race'
# bert_name = 'topk-hie-race'
#
# learning_rate = 4e-5
# num_train_epochs = 3
#
# metric = 'accuracy'
#
# output_dir = f'experiments/race/topk-evidence/combine/rule/idf_v1.0'
# rule_sentence_id_file = 'experiments/race/topk-evidence/combine/rule/sentence_id_rule1.0.json'

model_name = 'hfl_chinese_wwm_ext'

bert_base_model = "/users8/hzyang/proj/c3-master/pretrained/"+model_name
bert_base_vocab = "/users8/hzyang/proj/c3-master/pretrained/"+model_name+"/vocab.txt"

train_file = '/users8/hzyang/proj/c3-master/data/race/race_c3-combine-train.json'
dev_file = '/users8/hzyang/proj/c3-master/data/race/race_c3-combine-dev.json'
test_file = '/users8/hzyang/proj/c3-master/data/race/race_c3-combine-test.json'

learning_rate = 2e-5
num_train_epochs = 10

task_name = 'race_c3'
reader_name = 'multiple-race'
bert_name = 'topk-hie-race'

metric = 'accuracy'

output_dir = f'/users8/hzyang/proj/c3-master/Self-Training-MRC-master/experiments/'+task_name+'_'+str(num_train_epochs)+'/topk-evidence/combine/rule/idf_v1.0_'
rule_sentence_id_file = '/users8/hzyang/proj/c3-master/data/race/race_c3-sentence_id_rule.json'


cmd = f'CUDA_VISIBLE_DEVICES=5,6 /users8/hzyang/miniconda3/envs/python36/bin/python /users8/hzyang/proj/c3-master/Self-Training-MRC-master/main_multi_choice_top_k_evidence.py ' \
    f'--bert_model {model_name}  --vocab_file {bert_base_vocab} --model_file {bert_base_model} --output_dir {output_dir} --predict_dir {output_dir} ' \
    f'--train_file {train_file} --predict_file {dev_file} --test_file {test_file} ' \
    f'--max_seq_length 512 --train_batch_size 4 --predict_batch_size 2 ' \
    f'--learning_rate {learning_rate} --num_train_epochs {num_train_epochs} ' \
    f' --gradient_accumulation_steps 2 --per_eval_step 1000 ' \
    f'--bert_name {bert_name} --task_name {task_name} --reader_name {reader_name} ' \
    f'--metric {metric} ' \
    f'--do_label --sentence_id_file {rule_sentence_id_file} --evidence_lambda 0.8 '

cmd += ' --do_train --do_predict '

run_cmd(cmd)


'''
python /users8/hzyang/proj/c3-master/Self-Training-MRC-master/scripts/race-f-multiple-evidence/topk_evidence/combine/rule/rule_idf_1.0.py


'''