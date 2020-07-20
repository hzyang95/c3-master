#!/usr/bin/env python
# coding: utf-8

import json

import jieba
import nltk
# import jsonlines
from tqdm import tqdm
import math

import argparse
import itertools

from pyltp import SentenceSplitter

# sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentence_tokenizer = SentenceSplitter


class Jaccard(object):
    """docstring for Jaccard"""

    def __init__(self):
        super(Jaccard, self).__init__()
        pass

    def get_sim(self, str1, str2):
        a = set(jieba.cut(str1.lower()))
        b = set(jieba.cut(str2.lower()))
        c = a.intersection(b)
        return float(len(c)) / len(b) - len(a) * 1e-9


class IDF(object):
    """docstring for IDF"""

    def __init__(self):
        super(IDF, self).__init__()
        pass

    def get_sim(self, str1, str2, idf):
        a = set(jieba.cut(str1.lower()))
        b = set(jieba.cut(str2.lower()))
        c = a.intersection(b)

        b = sum([idf[w] if w in idf else max(idf.values()) for w in b])
        c = sum([idf[w] for w in c])
        return (c + 1e-15) / (b + 1e-15) - len(a) * 1e-9

    def get_idf(self, sentences):
        idf = {}
        for sentence in sentences:
            sentence = set(jieba.cut(sentence.lower()))
            for word in sentence:
                if word not in idf:
                    idf[word] = 0
                idf[word] += 1
        num_sentence = len(sentences)
        for word, count in idf.items():
            idf[word] = math.log((num_sentence + 1) / float(count + 1))
        return idf


class ILP(object):
    """docstring for ILP"""

    def __init__(self):
        super(ILP, self).__init__()
        pass

    def get_sim(self, sentences, word_weights, max_k=1):
        sentences = [set(jieba.cut(sentence.lower())) for sentence in sentences]
        max_set, max_value = [-1], -1
        for sentence_ids in itertools.combinations(range(len(sentences)), max_k):
            words = sentences[sentence_ids[0]]
            for sentence_id in sentence_ids[1:]:
                words = words + sentences[sentence_id]
            words = set(words)
            value = sum([word_weights[word] if word in word_weights else 0. for word in words]) / sum(word_weights.values())
            if value > max_value:
                max_set = sentence_ids
                max_value = value
        return max_set, max_value

    def get_word_weights0(self, query):
        query = set(query.lower().split())
        word_weights = {word: 1. for word in query}
        return word_weights

    def get_word_weights1(self, query, historys):
        query = set(jieba.cut(query.lower()))
        word_weights = {word: 1. for word in query}
        historys = [set(jieba.cut(history.lower())) for history in historys]
        for history in historys:
            for word in history:
                if word in word_weights:
                    continue
                word_weights[word] = 0.1
        return word_weights


class DatasetBase(object):
    """docstring for DatasetBase"""

    def __init__(self, mode: str = 'IDF'):
        super(DatasetBase, self).__init__()
        '''
        mode: `Jaccard/IDF/ILP`
        '''
        self.evidence = []
        self.mode = mode
        if self.mode == 'Jaccard':
            self.rule = Jaccard()
        elif self.mode == 'IDF':
            self.rule = IDF()
        elif self.mode == 'ILP':
            self.rule = ILP()
        else:
            raise ValueError('`mode`[input:%s] should be one of `Jaccard/IDF/ILP`' % (self.mode))

    def sort(self, top_k):
        threshold = sorted(self.evidence, key=lambda x: x[1], reverse=True)[min(top_k - 1,len(self.evidence)-1)][1]
        for _evidence in self.evidence:
            if _evidence[1] < threshold:
                _evidence[0] = -1
        return self.evidence

    def get_sim(self, *args):
        if self.mode == 'Jaccard':
            max_ids, max_value = self.jaccard(*args)
        elif self.mode == 'IDF':
            max_ids, max_value = self.idf(*args)
        elif self.mode == 'ILP':
            max_ids, max_value = self.ilp(*args)
        self.evidence.append([max_ids, max_value])

    def jaccard(self, sentences, query, top_k=1):
        values = []
        for sentence in sentences:
            values.append(self.rule.get_sim(sentence, query))
        values = sorted(enumerate(values), key=lambda x: x[1], reverse=True)[:top_k]
        max_ids = [x[0] for x in values]
        max_value = sum([x[1] for x in values])
        return max_ids, max_value

    def idf(self, sentences, query, top_k=1):
        idf = self.rule.get_idf(sentences)
        values = []
        for sentence in sentences:
            values.append(self.rule.get_sim(sentence, query, idf))
        values = sorted(enumerate(values), key=lambda x: x[1], reverse=True)[:top_k]
        max_ids = [x[0] for x in values]
        max_value = sum([x[1] for x in values])
        return max_ids, max_value

    def ilp(self, sentences, query, top_k=1):
        word_weights = self.rule.get_word_weights0(query)
        max_set, max_value = self.rule.get_sim(sentences, word_weights, top_k)
        return max_set, max_value

    def save(self, data, output_file):
        with open(output_file, 'w') as w:
            json_data = json.dumps(data, sort_keys=False, indent=4, separators=(',', ': '))
            w.write(json_data)



class RACE(DatasetBase):
    """docstring for CoQA"""

    def __init__(self, mode):
        super(RACE, self).__init__(mode)
        pass

    def process_file(self, input_file, num_evidences=2, top_k: int = 1000):
        with open(input_file, 'r') as f:
            data = json.load(f)

        # examples = []
        for instance in tqdm(data[:]):
            passage = instance['article']
            article_id = instance['id']

            article_sentences = sentence_tokenizer.split(passage)

            questions = instance['questions']
            answers = list(map(lambda x: {'A': 0, 'B': 1, 'C': 2, 'D': 3}[x], instance['answers']))
            options = instance['options']

            for q_id, (question, answer, option_list) in enumerate(zip(questions, answers, options)):
                # qas_id = f"{article_id}--{q_id}"

                for option in option_list:
                    self.get_sim(article_sentences, question + ' ' + option, num_evidences)
        # print(self.evidence)
        print(len(self.evidence))
        self.sort(top_k)
        output = {}
        for instance in tqdm(data[:]):
            article_id = instance['id']
            questions = instance['questions']
            options = instance['options']

            for q_id, (_, option_list) in enumerate(zip(questions, options)):
                qas_id = f'{article_id}--{q_id}'

                output[qas_id] = {'sentence_ids': []}
                for op_index, op in enumerate(option_list):
                    if self.evidence[0][0] == -1:
                        output[qas_id]['sentence_ids'].append([])
                    else:
                        output[qas_id]['sentence_ids'].append(self.evidence[0][0])
                    self.evidence = self.evidence[1:]

        return output

if __name__ == '__main__':

    # task_name = 'RACE'
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name',
                        default='RACE',
                        type=str)
    parser.add_argument('--input_file',
                        default='/users8/hzyang/proj/c3-master/data/race/race_c3-combine-train.json',
                        type=str)
    parser.add_argument('--num_evidences', type=int, default=2)
    parser.add_argument('--top_k', type=int, default=30000)
    parser.add_argument('--output_file',
                        default='/users8/hzyang/proj/c3-master/data/race/race_c3-sentence_id_rule.json',
                        type=str)

    args = parser.parse_args()

    task_name = args.task_name

    if task_name == 'RACE':
        train_race = RACE(mode='IDF')
        train_data = train_race.process_file(args.input_file, top_k=args.top_k, num_evidences=args.num_evidences)
        with open(args.output_file, 'w') as f:
            json.dump(train_data, f, indent=2)
