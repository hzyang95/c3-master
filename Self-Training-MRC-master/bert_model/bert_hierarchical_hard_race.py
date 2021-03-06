import torch
# from allennlp.nn.util import masked_softmax, masked_log_softmax
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from bert_model import layers, rep_layers
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class BertQAYesnoHierarchicalHardRACE(BertPreTrainedModel):
    """
    Hard:
    Hard attention, using gumbel softmax of reinforcement learning.
    """

    def __init__(self, config, evidence_lambda=0.8, num_choices=4, use_gumbel=True, freeze_bert=False):
        super(BertQAYesnoHierarchicalHardRACE, self).__init__(config)
        logger.info(f'The model {self.__class__.__name__} is loading...')
        logger.info(f'The coefficient of evidence loss is {evidence_lambda}')
        logger.info(f'Currently the number of choices is {num_choices}')
        logger.info(f'Use gumbel: {use_gumbel}')
        logger.info(f'If freeze BERT\'s parameters: {freeze_bert} ')

        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(config.hidden_dropout_prob)
        rep_layers.set_seq_dropout(True)
        rep_layers.set_my_dropout_prob(config.hidden_dropout_prob)

        self.bert = BertModel(config)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        # self.doc_sen_self_attn = layers.LinearSelfAttnAllennlp(config.hidden_size)
        # self.que_self_attn = layers.LinearSelfAttn(config.hidden_size)
        self.doc_sen_self_attn = rep_layers.LinearSelfAttention(config.hidden_size)
        self.que_self_attn = rep_layers.LinearSelfAttention(config.hidden_size)

        self.word_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)
        self.vector_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)

        self.classifier = nn.Linear(config.hidden_size * 2, 1)
        self.evidence_lam = evidence_lambda
        self.use_gumbel = use_gumbel
        self.num_choices = num_choices

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                sentence_span_list=None, sentence_ids=None, max_sentences: int = 0):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        sequence_output, _ = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)

        # mask: 1 for masked value and 0 for true value
        # doc, que, doc_mask, que_mask = layers.split_doc_que(sequence_output, token_type_ids, attention_mask)
        doc_sen, que, doc_sen_mask, que_mask, sentence_mask = \
            rep_layers.split_doc_sen_que(sequence_output, flat_token_type_ids, flat_attention_mask, sentence_span_list,
                                         max_sentences=max_sentences)

        batch, max_sen, doc_len = doc_sen_mask.size()
        # que_len = que_mask.size(1)

        # que_vec = layers.weighted_avg(que, self.que_self_attn(que, que_mask)).view(batch, 1, -1)
        que_vec = self.que_self_attn(que, que_mask).view(batch, 1, -1)

        doc = doc_sen.reshape(batch, max_sen * doc_len, -1)
        word_sim = self.word_similarity(que_vec, doc).view(batch * max_sen, doc_len)
        doc = doc_sen.reshape(batch * max_sen, doc_len, -1)
        doc_mask = doc_sen_mask.reshape(batch * max_sen, doc_len)
        word_hidden = rep_layers.masked_softmax(word_sim, doc_mask, dim=1).unsqueeze(1).bmm(doc)

        word_hidden = word_hidden.view(batch, max_sen, -1)

        doc_vecs = self.doc_sen_self_attn(doc, doc_mask).view(batch, max_sen, -1)

        sentence_sim = self.vector_similarity(que_vec, doc_vecs)
        sentence_hidden = self.hard_sample(sentence_sim, use_gumbel=self.use_gumbel, dim=-1,
                                           hard=True, mask=sentence_mask).bmm(word_hidden).squeeze(1)

        choice_logits = self.classifier(torch.cat([sentence_hidden, que_vec.squeeze(1)], dim=1)).reshape(-1, self.num_choices)

        sentence_scores = rep_layers.masked_softmax(sentence_sim, sentence_mask, dim=-1).squeeze_(1)
        output_dict = {
            'choice_logits': choice_logits.float(),
            'sentence_logits': sentence_scores.reshape(choice_logits.size(0), self.num_choices, max_sen).detach().cpu().float(),
        }
        loss = 0
        if labels is not None:
            choice_loss = F.cross_entropy(choice_logits, labels)
            loss += choice_loss
        if sentence_ids is not None:
            log_sentence_sim = rep_layers.masked_log_softmax(sentence_sim.squeeze(1), sentence_mask, dim=-1)
            sentence_loss = F.nll_loss(log_sentence_sim, sentence_ids.view(batch), reduction='sum', ignore_index=-1)
            loss += self.evidence_lam * sentence_loss / choice_logits.size(0)
        output_dict['loss'] = loss
        return output_dict

    def hard_sample(self, logits, use_gumbel, dim=-1, hard=True, mask=None):
        if use_gumbel:
            if self.training:
                probs = rep_layers.gumbel_softmax(logits, mask=mask, hard=hard, dim=dim)
                return probs
            else:
                probs = rep_layers.masked_softmax(logits, mask, dim=dim)
                index = probs.max(dim, keepdim=True)[1]
                y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
                return y_hard
        else:
            pass


class BertQAYesnoHierarchicalReinforceRACE(BertPreTrainedModel):
    """
    Hard attention using reinforce learning
    """

    def __init__(self, config, evidence_lambda=0.8, num_choices=4, sample_steps: int = 5, reward_func: int = 0, freeze_bert=False):
        super(BertQAYesnoHierarchicalReinforceRACE, self).__init__(config)
        logger.info(f'The model {self.__class__.__name__} is loading...')
        logger.info(f'The coefficient of evidence loss is {evidence_lambda}')
        logger.info(f'Currently the number of choices is {num_choices}')
        logger.info(f'Sample steps: {sample_steps}')
        logger.info(f'Reward function: {reward_func}')
        logger.info(f'If freeze BERT\'s parameters: {freeze_bert} ')

        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(config.hidden_dropout_prob)
        rep_layers.set_seq_dropout(True)
        rep_layers.set_my_dropout_prob(config.hidden_dropout_prob)

        self.bert = BertModel(config)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.doc_sen_self_attn = rep_layers.LinearSelfAttention(config.hidden_size)
        self.que_self_attn = rep_layers.LinearSelfAttention(config.hidden_size)

        self.word_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)
        self.vector_similarity = layers.AttentionScore(config.hidden_size, 250, do_similarity=False)

        # self.yesno_predictor = nn.Linear(config.hidden_size * 2, 3)
        self.classifier = nn.Linear(config.hidden_size * 2, 1)
        self.evidence_lam = evidence_lambda
        self.sample_steps = sample_steps
        self.reward_func = [self.reinforce_step, self.reinforce_step_1][reward_func]
        self.num_choices = num_choices

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                sentence_span_list=None, sentence_ids=None, max_sentences: int = 0):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        sequence_output, _ = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)

        # mask: 1 for masked value and 0 for true value
        # doc, que, doc_mask, que_mask = layers.split_doc_que(sequence_output, token_type_ids, attention_mask)
        doc_sen, que, doc_sen_mask, que_mask, sentence_mask = \
            rep_layers.split_doc_sen_que(sequence_output, flat_token_type_ids, flat_attention_mask, sentence_span_list,
                                         max_sentences=max_sentences)

        batch, max_sen, doc_len = doc_sen_mask.size()

        que_vec = self.que_self_attn(que, que_mask).view(batch, 1, -1)

        doc = doc_sen.reshape(batch, max_sen * doc_len, -1)
        word_sim = self.word_similarity(que_vec, doc).view(batch * max_sen, doc_len)
        doc = doc_sen.reshape(batch * max_sen, doc_len, -1)
        doc_mask = doc_sen_mask.reshape(batch * max_sen, doc_len)
        word_hidden = rep_layers.masked_softmax(word_sim, doc_mask, dim=1).unsqueeze(1).bmm(doc)

        word_hidden = word_hidden.view(batch, max_sen, -1)

        doc_vecs = self.doc_sen_self_attn(doc, doc_mask).view(batch, max_sen, -1)

        sentence_sim = self.vector_similarity(que_vec, doc_vecs)
        if self.training:
            _sample_prob, _sample_log_prob = self.sample_one_hot(sentence_sim, sentence_mask)
            loss_and_reward, _ = self.reward_func(word_hidden, que_vec, labels, _sample_prob, _sample_log_prob)
            output_dict = {'loss': loss_and_reward}
        else:
            _prob, _ = self.sample_one_hot(sentence_sim, sentence_mask)
            loss, _choice_logits = self.simple_step(word_hidden, que_vec, labels, _prob)
            sentence_scores = rep_layers.masked_softmax(sentence_sim, sentence_mask, dim=-1).squeeze_(1)
            output_dict = {
                'sentence_logits': sentence_scores.float(),
                'loss': loss,
                'choice_logits': _choice_logits.float()
            }

        return output_dict

    def sample_one_hot(self, _similarity, _mask):
        _probability = rep_layers.masked_softmax(_similarity, _mask)
        dtype = _probability.dtype
        _probability = _probability.float()
        # _log_probability = masked_log_softmax(_similarity, _mask)
        if self.training:
            _distribution = Categorical(_probability)
            _sample_index = _distribution.sample((self.sample_steps,))
            logger.debug(str(_sample_index.size()))
            new_shape = (self.sample_steps,) + _similarity.size()
            logger.debug(str(new_shape))
            _sample_one_hot = F.one_hot(_sample_index, num_classes=_similarity.size(-1))
            # _sample_one_hot = _similarity.new_zeros(new_shape).scatter(-1, _sample_index.unsqueeze(-1), 1.0)
            logger.debug(str(_sample_one_hot.size()))
            _log_prob = _distribution.log_prob(_sample_index)  # sample_steps, batch, 1
            assert _log_prob.size() == new_shape[:-1], (_log_prob.size(), new_shape)
            _sample_one_hot = _sample_one_hot.transpose(0, 1)  # batch, sample_steps, 1, max_sen
            _log_prob = _log_prob.transpose(0, 1)  # batch, sample_steps, 1
            return _sample_one_hot.to(dtype=dtype), _log_prob.to(dtype=dtype)
        else:
            _max_index = _probability.float().max(dim=-1, keepdim=True)[1]
            _one_hot = torch.zeros_like(_similarity).scatter_(-1, _max_index, 1.0)
            # _log_prob = _log_probability.gather(-1, _max_index)
            return _one_hot, None

    def reinforce_step(self, hidden, q_vec, label, prob, log_prob):
        batch, max_sen, hidden_dim = hidden.size()
        assert q_vec.size() == (batch, 1, hidden_dim)
        assert prob.size() == (batch, self.sample_steps, 1, max_sen)
        assert log_prob.size() == (batch, self.sample_steps, 1)
        expanded_hidden = hidden.unsqueeze(1).expand(-1, self.sample_steps, -1, -1)
        h = prob.matmul(expanded_hidden).squeeze(2)  # batch, sample_steps, hidden_dim
        q = q_vec.expand(-1, self.sample_steps, -1)
        # _logits = self.classifier(torch.cat([h, q], dim=2)).view(-1, self.num_choices)  # batch, sample_steps, 3
        # Note the rank of dimension here
        _logits = self.classifier(torch.cat([h, q], dim=2)).view(label.size(0), self.num_choices, self.sample_steps)\
            .transpose(1, 2).reshape(-1, self.num_choices)
        expanded_label = label.unsqueeze(1).expand(-1, self.sample_steps).reshape(-1)
        _loss = F.cross_entropy(_logits, expanded_label)
        corrects = (_logits.max(dim=-1)[1] == expanded_label).to(hidden.dtype)
        log_prob = log_prob.reshape(label.size(0), self.num_choices, self.sample_steps).transpose(1, 2).mean(dim=-1)
        reward1 = (log_prob.reshape(-1) * corrects).sum() / (self.sample_steps * label.size(0))
        return _loss - reward1, _logits

    def reinforce_step_1(self, hidden, q_vec, label, prob, log_prob):
        batch, max_sen, hidden_dim = hidden.size()
        assert q_vec.size() == (batch, 1, hidden_dim)
        assert prob.size() == (batch, self.sample_steps, 1, max_sen)
        assert log_prob.size() == (batch, self.sample_steps, 1)
        expanded_hidden = hidden.unsqueeze(1).expand(-1, self.sample_steps, -1, -1)
        h = prob.matmul(expanded_hidden).squeeze(2)  # batch, sample_steps, hidden_dim
        q = q_vec.expand(-1, self.sample_steps, -1)
        # _logits = self.classifier(torch.cat([h, q], dim=2)).view(-1, self.num_choices)  # batch * sample_steps, 3
        _logits = self.classifier(torch.cat([h, q], dim=2)).view(label.size(0), self.num_choices, self.sample_steps)\
            .transpose(1, 2).reshape(-1, self.num_choices)
        expanded_label = label.unsqueeze(1).expand(-1, self.sample_steps).reshape(-1)  # batch * sample_steps

        _loss = F.cross_entropy(_logits, expanded_label)

        _final_log_prob = F.log_softmax(_logits, dim=-1)
        # ignore_mask = (expanded_label == -1)
        # expanded_label = expanded_label.masked_fill(ignore_mask, 0)
        selected_log_prob = _final_log_prob.gather(1, expanded_label.unsqueeze(1)).squeeze(-1)  # batch * sample_steps
        assert selected_log_prob.size() == (label.size(0) * self.sample_steps,), selected_log_prob.size()
        log_prob = log_prob.reshape(label.size(0), self.num_choices, self.sample_steps).transpose(1, 2).mean(dim=-1)
        # reward2 = - (log_prob.reshape(-1) * (selected_log_prob * (1 - ignore_mask).to(log_prob.dtype))).sum() / (
        #         self.sample_steps * batch)
        reward2 = - (log_prob.reshape(-1) * selected_log_prob).sum() / (self.sample_steps * label.size(0))

        return _loss - reward2, _logits

    def simple_step(self, hidden, q_vec, label, prob):
        batch, max_sen, hidden_dim = hidden.size()
        assert q_vec.size() == (batch, 1, hidden_dim)
        assert prob.size() == (batch, 1, max_sen)
        h = prob.bmm(hidden)
        _logits = self.classifier(torch.cat([h, q_vec], dim=2)).view(-1, self.num_choices)
        if label is not None:
            _loss = F.cross_entropy(_logits, label)
        else:
            _loss = _logits.new_zeros(1)
        return _loss, _logits
