from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers import DistilBertPreTrainedModel, DistilBertModel, BertConfig

from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertSelfAttention


class BSC(BertPreTrainedModel):

    def __init__(self, config, num_rel):
        super(BSC, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_rel)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, n_class=1):
        seq_length = input_ids.size(2)
        print(seq_length)
        _, pooled_output = self.bert(input_ids.view(-1, seq_length),
                                     token_type_ids.view(-1, seq_length),
                                     attention_mask.view(-1, seq_length))
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = logits.view(-1, n_class)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.view(-1)
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
