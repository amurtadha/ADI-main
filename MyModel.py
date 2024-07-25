from transformers import BertModel, BertTokenizer


import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function
class Vanilla(nn.Module):
    def __init__(self, args, hidden_size=256):
        super(Vanilla, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrained_bert_name)
        self.encoder = AutoModel.from_pretrained(args.pretrained_bert_name, config=config)
        self.encoder.to('cuda')

        layers = [nn.Linear(config.hidden_size, args.lebel_dim)]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):

        input_ids,token_type_ids, attention_mask = inputs[:3]
        outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # pooled_output = outputs['last_hidden_state'][:, 0, :]
        pooled_output = outputs[-1]
        pooled_output = self.dropout(pooled_output)


        return  self.classifier(pooled_output)

class GradientReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None



def grad_reverse(x, alpha=1.0):
    return GradientReverseLayer.apply(x, alpha)
class CosFace(nn.Module):
    def __init__(self, in_features, out_features, s=1.0, m=0.01):
    # def __init__(self, in_features, out_features, s=30.0, m=0.4):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output



class DID(nn.Module):
    def __init__(self, opt, alpha=1.0):
        super(DID, self).__init__()
        self.bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.grl = GradientReverseLayer()
        self.alpha = alpha
        self.cosface = CosFace(self.bert.config.hidden_size, 2)
        self.cosface_DID = CosFace(self.bert.config.hidden_size, opt.lebel_dim)

    def forward(self, inputs, labels, reverse=False, alpha=1.0):

        input_ids, token_type_ids, attention_mask = inputs[:3]

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        if reverse:
            reversed_features = grad_reverse(pooled_output, alpha)
            logits = self.cosface(reversed_features, labels)
        else:
            logits = self.cosface_DID(pooled_output, labels)
        return logits
