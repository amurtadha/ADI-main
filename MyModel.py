

import torch.nn as nn
from transformers import  AutoModel, AutoConfig


from torch.autograd import Function
def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)

class ADI_Classifier(nn.Module):
    def __init__(self, args, hidden_size=256):
        super(ADI_Classifier, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrained_bert_name)
        self.encoder = AutoModel.from_pretrained(args.pretrained_bert_name, config=config)
        self.encoder.to('cuda')

        layers = [nn.Linear(config.hidden_size, args.lebel_dim)]
        layers_rev = [nn.Linear(config.hidden_size, 2)]

        self.classifier = nn.Sequential(*layers)
        self.discriminator = nn.Sequential(*layers_rev)

    def forward(self, inputs, alpha):

        input_ids,token_type_ids, attention_mask = inputs[:3]
        outputs = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = outputs['last_hidden_state'][:, 0, :]
        reverse_feature =ReverseLayerF.apply(pooled_output, alpha)
        logits = self.classifier(pooled_output)
        logits_revers = self.discriminator(reverse_feature)
        return pooled_output, logits, logits_revers


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


