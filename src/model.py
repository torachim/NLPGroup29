import torch
import torch.nn as nn
from transformers import XLNetModel

class DualHeadXLNet(nn.Module):
    def __init__(self, model_name='xlnet-base-cased'):
        super(DualHeadXLNet, self).__init__()
        
        # load base xlnet
        print(f"loading pre-trained {model_name}...")
        self.xlnet = XLNetModel.from_pretrained(model_name)
        
        # hidden size for xlnet-base is 768
        self.drop = nn.Dropout(p=0.3)
        
        # head 1: clarity (3 classes)
        self.clarity_head = nn.Linear(768, 3)
        
        # head 2: evasion (9 classes)
        self.evasion_head = nn.Linear(768, 9)

    def forward(self, input_ids, attention_mask):
        # forward pass through xlnet
        outputs = self.xlnet(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # get last hidden state
        # xlnet uses the last token for classification summary usually
        last_hidden_state = outputs.last_hidden_state
        
        # simple pooling: mean of all tokens
        # or just take the last token (often used for xlnet)
        # let's use mean pooling for robustness
        pooled_output = last_hidden_state.mean(dim=1)
        
        output = self.drop(pooled_output)
        
        # pass through both heads
        clarity_logits = self.clarity_head(output)
        evasion_logits = self.evasion_head(output)
        
        return clarity_logits, evasion_logits