import torch
import torch.nn as nn
from transformers import XLNetModel

class DualHeadXLNet(nn.Module):
    def __init__(self, model_name='xlnet-base-cased', num_evasion_labels=9):
        super(DualHeadXLNet, self).__init__()
        print(f"loading {model_name} with {num_evasion_labels} evasion labels...")
        self.xlnet = XLNetModel.from_pretrained(model_name)
        
        self.drop = nn.Dropout(p=0.1)
        self.clarity_head = nn.Linear(768, 3)
        # Dynamic size for evasion head
        self.evasion_head = nn.Linear(768, num_evasion_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        output = self.drop(pooled_output)
        return self.clarity_head(output), self.evasion_head(output)