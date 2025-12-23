import torch.nn as nn
from transformers import XLNetModel

class SingleHeadXLNet(nn.Module):
    def __init__(self, model_name='xlnet-base-cased', num_labels=3):
        super(SingleHeadXLNet, self).__init__()
        print(f"loading {model_name} with {num_labels} labels...")
        self.xlnet = XLNetModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # Mean Pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = (last_hidden_state * input_mask_expanded).sum(1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = sum_mask.clamp(min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        output = self.drop(pooled_output)
        logits = self.classifier(output)
        
        return logits