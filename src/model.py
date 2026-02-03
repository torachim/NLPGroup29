import torch.nn as nn
# imports neural network components
from transformers import XLNetModel

# wrapper class for xlnet classification model
class SingleHeadXLNet(nn.Module):
    # initializes model components and dimensions
    def __init__(self, model_name='xlnet-base-cased', num_labels=3):
        super(SingleHeadXLNet, self).__init__()
        print(f"loading {model_name} with {num_labels} labels...")
        # loads pretrained xlnet base model
        self.xlnet = XLNetModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.1)
        # defines final classification layer
        self.classifier = nn.Linear(768, num_labels)

    # performs forward pass of the network
    def forward(self, input_ids, attention_mask):
        # gets hidden states from xlnet
        outputs = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # creates single vector representation via attention masking
        # expands attention mask dimensions for broadcasting
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # computes weighted sum of embeddings
        sum_embeddings = (last_hidden_state * input_mask_expanded).sum(1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = sum_mask.clamp(min=1e-9)
        # calculates mean by dividing sum by mask
        pooled_output = sum_embeddings / sum_mask
        
        # passes through dropout and classifier
        output = self.drop(pooled_output)
        logits = self.classifier(output)
        
        return logits