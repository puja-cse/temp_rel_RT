import torch
from torch import nn
from transformers import AutoTokenizer, XLMRobertaModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def get_event_tensors (x, input_ids, event1_list, event2_list):
    tensor_verb1_cat_verb2 = torch.empty(0, 3*768)
    tensor_verb1_cat_verb2 = tensor_verb1_cat_verb2.to(device)
    sample_idx = 0 # i-th sample from the batch
    for ids in input_ids:
        tokens = tokenizer.convert_ids_to_tokens(ids)
        bert_token_idx = 0
        sent_token_idx = -1
        tens_e1 = torch.empty(0,768)
        tens_e2 = torch.empty(0,768)
        tens_e1=tens_e1.to(device)
        tens_e2 = tens_e2.to(device)
        for bert_token in tokens:
            if bert_token.startswith('▁'):
                sent_token_idx +=1
                if sent_token_idx == event1_list[sample_idx]+1:
                    tens_e1 = torch.mean(x[sample_idx, left_lim:bert_token_idx , :], dim=0)
                elif sent_token_idx == event2_list[sample_idx]+1: 
                    tens_e2 = torch.mean(x[sample_idx, left_lim:bert_token_idx , :], dim=0)
                left_lim = bert_token_idx
            elif  bert_token.startswith('<'):
                sent_token_idx+=1
                if sent_token_idx == event1_list[sample_idx]+1:
                    tens_e1 = torch.mean(x[sample_idx, left_lim:bert_token_idx , :], dim=0)
                elif sent_token_idx == event2_list[sample_idx]+1: 
                    tens_e2 = torch.mean(x[sample_idx, left_lim:bert_token_idx , :], dim=0)
            bert_token_idx+=1
        t = torch.cat((tens_e1, tens_e2, x[sample_idx, 0, :] ), dim =0)
        #t = torch.cat((x[sample_idx, 0, :] tens_e1, tens_e2), dim =0)
        tensor_verb1_cat_verb2 = torch.vstack((tensor_verb1_cat_verb2, t))
        sample_idx+=1
    return tensor_verb1_cat_verb2

class BertTempRel(nn.Module):
    
    def __init__(self, pretrained_model='xlm-roberta-base', vector_len=768*3, hidden_state=64, labels=4 ):
        """
        XLM-RoBerta model with additional layers for detection of temporal relation 
        """
        super(BertTempRel, self).__init__()
        self.bert_layer = XLMRobertaModel.from_pretrained(pretrained_model)
        self.linear1 = nn.Linear(vector_len, 256)
        self.lstm = nn.LSTM(input_size=vector_len, hidden_size=768, num_layers=3, batch_first=True)
        self.linear2 = nn.Linear(256, hidden_state) #use linear layer 2 with MLP
        self.linear3 = nn.Linear(768, hidden_state) #use linear layer 3 with lstm
        self.output_layer = nn.Linear(hidden_state, labels)


    def forward(self, input_ids, attention_mask , event1, event2):
        x= self.bert_layer(input_ids=input_ids, attention_mask=attention_mask)
        x= x[0] #last hidden state
        x = x.detach()
        x = get_event_tensors(x, input_ids, event1, event2)
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x=self.linear2(x)
        #x, _= self.lstm(x)
        #x=self.linear3(x)
        x = nn.functional.relu(x)
        x = self.output_layer(x)
        x = nn.functional.softmax(x,dim=1)
        return x