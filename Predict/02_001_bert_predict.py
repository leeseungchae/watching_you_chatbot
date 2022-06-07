

from google.colab import drive
drive.mount('/content/drive')

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split 

device = torch.device("cuda:0")
bertmodel, vocab = get_pytorch_kobert_model()

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

## Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 30
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

new_test = nlp.data.TSVDataset("/content/drive/MyDrive/Watching_You_PJ/project_ChatBot/02_preprocessing/BERT/train/02_001_bert_train.tsv" , field_indices=[0,2], num_discard_samples=1)
test_set = BERTDataset(new_test , 0, 1, tok, max_len, True, False)
test_input = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=4)

df = pd.read_csv('/content/drive/MyDrive/Watching_You_PJ/project_ChatBot/02_preprocessing/02_001_bert.csv')
df

category_list = list(df['MAIN'].unique())
category_list
cat_len = len(category_list)
cat_len

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=cat_len,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)

model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
model.load_state_dict(torch.load('/content/drive/MyDrive/Watching_You_PJ/project_ChatBot/03_DL_Model/02_001_bert_10.pth'))

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

def predict(predict_sentence):

    if len(predict_sentence) <5 :
        print("너무 짧아요")
        return False
    else:




      data = [predict_sentence, '0']
      dataset_another = [data]

      another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
      test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=4)
      
      model.eval()

      for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
          token_ids = token_ids.long().to(device)
          segment_ids = segment_ids.long().to(device)

          valid_length= valid_length
          label = label.long().to(device)

          out = model(token_ids, valid_length, segment_ids)
          prediction = out.cpu().detach().numpy().argmax()

          test_eval=[]
          test_eval_2=[]
          for i in out:
              logits=i
              logits = logits.detach().cpu().numpy()
          for k , c in enumerate(category_list):
          # for k,(c,p) in enumerate(zip(category_list,prediction)):
              
              if np.argmax(logits) == k:
                  value = logits[k]
                  value  = round(value*10)
                  if value <50 :
                      print("죄송합니다 다시 말씀해주세요")
                      print(value)
                      # print()
                      test_eval.append(c)
                      value = logits[k]
                      test_eval_2.append(value)
                      print(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")

                      return False
                  else :
                      test_eval.append(c)
                      value = logits[k]
                      test_eval_2.append(value)

              # elif np.argmax(logits) == 1:
              #     test_eval.append("텀블라사용")

          print(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")
          # print(value)
          print(test_eval_2[0])



end = 1
while end == 1 :
    sentence = input("하고싶은 말을 입력해주세요(5자 이상) : ")
    if sentence == 0 :
        
        break
    predict(sentence)


    print("\n")


