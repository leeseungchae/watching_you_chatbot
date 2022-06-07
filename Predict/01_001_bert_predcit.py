from bert_classs import *
import pandas as pd
from torch.utils.data import DataLoader
import torch.utils.data
import gluonnlp as nlp
import numpy as np
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model

def predict(predict_sentence):
    if len(predict_sentence) < 5:
        print("너무 짧아요")
        return False
    else:
        data = [predict_sentence, '0']
        dataset_another = [data]
        another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
        test_dataloader = DataLoader(another_test, batch_size=batch_size, num_workers=0)
        model.eval()

        for batch_id, (token_ids, valid_length, segment_ids,label) in enumerate(test_dataloader):

            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length

            label = label.long().to(device)

            out = model(token_ids, valid_length, segment_ids)
            prediction = out.cpu().detach().numpy().argmax()

            test_eval = []
            test_eval_2 = []
            for i in out:
                logits = i
                logits = logits.detach().cpu().numpy()
            for k, c in enumerate(category_list):
                # for k,(c,p) in enumerate(zip(category_list,prediction)):
                if np.argmax(logits) == k:
                    value = logits[k]
                    value = round(value * 10)
                    if value < 50:
                        print("죄송합니다 다시 말씀해주세요")
                        print(value)
                        # print()
                        test_eval.append(c)
                        value = logits[k]
                        test_eval_2.append(value)
                        print(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")

                        return False
                    else:
                        test_eval.append(c)
                        value = logits[k]
                        test_eval_2.append(value)
            print(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")
            print(test_eval_2[0])

device = torch.device('cpu')
bertmodel,vocab = get_pytorch_kobert_model()
model = BERTClassifier(bertmodel, dr_rate=0.5)
model.load_state_dict(torch.load('G:\\내 드라이브\\Watch_You\\01_001_bert_10.pth',
                                map_location=device))
# print(model)
max_len = 64
batch_size = 64

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
df = pd.read_csv('G:\\내 드라이브\\Watch_You\\01_001_bert.csv')
category_list = list(df['MAIN'].unique())

end = 1
while end == 1:
    sentence = input("하고싶은 말을 입력해주세요(5자 이상) : ")
    if sentence == 0:
        break
    predict(sentence)

    print("\n")