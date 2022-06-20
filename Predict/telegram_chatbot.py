import telegram
from telegram.ext import Updater, CommandHandler
from telegram.ext import MessageHandler, Filters
from bert_classs import *
import pandas as pd
from torch.utils.data import DataLoader
import torch.utils.data
import gluonnlp as nlp
import numpy as np
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
import time

class Chatbot():
    def __init__(self):
        super().__init__()
        self.token = "5560484908:AAE_ObxGlGoQD9UJ6gseYo14ImXXK-8X170"
        self.id = '5510557485'
        self.device = torch.device('cpu')
        self.bertmodel, self.vocab = get_pytorch_kobert_model()
        self.model = BERTClassifier(self.bertmodel, dr_rate=0.5)
        self.model.load_state_dict(torch.load('G:\\내 드라이브\\Watch_You\\01_001_bert_10.pth',map_location=self.device))

        self.max_len = 64
        self.batch_size = 64
        self.count = 0

        self.tokenizer = get_tokenizer()
        self.tok = nlp.data.BERTSPTokenizer(self.tokenizer, self.vocab, lower=False)
        self.df = pd.read_csv('G:\\내 드라이브\\Watch_You\\01_001_bert.csv')
        self.category_list = list(self.df['MAIN'].unique())
        self.bot = telegram.Bot(self.token)
        self.info_message = '''
안녕하세요. 
이수림 카페에 오신 걸 환영합니다.
주문은 키오스크를 이용해 주세요.
그 외 궁금하신 질문은 저에게 맞겨주세요.
        '''
        self.limit = '''
저는 5 글자 이상 말씀해 주셔야 이해할 수 있어요~~
        '''
        self.end = '''
더 필요하신 거 없으세요?
        '''
        self.bot.sendMessage(chat_id=self.id, text=self.info_message)
        self.bot.sendMessage(chat_id=self.id, text=self.limit)

        self.updater = Updater(token=self.token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.updater.start_polling()
        self.echo_handler = MessageHandler(Filters.text, self.handler)
        self.dispatcher.add_handler(self.echo_handler)
        # self.updater.dispatcher.add_handler(CommandHandler('start', self.start))
        self.start_handler = CommandHandler('start', self.start)
        self.dispatcher.add_handler(self.start_handler)

    # def start(self, update):
    #     self.bot.sendMessage(chat_id=self.id, text=self.info_message)
    #     self.bot.sendMessage(chat_id=self.id, text=self.limit)
    def start(self,update, context):
        context.bot.send_message(chat_id=self.id, text="자, 게임을 시작하지.")
        self.bot.sendMessage(chat_id=self.id, text=self.info_message)
        self.bot.sendMessage(chat_id=self.id, text=self.limit)


    def input_category(self,input_data):
        df = pd.read_csv('G:\\내 드라이브\\Watch_You\\input_label.csv')
        input_len = len(df['C'])
        for i in range(input_len):
            menu = df.iloc[i]
            if input_data == menu[1]:
                print(menu[2])
                self.bot.send_message(chat_id=self.id, text=menu[2])
        self.count = 0
        #

    def predict(self,predict_sentence, num):
        data = [predict_sentence, '0']
        dataset_another = [data]
        another_test = BERTDataset(dataset_another, 0, 1, self.tok, self.max_len, True, False)
        test_dataloader = DataLoader(another_test, batch_size=self.batch_size, num_workers=0)
        self.model.eval()

        for batch_id, (token_ids, valid_length, segment_ids,label) in enumerate(test_dataloader):

            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)
            valid_length = valid_length

            label = label.long().to(self.device)

            out = self.model(token_ids, valid_length, segment_ids)
            prediction = out.cpu().detach().numpy().argmax()

            test_eval = []
            for i in out:
                logits = i
                logits = logits.detach().cpu().numpy()
            for k, c in enumerate(self.category_list):
                if np.argmax(logits) == k:
                    value = logits[k]
                    value = round(value * 10)
                    print(num)
                    print(value)
                    if value < num:
                        print(c+'50이하!')
                        self.bot.send_message(chat_id=self.id, text='조금 더 정확하게 말씀해 주세요')
                        self.count = 1
                        print(1)
                        return 1
                    else:
                        print(c+'50이상!')
                        test_eval.append(c)
                        value = logits[k]
                        self.input_category(c)
                        return 0
            # print(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")

    ### 챗봇 답장
    def handler(self,update, context):
        self.user_text = update.message.text  # 사용자가 보낸 메세지를 user_text 변수에 저장
        if  len(self.user_text) < 5:
            time.sleep(1)
            self.bot.send_message(chat_id=self.id, text='너무 짧아요. 5 글자 이상 입력해 주세요' )
        elif self.count == 1:
            a = self.predict(self.user_text, 30)
            print(2)
        elif self.count == 0:
            a = self.predict(self.user_text, 50)
            print(3)
        if a == 0:
            time.sleep(0.7)
            self.bot.sendMessage(chat_id=self.id, text=self.end)

Chatbot()