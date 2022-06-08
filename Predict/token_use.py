import telegram
from telegram.ext import Updater
from telegram.ext import MessageHandler, Filters
from bert_classs import *
import pandas as pd
from torch.utils.data import DataLoader
import torch.utils.data
import gluonnlp as nlp
import numpy as np
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
import re
token = "5560484908:AAE_ObxGlGoQD9UJ6gseYo14ImXXK-8X170"
id = '5510557485'

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

API_key = 'knI%2FsEhuhoIf37FOmsc8uCq6qdcCXaJU9%2BKHEwgtLzMWGJ7A7LtC3w3Z3JvKzcE4cSrxn6reCcJi2FzIcKvKAQ%3D%3D'

def order(user_text):

    menu_ca = ["메뉴", "아이스/핫", "사이즈", "개수(10개미만)", "포장", "결제", "핸드폰"]

    menu_ca_2 = []

    menu_list = ['아메리카노', '카페라떼']
    tem_list = ['아이스', '핫']
    size_list = ['S(237ml)', 'R(355ml)', 'l(473ml)']
    count = [i for i in range(1, 11)]
    togo_list = ['매장', '포장']
    pay_list = ['신용카드', '모바일결제']
    phone_no = {}

    menu_list = {i + 1: string for i, string in enumerate(menu_list)}
    tem_list = {i + 1: string for i, string in enumerate(tem_list)}
    size_list = {i + 1: string for i, string in enumerate(size_list)}
    count = {i + 1: string for i, string in enumerate(count)}
    togo_list = {i + 1: string for i, string in enumerate(togo_list)}
    pay_list = {i + 1: string for i, string in enumerate(pay_list)}
    phone_no = {i + 1: string for i, string in enumerate(phone_no)}

    menu_ca_2.append(menu_list)
    menu_ca_2.append(tem_list)
    menu_ca_2.append(size_list)
    menu_ca_2.append(count)
    menu_ca_2.append(togo_list)
    menu_ca_2.append(pay_list)
    menu_ca_2.append(phone_no)

    menu_ca_2 = {i + 1: string for i, string in enumerate(menu_ca_2)}

    for i, menu in enumerate(menu_ca):

        print("-----" + str(menu) + " " "선택" "------")
        for key, value in menu_ca_2[i + 1].items():
            print(key, "", value)

        num = input("숫자만 입력해주세요")

        num = re.sub(r'[^0-9]', '', num)
        value = menu_ca_2[i + 1].get(int(num))

        if menu == "개수":
            num = re.sub(r'[^0-9]', '', num)
            value = int(num)

            if value > 10:
                value = None
                print("10개 이상은 카운터 문의해주세요")

        if menu == "핸드폰":
            num = re.sub(r'[^0-9]', '', num)
            value = str(num)

            if 9 < len(value) < 12:
                value = None
                print("너무 짧아요")

        while value is None:

            print("다시 입력해주세요.")
            print("-----" + str(menu) + " " "선택" "------")
            for key, value in menu_ca_2[i + 1].items():
                print(key, "", value)
            num = input("숫자만 입력해주3세요")
            num = re.sub(r'[^0-9]', '', num)
            value = menu_ca_2[i + 1].get(int(num))

def input_category(input_data):

    if input_data == "일반주문" or input_data == "더치커피티백판매문의":
        print("일반주문입니다.")
    elif input_data == "텀블러사용":
         bot.send_message(chat_id=id, text='텀블러를 카운터에 제시해주세요')

    elif input_data == "리필문의":
         bot.send_message(chat_id=id, text='리필은 아메리카노만 가능합니다.')

    elif input_data == "샷토핑추가시추가금액문의":
         bot.send_message(chat_id=id, text='주문창에서 추가해주세요')

    elif input_data == "쿠폰멤버십적립문의" or input_data == "현금영수증요청" or input_data == "모바일페이결제" or input_data == "영수증요청" or input_data == "결제문의" or input_data == "현금영수증문의" or input_data == "핸드폰번호이용적립" or input_data == "분할결제" or input_data == "영수증문의":
         bot.send_message(chat_id=id, text='결제창으로 돌아가서 다시 확인해주세요')
         print('결제창으로 돌아가기 알고리즘')

    elif input_data == "메뉴추천요구":
        print("카페 시그니쳐메뉴 구매창으로 이동")

    elif input_data == "메뉴추천요구":
        print("카페 시그니쳐메뉴 구매창으로 이동")

    elif input_data == "신메뉴문의" or input_data == "메뉴판요구" or input_data == "메뉴판문의" or input_data == "인기메뉴문의":
        print("메뉴창 보여주기")

    elif input_data == '와이파이문의':
        bot.send_message(chat_id=id, text="비밀번호는 영수증에 써져있습니다.")

    elif input_data == "화장실위치문의" or input_data == "화장실비밀번호문의":
        bot.send_message(chat_id=id, text='건물을 나가시고 왼쪽문 2층에 있습니다\n비밀번호는 159456입니다..')

    elif input_data == "주문취소":
        bot.send_message(chat_id=id, text="상품생산 중에는 취소가 불가능합니다.")

    elif input_data == "영업시간문의" or input_data == "휴일문의":
        bot.send_message(chat_id=id, text='영업시간 : 09:00 ~ 21:00\n휴일 : 격주 금요일')

    elif input_data == "기프티콘쿠폰멤버십사용문의" or input_data == "기프티콘쿠폰멤버십결제요구" or input_data == "기프티콘쿠폰멤버십메뉴변경문의" or input_data == "기프티콘쿠폰멤버실결제요구":
        print("기프티콘 관련창으로 이동하겠습니다.")

    elif input_data == "사이즈문의" or input_data == "사이즈요구":
        print('주문창으로 이동하겠습니다.')

    elif input_data == "테이크아웃문의" or input_data == "테이크아웃할인문의" or input_data == "테이카아웃요구":
        bot.send_message(chat_id=id, text="테이크아웃은 키오스크에 문의해주세요")

    elif input_data == "메뉴추가문의" or input_data == "시럽설탕요구" or input_data == "옵션추가요구" or input_data == "물우유량요구" or input_data == "식기류요구" or input_data == "얼음량요구" or input_data == "얼음요구" or input_data == "시럽요구" or input_data == "시럽":
        print("추가 메뉴창으로 이동하겠습니다.")

    elif input_data == "식기류쟁반반납":
        bot.send_message(chat_id=id, text='반납대에 놔주세요')
    else:
        bot.send_message(chat_id=id, text="카운터의 문의하세요")

def predict(predict_sentence):
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
                    # print("죄송합니다 다시 말씀해주세요")
                    # print(value)
                    # # print()
                    # test_eval.append(c)
                    # value = logits[k]
                    # test_eval_2.append(value)
                    # print(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")

                    return 'more'
                else:
                    if c == '일반주문':
                        order()
                    else:
                        input_category(c)
                    test_eval.append(c)
                    value = logits[k]
                    test_eval_2.append(value)
        print(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")
        return test_eval[0]


bot = telegram.Bot(token)
info_message = '''
- 1. 일반 주문 
- 2. 결제 요쳥 
- 3. 메뉴문의  
- 4. 와이파이 문의
- 5. 화장실 문의
- 6. 영업시간 문의
- 7. 휴일 문의
- 8. 사이즈 문의 
- 9. 기프티콘 이용 문의
- 10. 기타
'''
bot.sendMessage(chat_id=id, text=info_message)

updater = Updater(token=token, use_context=True)
dispatcher = updater.dispatcher
updater.start_polling()


### 챗봇 답장
def handler(update, context):
    user_text = update.message.text  # 사용자가 보낸 메세지를 user_text 변수에 저장
    if  len(user_text) < 5:
        bot.send_message(chat_id=id, text="내용이 너무 짧습니다. 조금 더 길게 설명해 주세요")
        bot.sendMessage(chat_id=id, text=info_message)

    else:
        category = predict(user_text)
        category = category.replace(" ", "")
        print(category)
        if category == 'more':
            bot.send_message(chat_id=id, text="죄송합니다. 내용을 못 알아 드렸습니다.\n조금 더 자세히 설명해 주세요")

        bot.sendMessage(chat_id=id, text=info_message)

echo_handler = MessageHandler(Filters.text, handler)
dispatcher.add_handler(echo_handler)


#
# if __name__ == '__main__':
#     main()