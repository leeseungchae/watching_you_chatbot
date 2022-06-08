from bert_classs import *
import pandas as pd
from torch.utils.data import DataLoader
import torch.utils.data
import gluonnlp as nlp
import numpy as np
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
import re
from datetime import datetime
def order():

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
         print('텀블러를 카운터에 제시해주세요')

    elif input_data == "리필문의":
         print('리필은 아메리카노만 가능합니다.')

    elif input_data == "샷토핑추가시추가금액문의":
         print('주문창에서 추가해주세요')

    elif input_data == "쿠폰멤버십적립문의" or input_data == "현금영수증요청" or input_data == "모바일페이결제" or input_data == "영수증요청" or input_data == "결제문의" or input_data == "현금영수증문의" or input_data == "핸드폰번호이용적립" or input_data == "분할결제" or input_data == "영수증문의":
         print('결제창으로 돌아가서 다시 확인해주세요')
         print('결제창으로 돌아가기 알고리즘')

    elif input_data == "메뉴추천요구":
        print("카페 시그니쳐메뉴 구매창으로 이동")

    elif input_data == "메뉴추천요구":
        print("카페 시그니쳐메뉴 구매창으로 이동")

    elif input_data == "신메뉴문의" or input_data == "메뉴판요구" or input_data == "메뉴판문의" or input_data == "인기메뉴문의":
        print("메뉴창 보여주기")

    elif input_data == '와이파이문의':
        print("비밀번호는 영수증에 써져있습니다.")

    elif input_data == "화장실위치문의" or input_data == "화장실비밀번호문의":
        print('건물을 나가시고 왼쪽문 2층에 있습니다.')
        print('비밀번호는 159456입니다.')

    elif input_data == "주문취소":
        print("상품생산 중에는 취소가 불가능합니다.")

    elif input_data == "영업시간문의" or input_data == "휴일문의":
        print('영업시간 : 09:00 ~ 21:00')
        print("휴일 : 격주 금요일")

    elif input_data == "기프티콘쿠폰멤버십사용문의" or input_data == "기프티콘쿠폰멤버십결제요구" or input_data == "기프티콘쿠폰멤버십메뉴변경문의" or input_data == "기프티콘쿠폰멤버실결제요구":
        print("기프티콘 관련창으로 이동하겠습니다.")

    elif input_data == "사이즈문의" or input_data == "사이즈요구":
        print('주문창으로 이동하겠습니다.')

    elif input_data == "테이크아웃문의" or input_data == "테이크아웃할인문의" or input_data == "테이카아웃요구":
        print("테이크아웃은 키오스크에 문의해주세요")

    elif input_data == "메뉴추가문의" or input_data == "시럽설탕요구" or input_data == "옵션추가요구" or input_data == "물우유량요구" or input_data == "식기류요구" or input_data == "얼음량요구" or input_data == "얼음요구" or input_data == "시럽요구" or input_data == "시럽":
        print("추가 메뉴창으로 이동하겠습니다.")

    elif input_data == "식기류쟁반반납":
        print('반납대에 놔주세요')
    else:
        print("카운터의 문의하세요")


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
                if np.argmax(logits) == k:
                    value = logits[k]
                    value = round(value * 10)
                    if value < 50:
                        print("죄송합니다 다시 말씀해주세요")
                        print(value)
                        test_eval.append(c)
                        value = logits[k]
                        test_eval_2.append(value)
                        print(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.1")

                        return False
                    else:
                        if c == '일반주문':
                            order()
                        else:
                            input_category(c)

                            test_eval.append(c)
                            value = logits[k]
                            test_eval_2.append(value)
                            print(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.2")
                            print(test_eval_2[0])

device = torch.device('cpu')
bertmodel,vocab = get_pytorch_kobert_model()
model = BERTClassifier(bertmodel, dr_rate=0.5)
model.load_state_dict(torch.load('G:\\내 드라이브\\Watching_You_PJ\\project_ChatBot\\03_DL_Model\\01_001_bert_10.pth',
                                 map_location=device))
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
df = pd.read_csv('G:\\내 드라이브\\Watching_You_PJ\\project_ChatBot\\02_preprocessing\\01_001_bert.csv')
category_list = list(df['MAIN'].unique())
#
# def Order() {
#
# }

end = 1
while end == 1:
    sentence = input("하고싶은 말을 입력해주세요(5자 이상) : ")
    if sentence == '종료':
        break
    predict(sentence)

    print("\n")