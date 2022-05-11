import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re

import soynlp

from gensim.models.fasttext import FastText

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras.models import Model

#from IPython.core.display import display, HTML

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings("ignore")

def del_special(df):
    # 특수문자, 숫자 제거
    pattern = re.compile('[^ a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣]+')
    title_prep = [pattern.sub(" ", doc) for doc in df['title']]
    body_prep = [pattern.sub(" ", doc) for doc in df['body']]
    df['title_prep'] = title_prep
    df['body_prep'] = body_prep

    return df

def tokenize_with_del_stopword(df, tokenizer, noun_scores):
    # 토큰화
    title_tok = [tokenizer.tokenize(title) for title in df['title_prep']]
    body_tok = [tokenizer.tokenize(body) for body in df['body_prep']]

    title_noun = []
    body_noun = []

    # 불용어 제거
    stopwords = ['의', '가', '이', '은', '들', '는', '좀', ' 잘', '걍', '과', '도', '를', '으로',
                 '자', '에', '와', '한', '하다', '다', '하', '고', '인', '듯',
                 '번째', '만건', '천건', '건', '곳']

    for tok in title_tok:
        nouns = [word for word in tok if word in noun_scores.keys() and word not in stopwords]
        title_noun.append(nouns)

    for tok in body_tok:
        nouns = [word for word in tok if word in noun_scores.keys() and word not in stopwords]
        body_noun.append(nouns)

    df['title_noun'] = title_noun
    df['body_noun'] = body_noun

    return df

def noun2sequences(df, word2index):
    title_sequences = []
    body_sequences = []
    for noun in df['title_noun']:
        seq = [word2index[word] for word in noun if word in word2index.keys()]
        title_sequences.append(seq)

    for noun in df['body_noun']:
        seq = [word2index[word] for word in noun if word in word2index.keys()]
        body_sequences.append(seq)

    df['title_sequences'] = title_sequences
    df['body_sequences'] = body_sequences
    return df

def separated_padding_(df, title_idx_col, body_idx_col,  # 데이터프레임, 정수화 된 title column명, 정수화 된 body column명,
                       max_title_seq_len, max_body_seq_len):  # 제목 길이, 본문 길이

    MAX_SEQUENCE_LENGTH = max_title_seq_len + max_body_seq_len

    pad_title_sequences = pad_sequences(df[title_idx_col], maxlen=max_title_seq_len, padding='post', truncating='post')
    pad_body_sequences = pad_sequences(df[body_idx_col], maxlen=max_body_seq_len, padding='post', truncating='post')
    sequences = np.concatenate((pad_title_sequences, pad_body_sequences), axis=1)
    labels = df['label']

    _x_ = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    _y_ = to_categorical(np.asarray(labels), num_classes=2)  # label을 2 class로 범주화 [0,1], [1,0] 형태

    return _x_, _y_

# 시각화
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def attention2color(attention_score):
    attention_score = np.asarray(attention_score, dtype=int)
    r = 255 - int(attention_score * 255)
    color = rgb_to_hex((255, r, r))
    return str(color)

def visualize_attention(model, word2index, x_):
    # Make new model for output predictions and attentions
    model_att = Model(inputs=model.input, \
                      outputs=[model.output, model.get_layer('attention_vec').output[-1]])

    title_tokenized_sample = [wordidx for wordidx in x_[0][:9] if wordidx != 0]
    body_tokenized_sample = [wordidx for wordidx in x_[0][9:] if wordidx != 0]
    label_probs, attentions = model_att.predict(x_[0:1])  # Perform the prediction

    # Get decoded text and labels
    id2word = dict(map(reversed, word2index.items()))
    title_decoded_text = [id2word[word] for word in title_tokenized_sample]
    body_decoded_text = [id2word[word] for word in body_tokenized_sample]

    # Get classification
    label = np.argmax((label_probs > 0.5).astype(int).squeeze())  # Only one
    label2id = ['Real News', 'Fake News']

    # Get word attentions using attenion vector
    token_attention_dic = {}
    max_score = 0.0
    min_score = 0.0

    attentions_text = attentions[0][np.where(x_[0] != 0)]

    # attention 값 정규화
    attentions_text = (attentions_text - np.min(attentions_text)) / (np.max(attentions_text) - np.min(attentions_text))

    # 다른 HTML로 넘겨줄 경우, 이 변수들 return
    title_attentions_text = attentions_text[np.where(x_[0][:9] != 0)]
    body_attentions_text = attentions_text[np.where(x_[0] != 0)[len(title_attentions_text):]]

    for token, title_attention in zip(title_decoded_text, title_attentions_text):
        title_token = token
        title_attention = title_attention

    for token, body_attention in zip(body_decoded_text, body_attentions_text):
        body_token = token
        body_attention = body_attention

    return label_probs, attention2color(title_attention), title_token, attention2color(body_attention), body_token

# title = "목포항서 유조부선 구멍 생겨…긴급 방제"
# body = "부산항 북항에 정박해 있던 유조부선(유조선의 부선)에서 기름이 바다로 유출돼 해경이 긴급 방제 작업에 나섰다. 4일 부산해경에 따르면 이날 오전 10시 44분쯤 부산시 동구 좌천동 부산항 북항 제5 부두에서 기름 공급 작업을 준비 중이던 유조부선 A호 선내에 있던 중질유 일부가 바다로 유출됐다. 해경은 방제정 등 총 6척의 선박을 동원해 사고 선박 주변에 오일펜스를 설치했다. 이어 유흡착재를 이용해 바다 위 검은색 기름띠 등 방제 작업을 진행하고 있다. 해경은 A호가 화물유 자체 이송 중 밸브 오작동으로 중질유가 바다로 흘러넘쳐 발생한 것으로 보고 있다. 해경은 방제작업이 마무리되는 대로 선박 관계자 등을 상대로 자세한 사고 원인과 유출량 등을 조사할 예정이다."

def Visualization_Result(title, body, label):
    df = pd.DataFrame(columns=['title', 'body', 'label'])
    df.loc[0] = [title, body, 1]

    with open('data/soy_tokenizer.pkl', 'rb') as f:
        tokenizer_ = pickle.load(f)

    with open('data/soy_noun_score.pkl', 'rb') as f:
        noun_score_ = pickle.load(f)

    df = del_special(df)
    df = tokenize_with_del_stopword(df, tokenizer_, noun_score_)
    df_noun = df[['title_noun', 'body_noun', 'label']]

    # Load fasttext model
    fasttext_model = FastText.load('data/fasttext_model_100')
    word2index = {tok: tok_index + 1 for tok_index, tok in enumerate(fasttext_model.wv.index2word)}

    df_noun = noun2sequences(df_noun, word2index)
    df_seq = df_noun[['title_noun', 'body_noun', 'title_sequences', 'body_sequences', 'label']]

    # padding
    max_t_seq_len = 9
    max_b_seq_len = 254
    x_, y_ = separated_padding_(df_seq, 'title_sequences', 'body_sequences', max_t_seq_len, max_b_seq_len)

    # 모델 로드
    # load: 1 input biLSTM + Attention mechanism
    filepath = 'data/1input_bilstm_att_model'
    bilstm_att_model = load_model(filepath=filepath)

    label_probs, title_color, title_token, body_color, body_token = visualize_attention(bilstm_att_model, word2index, x_)

    return label_probs, title_color, title_token, body_color, body_token
#
# if __name__ == "__main__":
#     cat_num=3
#     keyword='박원순'
#     print(timeline(cat_num, keyword))