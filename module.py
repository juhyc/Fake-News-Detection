import pickle
import re
# 경고 메시지 숨기기
import warnings

import numpy as np
import pandas as pd
from gensim.models.fasttext import FastText
# import tensorflow
from keras import Input
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.layers import BatchNormalization
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Layer
from keras.layers import LeakyReLU  # 추가
from keras.models import Model
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

# from IPython.core.display import display, HTML
warnings.filterwarnings("ignore")

class keras_Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, return_attention=False,
                 **kwargs):
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(keras_Attention, self).__init__(**kwargs)


    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]


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


def rgb_to_hex(rgb):
    return '#%02x%02x%02x%02x' % rgb


def attention2color(attention_score):
    r = 255 - int(attention_score * 255)
    color = rgb_to_hex((255, 0, 0, 255-r))
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

    title_token = []
    title_color = []
    body_token = []
    body_color = []

    for token, title_attention in zip(title_decoded_text, title_attentions_text):
        title_token.append(token)
        title_color.append(attention2color(title_attention))

    for token, body_attention in zip(body_decoded_text, body_attentions_text):
        body_token.append(token)
        body_color.append(attention2color(body_attention))

    return label_probs, title_color, title_token, body_color, body_token


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

    embedding_matrix = np.zeros((len(word2index) + 1, 100))
    for word, i in word2index.items():
        embedding_vector = fasttext_model.wv[word]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # 모델 로드: one input biLSTM + Attention mechanism
    inputs = Input(shape=(263,))

    embedded_inputs = Embedding(len(word2index) + 1,
                                100,
                                weights=[embedding_matrix],
                                trainable=False,
                                name='embedding')(inputs)

    outs = Bidirectional(LSTM(100, dropout=0.2, return_sequences=True))(embedded_inputs)
    outs = BatchNormalization()(outs)
    sentence, word_scord = keras_Attention(return_attention=True, name="attention_vec")(outs)
    fc = Dense(256, kernel_initializer='he_normal', activation=LeakyReLU(alpha=0.3))(sentence)
    fc = Dense(128, kernel_initializer='he_normal', activation=LeakyReLU(alpha=0.3))(fc)
    fc = Dense(64, kernel_initializer='he_normal', activation=LeakyReLU(alpha=0.3))(fc)
    output = Dense(2, activation='softmax')(fc)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    model.load_weights('data/1_input_bilstm_attention_ver219.ckpt').expect_partial()  # 추가

    label_probs, title_color, title_token, body_color, body_token = visualize_attention(model, word2index, x_)

    return label_probs, title_color, title_token, body_color, body_token