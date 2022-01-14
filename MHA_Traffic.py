import numpy as np
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField, Bidirectional, BatchNormalization
from keras import Input, metrics, losses, layers, models, callbacks
from keras.engine.saving import load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, concatenate, Flatten, Dropout, Embedding, LSTM, GRU, \
    TimeDistributed, CuDNNGRU, Lambda
from keras.models import Model
from keras.regularizers import l2
from tqdm import tqdm
import os
import json
from sklearn.metrics import f1_score, recall_score, precision_score, mean_absolute_error, r2_score,accuracy_score
from keras.engine import Layer
from keras import initializers as initializers, regularizers, constraints
from keras import backend as K
import re
from Evaluation_func import term_score,Accuarcy_zero,Accuarcy_one

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'
config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)


epochs = 10
batch_size = 8
learning_rate = 8e-4
maxlen_sentence = 12
maxlen_word = 120
max_features = 21130
embedding_dims = 200
hidden_size = 50
l2_reg = l2(1e-13)
scale_factor = 0.5
term_num=15


command = 1


def pad_docs(index_docs, doc_max_sentence_num=maxlen_sentence, sentence_max_word_num=maxlen_word, padding_value=0):
    data = []
    for doc in index_docs:
        doc_data = []
        for sentence in doc:
            # 句子 word 数补齐成 sentence_max_word_num
            if len(sentence) < sentence_max_word_num:
                sentence.extend([padding_value] * (sentence_max_word_num - len(sentence)))
            doc_data.append(sentence[:sentence_max_word_num])
        # 每篇文章句子数补够 doc_max_sentence_num
        if len(doc_data) < doc_max_sentence_num:
            doc_data.extend([[padding_value] * sentence_max_word_num] * (doc_max_sentence_num - len(doc_data)))
        data.append(doc_data[:doc_max_sentence_num])
    data = np.array(data)
    return data


def load_data(filename):
    All_d = []
    with open(filename, encoding='utf-8') as f:
        for l in f.readlines():
            if l:
                d = []
                tmp_dic = json.loads(l)
                c = [0 for _ in range(15)]
                a = tmp_dic['meta']["term_of_imprisonment"]['imprisonment']
                if 4 <= a <= 6:
                    c[0] = 1
                if 7 <= a <= 9:
                    c[1] = 1
                if 10 <= a <= 12:
                    c[2] = 1
                if 13 <= a <= 18:
                    c[3] = 1
                if 19 <= a <= 24:
                    c[4] = 1
                if 25 <= a <= 30:
                    c[5] = 1
                if 31 <= a <= 36:
                    c[6] = 1
                if 37 <= a <= 42:
                    c[7] = 1
                if 43 <= a <= 48:
                    c[8] = 1
                if 49 <= a <= 54:
                    c[9] = 1
                if 55 <= a <= 60:
                    c[10] = 1
                if 61 <= a <= 66:
                    c[11] = 1
                if 67 <= a <= 72:
                    c[12] = 1
                if a>72:
                    c[13] = 1
                b = tmp_dic['fact']
                e = [0 for _ in range(14)]
                if "死亡" in b:
                    e[0] = 1
                elif "重伤" in b:
                    e[1] = 1
                else: e[2] = 1
                if "全部责任" in b:
                    e[3] = 1
                elif "主要责任" in b:
                    e[4] = 1
                else: e[5] = 1
                if "醉酒驾驶" in b or "醉驾" in b:
                    e[6] = 1
                if "吸毒" in b or "毒驾" in b:
                    e[7] = 1
                if "未取得机动车驾驶证" in b or "未取得驾驶证" in b or "无证驾驶" in b:
                    e[8] = 1
                if "超载" in b:
                    e[9] = 1
                if "逃逸" in b or "逃离" in b:
                    e[10] = 1
                if "自首" in b:
                    e[11] = 1
                if "谅解" in b or "从轻处罚" in b or "一次性处理" in b:
                    e[12] = 1
                if "刑事和解" in b:
                    e[13] = 1
                b = b.replace('\r','')
                b = b.replace('\n','')
                pattern = r'。'
                result_list = re.split(pattern, b)
                # pad_doc = pad_docs(result_list[:-1])
                d.append(result_list[:-1])
                d.append(c)
                d.append(e)
                All_d.append(d)
        #print(All_d)
    return All_d


train_data = load_data('traffic_train_2.json')
valid_data = load_data('traffic_test_3.json')
classifier_loss_weight = [0.5687815855281374, 1.3653068872132281, 1.6971210810521902, 3.051981766345029, 2.0526931512723032, 1.3086302255819562, 0.6832761219465854, 1.8273899661657453, 0.4981982593317223, 0.5233279955397669, 0.38025111209226403, 0.3327524015753477, 0.19701602304635912, 0.18528538944313647, 0.18653311660117455]



class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_labels,batch_features = [],[], []
        for is_end, d in self.sample(random):
            token_ids_doc=[]
            for sentence in d[0]:
                #print(sentence)
                text_tokened = tokenizer.tokenize(sentence, maxlen=maxlen_word)
                token_ids_sen = tokenizer.tokens_to_ids(text_tokened)
                token_ids_doc.append(list(token_ids_sen))
            batch_token_ids.append(token_ids_doc)
            batch_labels.append(d[1])
            batch_features.append(d[2])
            if len(batch_token_ids) == batch_size or is_end:
                batch_token_ids = pad_docs(batch_token_ids)
                batch_labels = np.array(batch_labels)
                batch_features = np.array(batch_features)
                yield ({'input_1': batch_token_ids},
                       {'term':batch_labels})
                batch_token_ids, batch_features, batch_labels = [], [], []


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape=(hidden_size,maxlen_sentence,embedding_dims)):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(name='{}_u'.format(self.name), shape=(input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        self.a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            self.a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        self.a /= K.cast(K.sum(self.a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        self.a = K.expand_dims(self.a)
        weighted_input = x * self.a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def scale(x):
    return x*scale_factor


class Dot_Attention(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Dot_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(14,input_shape[0][-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(maxlen_sentence,14),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        super(Dot_Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, xx, mask=None):
        x = xx[0]
        q = xx[1]
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b
        ait = K.batch_dot(uit,q)
        #ait = ait/K.sqrt(14)
        self.a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            self.a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        self.a /= K.cast(K.sum(self.a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        self.a = K.expand_dims(self.a)
        weighted_input = x * self.a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])


def multiple_binary_loss(y_pred,y_true):
    loss = 0
    print('y_pred',y_pred)
    print('y_true',y_true)
    for i in range(term_num):
        loss = loss + classifier_loss_weight[i]*K.binary_crossentropy(y_pred[:][i], y_true[:][i])
    loss = loss/15
    return loss



#HAN_1 = HAN(maxlen_sentence=16, maxlen_word=120, max_features=21130, embedding_dims=200, hidden_size=150, l2_reg=l2(1e-13))
input_word = Input(shape=(maxlen_word,), name='word')
embedder = Embedding(max_features + 1, embedding_dims, input_length=maxlen_word,
                     trainable=True)
embedding_vector = embedder(input_word)
x_word = Bidirectional(CuDNNGRU(hidden_size, return_sequences=True, kernel_regularizer=l2_reg))(embedding_vector)  # LSTM or GRU
x_word = Bidirectional(CuDNNGRU(hidden_size, return_sequences=True, kernel_regularizer=l2_reg))(x_word)
WordAtt = AttentionWithContext()
x_word = WordAtt(x_word)
model_word_GRU = Model(input_word, x_word)

# Sentence part
input_sen = Input(shape=(maxlen_sentence, maxlen_word))
#input_feature = Input(shape=(14,))
x_sentence = TimeDistributed(model_word_GRU)(input_sen)
x_sentence = Bidirectional(CuDNNGRU(hidden_size, return_sequences=True, kernel_regularizer=l2_reg))(
    x_sentence)  # LSTM or GRU
SenAtt1 = AttentionWithContext()
SenAtt2 = AttentionWithContext()
SenAtt3 = Dot_Attention()
x_sentence_1 = SenAtt1(x_sentence)
x_sentence_2 = SenAtt2(x_sentence)
#output_f = Dense(14, activation='softmax')(x_sentence_2)
#x_sentence_3 = SenAtt3([x_sentence,output_f])
output = layers.concatenate([x_sentence_2,x_sentence_1])
output = Dense(term_num,activation='sigmoid', name='term')(output)

# B0 = Dense(2, activation='softmax')(output)
# B1 = Dense(2, activation='softmax')(output)
# B2 = Dense(2, activation='softmax')(output)
# B3 = Dense(2, activation='softmax')(output)
# B4 = Dense(2, activation='softmax')(output)
# B5 = Dense(2, activation='softmax')(output)
# B6 = Dense(2, activation='softmax')(output)
# B7 = Dense(2, activation='softmax')(output)
# B8 = Dense(2, activation='softmax')(output)
# B9 = Dense(2, activation='softmax')(output)
# B10 = Dense(2, activation='softmax')(output)
# B11 = Dense(2, activation='softmax')(output)
# B12 = Dense(2, activation='softmax')(output)
# B13 = Dense(2, activation='softmax')(output)
# B14 = Dense(2, activation='softmax')(output)
#output = layers.concatenate([B0, B1, B2,B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14], name='term')

model = models.Model(input_sen, output,)
model.summary()

my_losses = {'term': 'binary_crossentropy'}

model.compile(
    loss=my_losses,
    optimizer=Adam(learning_rate),  # 用足够小的学习率
    metrics=['binary_accuracy'],
)


def recognize(text):
    output2 = []
    input_token = []
    for sentence in text:
        text_tokened = tokenizer.tokenize(sentence, maxlen=maxlen_word)
        token_ids_sen = tokenizer.tokens_to_ids(text_tokened)
        input_token.append(list(token_ids_sen))
    output2.append(input_token)
    output1 = pad_docs(output2)
    output1 = model.predict(output1)
    output1 = output1[0]
    result_in_number = 0
    for w in output1:
        if(w>0.5):
            result_in_number = result_in_number+1
    result_in_list = [0 for _ in range(term_num)]
    result_in_list[result_in_number] = 1
    return result_in_list,result_in_number


def evaluate(data):
    """评测函数
    """
    val_targ = []
    val_predict = []
    val_targ_num = []
    val_predict_num = []
    for text in tqdm(data, ncols=100):
        pred_in_list,pred_in_number = recognize(text[0])
        targ = text[1]
        targ_num=0
        for w in targ:
            if (w > 0.5):
                targ_num = targ_num+ 1
        targ_in_list = [0 for _ in range(term_num)]
        targ_in_list[targ_num] = 1
        val_predict_num.append(pred_in_number)
        val_targ_num.append(targ_num)
        val_targ.append(targ_in_list)
        val_predict.append(pred_in_list)
    Term_S = term_score(val_targ_num,val_predict_num,term_num)
    ACC_0 = Accuarcy_zero(val_targ_num,val_predict_num)
    ACC_1 = Accuarcy_one(val_targ_num,val_predict_num)
    F1 = f1_score(val_targ, val_predict,average='macro')
    recall = recall_score(val_targ, val_predict, average='macro')
    precision = precision_score(val_targ, val_predict, average='macro')
    acc = accuracy_score(val_targ, val_predict)
    print('F1:',F1,' pre:',precision,' recall:',recall,' acc_0:',ACC_0,' acc_1:',ACC_1,' Term_S',Term_S)
    return F1,recall,precision,Term_S


class Evaluator(callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        super().__init__()
        self.TS = 0

    def on_epoch_end(self, epoch, logs=None):
        model.save('best_mha_traffic.h5')
        print('model saved!')
        F1,recall,precision,TS = evaluate(valid_data)
        # 保存最优
        if TS >= self.TS:
            self.TS= TS
        print(
            'valid:  F1: %.5f, precision: %.5f, recall: %.5f,TS: %.5f best TS: %.5f\n' %
            (F1, precision, recall,TS,self.TS)
        )


if __name__ == "__main__":
    if command == 1:
       evaluator = Evaluator()
       #model = load_model('best_mha_traffic.h5',custom_objects={'AttentionWithContext': AttentionWithContext, 'Dot_Attention':Dot_Attention})
       train_generator = data_generator(train_data, batch_size)
       model.fit(
           train_generator.forfit(),
           steps_per_epoch=len(train_generator),
           epochs=epochs,
           callbacks=[evaluator]
       )
    else:
        model.load_weights('best_mha_traffic.h5') #if use load_model, we would have name(?) conflict.
        test_data = load_data('traffic_test_3.json')
        evaluate(test_data)