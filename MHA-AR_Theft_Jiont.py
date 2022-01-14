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


epochs = 14
batch_size = 8
learning_rate = 8e-4
maxlen_sentence = 12
maxlen_word = 120
max_features = 21130
embedding_dims = 200
hidden_size = 50
l2_reg = l2(1e-13)
scale_factor = 0.5
term_num=10
money_num = 9

command = 1


def pad_docs(index_docs, doc_max_sentence_num=maxlen_sentence, sentence_max_word_num=maxlen_word, padding_value=0):
    data = []
    for doc in index_docs:
        doc_data = []
        for sentence in doc:
            if len(sentence) < sentence_max_word_num:
                sentence.extend([padding_value] * (sentence_max_word_num - len(sentence)))
            doc_data.append(sentence[:sentence_max_word_num])
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
                c = [0 for _ in range(10)]
                a = tmp_dic['meta']["term_of_imprisonment"]['imprisonment']
                if 0<=a<=3:
                    c[0] = 1
                if 4 <= a <= 6:
                    c[1] = 1
                if 7 <= a <= 9:
                    c[2] = 1
                if 10 <= a <= 12:
                    c[3] = 1
                if 13 <= a <= 15:
                    c[4] = 1
                if 16 <= a <= 18:
                    c[5] = 1
                if 19 <= a <= 21:
                    c[6] = 1
                if 22 <= a <= 24:
                    c[7] = 1
                if 25 <= a <=60:
                    c[8] = 1
                if a>60:
                    c[9] = 1
                g = [0 for _ in range(money_num)]
                f = tmp_dic['meta']["punish_of_money"]
                if f == 0:
                    g[0] = 1
                if 0 < f <= 1000:
                    g[1] = 1
                if 1000 < f <= 2000:
                    g[2] = 1
                if 2000 < f <= 3000:
                    g[3] = 1
                if 3000 < f <= 4000:
                    g[4] = 1
                if 4000 < f <= 5000:
                    g[5] = 1
                if 5000 < f <= 6000:
                    g[6] = 1
                if 6000 < f <= 10000:
                    g[7] = 1
                if 10000 < f:
                    g[8] = 1
                b = tmp_dic['fact']
                e = [0 for _ in range(10)]
                if "特别巨大" in b:
                    e[0] = 1
                elif "巨大" in b:
                    e[1] = 1
                else: e[2] = 1
                if "累犯" in b:
                    e[3] = 1
                if "自首" or "投案" in b:
                    e[4] = 1
                if "从轻" in b: e[5] = 1
                if "赔退" in b or "退赔" in b or "赔偿" in b:
                    e[6] = 1
                if "谅解" in b or "和解" in b:
                    e[7] = 1
                if "未遂" in b:
                    e[8] = 1
                if "多次" in b:
                    e[9] = 1
                b = b.replace('\r','')
                b = b.replace('\n','')
                pattern = r'。'
                result_list = re.split(pattern, b)
                d.append(result_list[:-1])
                d.append(c)
                d.append(e)
                d.append(g)
                All_d.append(d)
    return All_d


train_data = load_data('theft_train.json')
valid_data = load_data('theft_test.json')
classifier_loss_weight = [0.5687815855281374, 1.3653068872132281, 1.6971210810521902, 3.051981766345029, 2.0526931512723032, 1.3086302255819562, 0.6832761219465854, 1.8273899661657453, 0.4981982593317223, 0.5233279955397669, 0.38025111209226403, 0.3327524015753477, 0.19701602304635912, 0.18528538944313647, 0.18653311660117455]



class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_token_ids, batch_labels,batch_features,batch_money = [],[], [],[]
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
            batch_money.append(d[3])
            if len(batch_token_ids) == batch_size or is_end:
                batch_token_ids = pad_docs(batch_token_ids)
                batch_labels = np.array(batch_labels)
                batch_features = np.array(batch_features)
                batch_money = np.array(batch_money)
                yield ({'input_1': batch_token_ids},
                       {'feature': batch_features,
                        'term':batch_labels,
                        'money':batch_money})
                batch_token_ids, batch_features, batch_labels,batch_money = [], [],[], []


def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
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

        if mask is not None:
            self.a *= K.cast(mask, K.floatx())

        self.a /= K.cast(K.sum(self.a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        self.a = K.expand_dims(self.a)
        weighted_input = x * self.a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def scale(x):
    return x*scale_factor


class Dot_Attention(Layer):
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
        self.W = self.add_weight(shape=(10,input_shape[0][-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(maxlen_sentence,10),
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
        self.a = K.exp(ait)

        if mask is not None:
            self.a *= K.cast(mask, K.floatx())

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
x_sentence = TimeDistributed(model_word_GRU)(input_sen)
x_sentence = Bidirectional(CuDNNGRU(hidden_size, return_sequences=True, kernel_regularizer=l2_reg))(
    x_sentence)  # LSTM or GRU
SenAtt1 = AttentionWithContext()
SenAtt2 = AttentionWithContext()
SenAtt3 = Dot_Attention()
x_sentence_1 = SenAtt1(x_sentence)
x_sentence_2 = SenAtt2(x_sentence)
output_f = Dense(10, activation='sigmoid',name = 'feature')(x_sentence_2)
x_sentence_3 = SenAtt3([x_sentence,output_f])
outputw = layers.concatenate([x_sentence_1,x_sentence_2,x_sentence_3])
output = Dense(term_num,activation='sigmoid', name='term')(outputw)
oo = Dense(money_num,activation='sigmoid', name='money')(outputw)
model = models.Model(input_sen, [output,output_f,oo])
model.summary()

my_losses = {'term': 'binary_crossentropy', 'feature': 'binary_crossentropy','money': 'binary_crossentropy'}

model.compile(
    loss=my_losses,
    optimizer=Adam(learning_rate),
    metrics=['binary_accuracy'],
    loss_weights=[2.,1., 2.]
)


def recognize(text):
    outputw = []
    input_token = []
    for sentence in text:
        text_tokened = tokenizer.tokenize(sentence, maxlen=maxlen_word)
        token_ids_sen = tokenizer.tokens_to_ids(text_tokened)
        input_token.append(list(token_ids_sen))
    outputw.append(input_token)
    output1 = pad_docs(outputw)
    outputz = model.predict(output1)
    output1 = outputz[0]
    output2 = outputz[2]
    result_in_number = 0
    for w in output1[0]:
        if(w>0.5):
            result_in_number = result_in_number+1
    result_in_list = [0 for _ in range(term_num)]
    result_in_list[result_in_number] = 1
    result_in_number_2 = 0
    for w in output2[0]:
        if (w > 0.5):
            result_in_number_2 = result_in_number_2 + 1
    result_in_list_2 = [0 for _ in range(money_num)]
    result_in_list_2[result_in_number_2] = 1
    return result_in_list,result_in_number,result_in_list_2,result_in_number_2


def evaluate(data):
    """评测函数
    """
    val_targ = []
    val_predict = []
    val_targ_num = []
    val_predict_num = []
    val_targ_2 = []
    val_predict_2 = []
    val_targ_num_2 = []
    val_predict_num_2 = []
    for text in tqdm(data, ncols=100):
        pred_in_list, pred_in_number,pred_in_list_2, pred_in_number_2 = recognize(text[0])
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
        val_predict.append(pred_in_list) #断开
        targ_2 = text[3]
        targ_num_2 = 0
        for w in targ_2:
            if (w > 0.5):
                targ_num_2 = targ_num_2 + 1
        targ_in_list_2 = [0 for _ in range(money_num)]
        targ_in_list_2[targ_num_2] = 1
        val_predict_num_2.append(pred_in_number_2)
        val_targ_num_2.append(targ_num_2)
        val_targ_2.append(targ_in_list_2)
        val_predict_2.append(pred_in_list_2)
    Term_S = term_score(val_targ_num,val_predict_num,term_num)
    ACC_0 = Accuarcy_zero(val_targ_num,val_predict_num)
    ACC_1 = Accuarcy_one(val_targ_num,val_predict_num)
    Term_S_2 = term_score(val_targ_num_2, val_predict_num_2, money_num)
    ACC_0_2 = Accuarcy_zero(val_targ_num_2,val_predict_num_2)
    ACC_1_2 = Accuarcy_one(val_targ_num_2,val_predict_num_2)
    F1 = f1_score(val_targ, val_predict,average='macro')
    recall = recall_score(val_targ, val_predict, average='macro')
    precision = precision_score(val_targ, val_predict, average='macro')
    print('P_acc_0:',ACC_0,' P_acc_1:',ACC_1,' P_TermS',Term_S,' M_acc_0:',ACC_0_2,' M_acc_1:',ACC_1_2,' M_TermS',Term_S_2,)
    return F1,recall,precision,Term_S


class Evaluator(callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        super().__init__()
        self.TS = 0

    def on_epoch_end(self, epoch, logs=None):
        model.save('best_armah_theft_J.h5')
        print('model saved!')
        F1,recall,precision,TS = evaluate(valid_data)


if __name__ == "__main__":
    if command == 1:
       evaluator = Evaluator()
       model = load_model('best_armah_theft_J.h5',custom_objects={'AttentionWithContext': AttentionWithContext, 'Dot_Attention':Dot_Attention})
       train_generator = data_generator(train_data, batch_size)
       model.fit(
           train_generator.forfit(),
           steps_per_epoch=len(train_generator),
           epochs=epochs,
           callbacks=[evaluator]
       )
    else:
        model.load_weights('best_armah_theft_J.h5') #if use load_model, we would have name(?) conflict.
        test_data = load_data('traffic_test_3.json')
        evaluate(test_data)
