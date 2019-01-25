#-*- coding: utf-8 -*-
import yaml
from gensim.models import word2vec
from keras.preprocessing import sequence
import re,jieba
import numpy as np
from keras.callbacks import EarlyStopping
from collections import Counter
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense,Activation
from numpy import array
import sys
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_yaml,load_model
jieba.load_userdict("data/final_dict.txt")

sys.setrecursionlimit(10000)
maxlen = 30
import os



def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if combined and (model is not None):
        # gensim_dict = Dictionary()
        # gensim_dict.doc2bow(model.wv.vocab.keys(),
        #                     allow_update=True)
        # w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
        # w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量


        ''' Words become integers
        '''
        data=[]
        sentences = combined.strip().split(' ')
        for i in range(31 if len(sentences)>=31 else len(sentences)):
            word=sentences[i]
            try:
#               word = unicode(word, errors='ignore')
                data.append(model[word])
            except:
#                       new_txt.append(np.array([0.0]*200))     #word2vec模型中没有的词语剔除
                 pass
        data=sequence.pad_sequences([data], maxlen=maxlen, padding='post',dtype='float32')  # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return data[0]
    return    None

def create_dictionaries_sentence(model,sentence):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if sentence and model:
        # gensim_dict = Dictionary()
        # gensim_dict.doc2bow(model.wv.vocab.keys(),
        #                     allow_update=True)
        # w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
        # w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量


        ''' Words become integers
        '''
        data=[]
        sentences = sentence[0].strip().split(' ')
        for i in range(31 if len(sentences)>=31 else len(sentences)):
            word=sentences[i]
            try:
#               word = unicode(word, errors='ignore')
                data.append(model[word])
            except:
#                       new_txt.append(np.array([0.0]*200))     #word2vec模型中没有的词语剔除
                 pass
        data=sequence.pad_sequences([data], maxlen=maxlen, padding='post',dtype='float32')  # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return data[0]
    return    None



def input_transform(string):
    pattern = re.compile(u"[\u4e00-\u9fa5]+")
    result = re.findall(pattern, string)
    words=''
    for sentence in result:
        words+=' '+' '.join([x for x in jieba.lcut(sentence.strip())])
    tmp_list = []
    tmp_list.append(words)
    # words=np.array(tmp_list).reshape(1,-1)
#    model = word2vec.Word2Vec.load('data/model/word2vec/word2vec.model')
    model = word2vec.Word2Vec.load('data/model/word2vec/word2vec')
    combined = create_dictionaries_sentence(model, tmp_list)
    return combined

# def loadfile():
#     fopen = open('data/koubei/pos.txt', 'r')
#     pos = []
#     for line in fopen:
#         pos.append(line.strip())
#     fopen = open('data/koubei/pos_400000.txt', 'r')
#     for line in fopen:
#          pos.append(line.strip())
#     fopen = open('data/koubei/neg.txt', 'r')
#     neg = []
#     for line in fopen:
#         neg.append(line.strip())
#     fopen = open('data/koubei/neg_400000.txt', 'r')
#     for line in fopen:
#         neg.append(line.strip())
#     combined=np.concatenate((pos[:100000], neg[:100000]))
#     labels = np.concatenate((np.ones(100000,dtype=int), np.zeros(100000,dtype=int)))
#     return combined,labels
#
def tokenizer(text_list):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    #text = [jieba.lcut(document.replace('\n', '')) for str(document) in text]
    result_list = []
    # for text in text_list:
    #     result_list.append(' '.join(jieba.cut(text)).encode('utf-8').strip())
#    return result_list

    for text in text_list:
        string=text.decode('utf-8')
        pattern = re.compile(u"[\u4e00-\u9fa5]+")
        result = re.findall(pattern, string)
        words=''
        for sentence in result:
            words+=' '+' '.join([x for x in jieba.lcut(sentence.strip())])
        result_list.append(words.strip())
    return result_list


def save_train_corpus(data,labels):
    fou=open('data/train_corpus.txt','w')
    model = word2vec.Word2Vec.load('data/model/word2vec/word2vec')

    for line, label in zip(data, labels):
        tmp_line=create_dictionaries(model,line)
        tmp = []
        if tmp_line is not None  and tmp_line.shape==(30,200):
            for x in tmp_line:
                tmp.append(','.join(map(lambda x: str(x), x)))
            tmp.append(str(label))
            re_tmp = ';'.join(tmp)
            fou.write(re_tmp + '\n')
    fou.close()

def load_train_corpus(file):
    file = open(file)
    lines=[]
    i=0
    tmp=''
    while i <120000:
        i += 1
        if  i>80000:
            lines.append(file.readline())
        tmp=file.readline()
    data = []
    labels = []
    i=0
    coun=[]
    for line in lines:
        words = line.strip().split(';')
        tmp=[]
        if len(words)==31:
            i+=1
            for word in words[:-1]:
                tmp+=map(float, word.strip().split(','))
            data+=tmp
            labels.append(array([int(words[-1])]))
            coun.append(words[-1])
    lines = []
    print  'train_pos_neg' ,Counter(coun).most_common()
    coun = []
    data=np.array(data)
    data,labels=data.reshape(i,30,200),array(labels)
    return array(data),array(labels)

def load_train_corpus_batch(batch_size,file):
    data = []
    labels = []
    i=0
    coun = []
    while i <batch_size:
        line=file.readline()
        words = line.strip().split(';')
        tmp=[]
        if len(words)==31:
            i+=1
            for word in words[:-1]:
                tmp+=map(float, word.strip().split(','))
            data += tmp
            labels.append(array([int(words[-1])]))
            coun.append(words[-1])
    print  'train_pos_neg', Counter(coun).most_common()
    data=np.array(data)
    data,labels=data.reshape(i,30,200),array(labels)
    yield array(data),array(labels)


def get_data(data,labels):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)
    return array(x_train),y_train,array(x_test),y_test

def shuffle_data(data,labels):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    return  data,labels

def train_lstm(x_train,y_train,x_test,y_test):
    print x_train.shape
    model=Sequential()
    model.add(LSTM(100,input_dim=200,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print 'Compiling the Model...'
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print "Train..."
    batch_size=128
    n_epoch=15
    early_stopping = EarlyStopping(monitor='val_loss', patience = 2, verbose = 1)

    model.fit(x_train, y_train, batch_size=128,
              nb_epoch=2, verbose=1,validation_data=(x_test, y_test),callbacks=[early_stopping])
    print "Evaluate..."
    score = model.evaluate(x_test, y_test,batch_size=batch_size)

    # yaml_string = model.to_yaml()
    #
    # with open('data/model/lstm/lstm_koubei.yml', 'w') as outfile:
    #     outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('data/model/lstm/lstm_koubei.h5')
    model.save('data/model/lstm/lstm_koubei1.h5')
    print 'Test score:', score


def train_lstm_batch(file):
    model=Sequential()
    model.add(LSTM(100,input_dim=200,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print 'Compiling the Model...'
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=[auc,'accuracy'])
    print "Train..."
    batch_size=5
    n_epoch=15
    early_stopping = EarlyStopping(monitor='val_loss', patience = 2, verbose = 1)
    (x_train, y_train)=load_train_corpus_batch(batch_size,file)
    x_test, y_test=load_train_corpus_batch(20000,file)
    model.fit_generator(load_train_corpus_batch(batch_size,file))

    model.fit(x_train, y_train, batch_size=128,
              nb_epoch=100, verbose=1,validation_data=(x_test, y_test),callbacks=[early_stopping])
    print "Evaluate..."
    score = model.evaluate(x_test, y_test,batch_size=batch_size)

    # yaml_string = model.to_yaml()
    #
    # with open('data/model/lstm/lstm_koubei.yml', 'w') as outfile:
    #     outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('data/model/lstm/lstm_koubei.h5')


    model.save('data/model/lstm/lstm_koubei.model')
    print 'Test score:', score



def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
def roc(y_true, y_pred):
    auc = tf.metrics.roc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def tmp_lstm_predict(string):
    model=load_model('data/model/lstm/lstm_koubei1.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    print model.inputs
    data = input_transform(string).reshape([1, 30, 200])
    data.reshape(1, -1)
    result=model.predict(data)
    print(result)


def lstm_predict(string):
#    print 'loading model......'
    with open('data/model/lstm/tmp/lstm_koubei.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

#    print 'loading weights......'
    model.load_weights('data/model/lstm/tmp/lstm_koubei.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    data=input_transform(string).reshape([1,30,200])
    data.reshape(1,-1)
    #print data
    result=model.predict_classes(data)
    if result[0][0]==1:
        print(string,' positive')
    else:
        print(string,' negative')
def lstm_predict_batch(data):
    result=[]
    tokenwords = tokenizer(data)
    model = word2vec.Word2Vec.load('data/model/word2vec/word2vec')
    for line  in tokenwords:
        tmp_line=create_dictionaries(model,line)
        if tmp_line is not None  and tmp_line.shape==(30,200):
            result.append(tmp_line)
    result=array(result)


    with open('data/model/lstm/lstm_koubei.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    #    print 'loading weights......'
    model.load_weights('data/model/lstm/lstm_koubei.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    result = model.predict_classes(result)
    result=map(lambda  x : 'positive ' if x[0]==1   else ' negative',result)
    return result








def lstm_predict_prob(string):
#    print 'loading model......'
    with open('data/model/lstm/tmp/lstm_koubei.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

#    print 'loading weights......'
    model.load_weights('data/model/lstm/tmp/lstm_koubei.h5')
    model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    # data = input_transform(string)
    data = input_transform(string).reshape([1, 30, 200])
    data.reshape(1, -1)
        # print data
    result = model.predict(data)
    print(result[0][0])




