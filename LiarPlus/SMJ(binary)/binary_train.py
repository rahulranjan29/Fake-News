# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 17:13:27 2019

@author: Rahul Verma
"""

import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.base import TransformerMixin
#from nltk.tokenize import word_tokenize
import re
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

## CODE HAS BEEN CLEANED.ONLY NECESSARY CODES ARE AVAILABLE


##Load train data





def preprocess(data):
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•−β∅³π‘₹´°£€\×™√²—–&'
    replacer_tag = 'tag' 
    replacer_name = 'username'
    
    def clean_special_chars(text, punct):
        text = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))', 'llinkk', text)
        text = re.sub(r'@[\w]+', replacer_name, text)
        text = re.sub(r'#[\w]+', replacer_tag, text)
        
        text = re.sub(r'\d+',"10", text) 
        
        for search_text, replace_text in [
            ('?', ' question '),
            ('!', ' exclamation '),
            ('.', ' dot ')
        ]:
            text = text.replace(search_text, replace_text)
       
        text=re.sub('[\ \n]+',' ',text)

        for p in punct:
            text = text.replace(p, ' ')
            
        return text

    data = clean_special_chars(data, punct)
    return data

class MeanEmbeddingTransformer(TransformerMixin):
    
    def __init__(self,args):
        self.args=args
        self._vocab, self._E = self._load_words()
        
    
    def _load_words(self):
        E = {}
        vocab = []
#'D:\\glove\\glove.6B.100d.txt'
        with open(self.args.embed_path, 'r', encoding="utf8") as file:# Put path to 100D Glove Embeddings here
            for i, line in enumerate(file):
                l = line.split(' ')
                if l[0].isalpha():
                    v = [float(i) for i in l[1:]]
                    E[l[0]] = np.array(v)
                    vocab.append(l[0])
        return np.array(vocab), E            

    
    def _get_word(self, v):
        for i, emb in enumerate(self._E):
            if np.array_equal(emb, v):
                return self._vocab[i]
        return None
    
    def _doc_mean(self, doc):
        return np.mean(np.array([self._E[w.lower().strip()] for w in doc if w.lower().strip() in self._E]), axis=0)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([self._doc_mean(doc) for doc in X])
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

def to_categorical(data, nb_classes):
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def normalize(list_v,max_value):
    newList = [x / max_value for x in list_v]
    return np.asarray(newList).reshape((len(list_v),1))

def count_tokenizer(sent):
    vect=CountVectorizer(stop_words='english')
    return vect.fit_transform(sent)

def load_sentences(data):
        sentences=[]
        labels=[]
        party=[]
        state=[]
        venue=[]
        job=[]
        subject=[]
        speaker=[]
        barely_true=[]
        false_counts=[]
        half_true=[]
        mostly_true=[]
        pants_on_fire=[]
        #count_words=0
        for i in range(len(data)):
            sent,label,justification=data['statement'][i],data['binary_label'][i],data['justification'][i]
           # count_words+=len(sent.split(' '))
            if justification is not np.nan:
                comb=sent+justification
            else:
                comb=sent ## Some statements do not have justification, in such a case use just the statement
            comb=preprocess(comb)
            party.append(data['party_id'][i]),state.append(data['state_id'][i]),venue.append(data['venue_id'][i]),job.append(data['job_id'][i])
            subject.append(data['subject_id'][i]),speaker.append(data['speaker_id'][i])
            barely_true.append(data['barely true counts'][i])
            false_counts.append(data['false counts'][i])
            half_true.append(data['half true counts'][i])
            mostly_true.append(data['mostly true counts'][i])
            pants_on_fire.append(data['pants on fire counts'][i])
            sentences.append(comb)
            labels.append(label)
        #print(count_words/len(sentences))
        
        bt=np.max(state)
        fc=np.max(party)
        ht=np.max(job)
        mt=np.max(venue)
        pof=np.max(subject)
        barely_true=normalize(barely_true,bt)
        false_counts=normalize(false_counts,fc)
        half_true=normalize(half_true,ht)
        mostly_true=normalize(mostly_true,mt)
        pants_on_fire=normalize(pants_on_fire,pof)
        assert len(sentences)==len(labels)
        return sentences,labels,party,state,venue,job,subject,speaker,barely_true,false_counts,half_true,mostly_true,pants_on_fire
        



def get_pandas_df(filename):
    num_party = 6
    num_state = 51
    num_venue = 12
    num_job = 11
    num_sub = 14
    num_speaker = 21

    data=pd.read_csv(filename)
    data_text,data_label,party,state,venue,job,subject,speaker,barely_true,false_counts,half_true,mostly_true,pants_on_fire=load_sentences(data)
    party=to_categorical(party,num_party)
    state=to_categorical(state,num_state)
    venue=to_categorical(venue,num_venue)
    job=to_categorical(job,num_job)
    subject=to_categorical(subject,num_sub)
    speaker=to_categorical(speaker,num_speaker)
    

    data_new=[data_text,party,state,venue,job,subject,speaker,barely_true,false_counts,half_true,mostly_true,pants_on_fire]
    return data_new,data_label



def main(args):
    train_data,train_label=get_pandas_df(args.train_path)
    val_data,val_label=get_pandas_df(args.val_path)
    test_data,test_label=get_pandas_df(args.test_path)
    
    met=MeanEmbeddingTransformer()
    
    train_data[0]=met.fit_transform(train_data[0])
    val_data[0]=met.transform(val_data[0])
    test_data[0]=met.transform(test_data[0])
    train=np.hstack([x for x in train_data])
    val=np.hstack([x for x in val_data])
    test=np.hstack([x for x in test_data])
    
    
    clf=Pipeline([('clf',LogisticRegression('l2',random_state=41))])
    clf.fit(train,train_label)

    train_pred=clf.predict(train)
    print(accuracy_score(train_label,train_pred))

    pred=clf.predict(val)
    print(accuracy_score(val_label,pred))

    test_pred=clf.predict(test)
    print(accuracy_score(test_label,test_pred))
    
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--embed_path',type=str,default=None,help='path to 100d file')
    parser.add_argument('--train_path',type=str,default=None,help='path to train data file')
    parser.add_argument('--val_path',type=str,default=None,help='path to val data file')
    parser.add_argument('--test_path',type=str,default=None,help='path to test data file')
    args=parser.parse_args()
    main(args)




