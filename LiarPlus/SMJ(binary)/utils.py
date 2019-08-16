# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 18:12:43 2019

@author: Rahul Verma
"""

import torch
from torchtext import data
from torchtext.vocab import Vectors
import spacy
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def to_categorical(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def normalize(list_v,max_value):
    newList = [x / max_value for x in list_v]
    return newList
    
class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}
    
    
   
    def load_sentences(self,data):
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
        for i in range(len(data)):
            sent,label,justification=data['statement'][i],data['binary_label'][i],data['justification'][i]
            if justification is not np.nan:
                comb=sent+justification
            else:
                comb=sent
            party.append(data['party_id'][i]),state.append(data['state_id'][i]),venue.append(data['venue_id'][i]),job.append(data['job_id'][i])
            subject.append(data['subject_id'][i]),speaker.append(data['speaker_id'][i])
            barely_true.append(data['barely true counts'][i])
            false_counts.append(data['false counts'][i])
            half_true.append(data['half true counts'][i])
            mostly_true.append(data['mostly true counts'][i])
            pants_on_fire.append(data['pants on fire counts'][i])
            sentences.append(comb)
            labels.append(label)
       
        
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
        
        return sentences,labels,party,state,venue,job,subject,speaker,barely_true,false_counts,half_true,mostly_true,pants_on_fire
    
    def get_pandas_df(self, filename):
        '''
        Load the data into Pandas.DataFrame object
        This will be used to convert data to torchtext object
         '''
        data=pd.read_csv(filename)
        data_text,data_label,party,state,venue,job,subject,speaker,barely_true,false_counts,half_true,mostly_true,pants_on_fire=self.load_sentences(data)
        
        full_df = pd.DataFrame({"text":data_text, "label":data_label,"party":party,"state":state,
                                "venue":venue,"job":job,"subject":subject,"speaker":speaker,
                                'bt':barely_true,'fc':false_counts,'ht':half_true,'mt':mostly_true,'pof':pants_on_fire})
        return full_df
    
    def load_data(self, w2v_file, train_file, test_file, val_file=None):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data
        
        Inputs:
            w2v_file (String): absolute path to file containing word embeddings (GloVe/Word2Vec)
            train_file (String): absolute path to training file
            test_file (String): absolute path to test file
            val_file (String): absolute path to validation file
        '''

        NLP = spacy.load('en')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
        
        # Creating Field for data
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True,fix_length=self.config.max_sen_len)
        LABEL = data.Field(sequential=False, use_vocab=False)
        PARTY=data.Field(sequential=False,use_vocab=False)
        STATE=data.Field(sequential=False,use_vocab=False)
        JOB=data.Field(sequential=False,use_vocab=False)
        VENUE=data.Field(sequential=False,use_vocab=False)
        SUBJECT=data.Field(sequential=False,use_vocab=False)
        SPEAKER=data.Field(sequential=False,use_vocab=False)
        BT=data.Field(sequential=False,use_vocab=False)
        FC=data.Field(sequential=False,use_vocab=False)
        HT=data.Field(sequential=False,use_vocab=False)
        MT=data.Field(sequential=False,use_vocab=False)
        POF=data.Field(sequential=False,use_vocab=False)
        datafields = [("text",TEXT),("label",LABEL),("party",PARTY),("state",STATE),\
                      ("venue",VENUE),("job",JOB),("subject",SUBJECT),("speaker",SPEAKER),\
                      ('bt',BT),('fc',FC),('ht',HT),('mt',MT),('pof',POF)]
        
        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file)
        print((train_df.values.tolist()[0])) 
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        
        train_data = data.Dataset(train_examples, datafields)
        
        
        test_df = self.get_pandas_df(test_file)
        test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, datafields)
        
        # If validation file exists, load it. Otherwise get validation data from training data
        if val_file:
            val_df = self.get_pandas_df(val_file)
            val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_examples, datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.8)
        
        TEXT.build_vocab(train_data, vectors=Vectors(w2v_file))
        self.word_embeddings = TEXT.vocab.vectors
        self.vocab = TEXT.vocab
        
        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)
        
        self.val_iterator, self.test_iterator = data.BucketIterator.splits(
            (val_data, test_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)
        
        print ("Loaded {} training examples".format(len(train_data)))
        print ("Loaded {} test examples".format(len(test_data)))
        print ("Loaded {} validation examples".format(len(val_data)))

import torch 
import torch.nn as nn
def to_one_hot(x,num_class):

    assert torch.max(x)<num_class
    return nn.functional.one_hot(x,num_class)
def evaluate_model(model, iterator):
    model.eval()
    all_preds = []
    all_y = []
    num_party = 6
    num_state = 51
    num_venue = 12
    num_job = 11
    num_sub = 14
    num_speaker = 21
    
    for idx,batch in enumerate(iterator):
        party=to_one_hot(batch.party,num_party)
        state=to_one_hot(batch.state,num_state)
        job=to_one_hot(batch.job,num_job)
        venue=to_one_hot(batch.venue,num_venue)
        subject=to_one_hot(batch.subject,num_sub)
        speaker=to_one_hot(batch.speaker,num_speaker)
        bt=batch.bt.unsqueeze(1)
        fc=batch.fc.unsqueeze(1)
        ht=batch.ht.unsqueeze(1)
        mt=batch.mt.unsqueeze(1)
        pof=batch.pof.unsqueeze(1)
        if torch.cuda.is_available():
            x = batch.text.cuda()
            x_meta=torch.cat([party,state,job,venue,subject,speaker,bt,fc,ht,mt,pof],dim=1).cuda().float()
        else:
            x = batch.text
            x_meta=torch.cat([party,state,job,venue,subject,speaker,bt,fc,ht,mt,pof],dim=1).float()
            
            
        y_pred = model(x,x_meta)
        predicted = torch.max(y_pred.cpu().data, 1)[1] 
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score