# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 18:08:59 2019

@author: Rahul Verma
"""

import torch
from torch import nn
import numpy as np
from utils import *

class TextCNN(nn.Module):
    def __init__(self, config, vocab_size, word_embeddings):
        super(TextCNN, self).__init__()
        self.config = config
        
        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=True)
        
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.embed_size, out_channels=128, kernel_size=self.config.kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[0]+1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.embed_size, out_channels=128, kernel_size=self.config.kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[1]+1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.embed_size, out_channels=128, kernel_size=self.config.kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size[2]+1)
        )
        
        self.dropout = nn.Dropout(self.config.dropout_keep)
        
        # Fully-Connected Layer
        self.fc0 = nn.Linear(self.config.num_channels*len(self.config.kernel_size), 128)
        self.relu=nn.ReLU()
        
        self.d1=nn.Linear(self.config.meta_in,64)
        self.d2=nn.Linear(64,64)
        self.fc_final=nn.Linear(64+128,self.config.output_size)
        
    
        
    def forward(self,x,x_meta):
        #x.shape = (max_sen_len, batch_size)
        embedded_sent = self.embeddings(x).permute(1,2,0)
        #embedded_sent.shape = (batch_size=64,embed_size=300,max_sen_len=20)
        
        conv_out1 = self.conv1(embedded_sent).squeeze(2) #shape=(64, num_channels, 1) (squeeze 1)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)
        
        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        text_feature_map = self.dropout(all_out)
        text_out = self.relu(self.fc0(text_feature_map))
        
        meta_out=self.relu(self.d1(x_meta))
        meta_out=self.relu(self.d2(meta_out))
        concat=torch.cat((text_out,meta_out),1)
        
        final=self.fc_final(concat)
        return final
    
    def add_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def add_loss_op(self, loss_op):
        self.loss_op = loss_op
    
    def cosine_rampdown(self,current, rampdown_length):
        assert 0 <= current <= rampdown_length
        return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
    
    def reduce_lr_new(self,epoch):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            #print(g['lr'])
            g['lr'] = self.cosine_rampdown(epoch,50)*g['lr'] 
            print("LR-->"+str(g['lr']))        
    
    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2
    
    def to_one_hot(self,x,num_class):
        
        assert torch.max(x)<num_class
        return nn.functional.one_hot(x,num_class)
        
    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
        num_party = 6
        num_state = 51
        num_venue = 12
        num_job = 11
        num_sub = 14
        num_speaker = 21
        
        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config.max_epochs/3)) or (epoch == int(2*self.config.max_epochs/3)):
            self.reduce_lr()
            
            
        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            party=self.to_one_hot(batch.party,num_party)
            state=self.to_one_hot(batch.state,num_state)
            job=self.to_one_hot(batch.job,num_job)
            venue=self.to_one_hot(batch.venue,num_venue)
            subject=self.to_one_hot(batch.subject,num_sub)
            speaker=self.to_one_hot(batch.speaker,num_speaker)
            bt=batch.bt.unsqueeze(1)
            fc=batch.fc.unsqueeze(1)
            ht=batch.ht.unsqueeze(1)
            mt=batch.mt.unsqueeze(1)
            pof=batch.pof.unsqueeze(1)
            if torch.cuda.is_available():
                x = batch.text.cuda()
                x_meta=torch.cat([party,state,job,venue,subject,speaker,bt,fc,ht,mt,pof],dim=1).cuda().float()
                y = (batch.label).type(torch.cuda.LongTensor)
              
            else:
                x = batch.text
                x_meta=torch.cat([party,state,job,venue,subject,speaker,bt,fc,ht,mt,pof],dim=1).float()
                y = (batch.label).type(torch.LongTensor)
                
            y_pred = self.__call__(x,x_meta)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
    
            if i % 100 == 0:
                print("Iter: {}".format(i+1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []
                self.train()
                
        return train_losses, val_accuracies