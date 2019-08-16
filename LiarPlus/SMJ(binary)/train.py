# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 18:12:12 2019

@author: Rahul Verma
"""

from utils import *
from model import *

import torch.backends.cudnn as cudnn
import numpy as np
import torch.optim as optim
from torch import nn
import torch
import argparse

if __name__=='__main__':
    seed=5
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--embed_path',type=str,default=None,help='path to 100d file')
    parser.add_argument('--train_path',type=str,default=None,help='path to train data file')
    parser.add_argument('--val_path',type=str,default=None,help='path to val data file')
    parser.add_argument('--test_path',type=str,default=None,help='path to test data file')
    config=parser.parse_args()
    config.embed_size=300
    config.num_channels=128
    config.lr=0.025
    config.kernel_size=[2,5,8]
    config.dropout_keep=0.7
    config.meta_in=120
    config.batch_size=48
    config.max_sen_len=25
    config.output_size=2
    config.max_epochs=25
    
    
    train_file = config.train_path ## Path to newly created train_data
    test_file = config.test_path## Path to newly created train data
    w2v_file = config.embed_path ## path to 300d glove embeddings
    val_file= config.val_path## Path to newly created val data
    dataset = Dataset(config)
    dataset.load_data(w2v_file, train_file, test_file,val_file)
    
    # Create Model with specified optimizer and loss function
    ##############################################################
    model = TextCNN(config, len(dataset.vocab), dataset.word_embeddings)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.lr,momentum=0.9,weight_decay=1e-4,nesterov=True)
    
    criterion = nn.CrossEntropyLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(criterion)
    ##############################################################
    
    train_losses = []
    val_accuracies = []
    best=0.0
    best_epoch=0
    best_test=0.0
    for i in range(config.max_epochs):
        print ("Epoch: {}".format(i))
        model.train()
        train_loss,_ = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        train_accuracy=evaluate_model(model,dataset.train_iterator)
        val_accuracy = evaluate_model(model,dataset.val_iterator)
        print("Training Accuracy :"+str(train_accuracy))
        print("Val Accuracy :"+str(val_accuracy))
        print('\n') 
        if val_accuracy>best:
            best_test=evaluate_model(model,dataset.test_iterator)
            best=val_accuracy
            best_epoch=i
            #torch.save({'epoch':i+1,
             #   'state_dict':model.state_dict(),
              #  'optimizer_dict':optimizer.state_dict()},
               # 'best_{}.pth'.format(i+1) )
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    train_acc = evaluate_model(model, dataset.train_iterator)
    val_acc = evaluate_model(model, dataset.val_iterator)
    test_acc = evaluate_model(model, dataset.test_iterator)

    
    print('Best Val Acc:{:.4f}'.format(best))
    print('Best Test Acc:{:.4f}'.format(best_test))
    
        