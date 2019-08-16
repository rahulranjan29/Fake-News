# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 21:46:06 2019

@author: Rahul Verma
"""

import pandas as pd

def change_label(df):
    label_dict={'pants-fire':0,'false':1,'barely-true':2,'half-true':3,'mostly-true':4,'true':5}
    b_label_dict={'pants-fire':0,'false':0,'barely-true':0,'half-true':1,'mostly-true':1,'true':1}
    df['multi_label']=df['label'].apply(lambda x:label_dict[x])
    df['binary_label']=df['label'].apply(lambda x:b_label_dict[x])
    return df




def vect_speakers(df):
    keys = ['barack-obama', 'donald-trump', 'hillary-clinton', 'mitt-romney', 
            'scott-walker', 'john-mccain', 'rick-perry', 'chain-email', 
            'marco-rubio', 'rick-scott', 'ted-cruz', 'bernie-s', 'chris-christie', 
            'facebook-posts', 'charlie-crist', 'newt-gingrich', 'jeb-bush', 
            'joe-biden', 'blog-posting','paul-ryan']
    
    def map_speaker(speaker):
        if isinstance(speaker,str):
            speaker=speaker.lower()
            num_speakers=[s for s in keys if s in speaker]
            if len(num_speakers)>0:
                return keys.index(num_speakers[0])
            else:
                return len(keys)
        else:
            return len(keys)
        
    df['speaker_id']=df['speaker'].apply(map_speaker)
    return df
    
def vect_subjects(df):
    subject_list = ['health','tax','immigration','election','education',
'candidates-biography','economy','gun','jobs','federal-budget','energy','abortion','foreign-policy']
    subject_dict = {'health':0,'tax':1,'immigration':2,'election':3,'education':4,
'candidates-biography':5,'economy':6,'gun':7,'jobs':8,'federal-budget':9,'energy':10,'abortion':11,'foreign-policy':12}
    
    def map_subject(subject):
        if isinstance(subject, str):
            subject = subject.lower()
            matches = [s for s in subject_list if s in subject]
            if len(matches) > 0:
                return subject_dict[matches[0]] #Return index of first match
            else:
                return 13 #This maps any other subject to index 13
        else:
            return 13    
    
    df['subject_id']=df['subjects'].apply(map_subject)
    return df

def vect_job(df):
    job_list = ['president', 'u.s. senator', 'governor', 'president-elect', 'presidential candidate', 
            'u.s. representative', 'state senator', 'attorney', 'state representative', 'congress']
    job_dict = {'president':0, 'u.s. senator':1, 'governor':2, 'president-elect':3, 'presidential candidate':4, 
            'u.s. representative':5, 'state senator':6, 'attorney':7, 'state representative':8, 'congress':9}
    
    def map_job(job):
        if isinstance(job, str):
            job = job.lower()
            matches = [s for s in job_list if s in job]
            if len(matches) > 0:
                return job_dict[matches[0]] #Return index of first match
            else:
                return 10 #This maps any other job to index 10
        else:
            return 10 #Nans or un-string data goes here.
    df['job_id'] = df['job'].apply(map_job)
    return df

    
    
def vect_party(df):
    party_dict = {'republican':0,'democrat':1,'none':2,'organization':3,'newsmaker':4}
    def map_party(party):
        if party in party_dict:
            return party_dict[party]
        else:
            return 5
    df['party_id'] = df['party'].apply(map_party)
    return df


def vect_states(df):
    states = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado',
         'Connecticut','Delaware','Florida','Georgia','Hawaii','Idaho', 
         'Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana',
         'Maine' 'Maryland','Massachusetts','Michigan','Minnesota',
         'Mississippi', 'Missouri','Montana','Nebraska','Nevada',
         'New Hampshire','New Jersey','New Mexico','New York',
         'North Carolina','North Dakota','Ohio',    
         'Oklahoma','Oregon','Pennsylvania','Rhode Island',
         'South  Carolina','South Dakota','Tennessee','Texas','Utah',
         'Vermont','Virginia','Washington','West Virginia',
         'Wisconsin','Wyoming']
    states_dict = {'wyoming': 48, 'colorado': 5, 'washington': 45, 'hawaii': 10, 'tennessee': 40, 'wisconsin': 47, 'nevada': 26, 'north dakota': 32,\
                   'mississippi': 22, 'south dakota': 39, 'new jersey': 28, 'oklahoma': 34, 'delaware': 7, 'minnesota': 21, 'north carolina': 31,\
                   'illinois': 12, 'new york': 30, 'arkansas': 3, 'west virginia': 46, 'indiana': 13, 'louisiana': 17, 'idaho': 11,\
                   'south  carolina': 38, 'arizona': 2, 'iowa': 14, 'mainemaryland': 18, 'michigan': 20, 'kansas': 15, 'utah': 42,\
                   'virginia': 44, 'oregon': 35, 'connecticut': 6, 'montana': 24, 'california': 4, 'massachusetts': 19, 'rhode island': 37,\
                   'vermont': 43, 'georgia': 9, 'pennsylvania': 36, 'florida': 8, 'alaska': 1, 'kentucky': 16, 'nebraska': 25, \
                   'new hampshire': 27, 'texas': 41, 'missouri': 23, 'ohio': 33, 'alabama': 0, 'new mexico': 29}
    def map_state(state):
        if isinstance(state, str):
            state = state.lower()
            if state in states_dict:
                return states_dict[state]
            else:
                if 'washington' in state:
                    return states_dict['washington']
                else:
                    return 50 #This maps any other location to index 50
        else:
            return 50 
        
    df['state_id'] = df['state'].apply(map_state) 
    return df


def vect_subject(df):
    subject_list = ['health','tax','immigration','election','education',\
                    'candidates-biography','economy','gun','jobs','federal-budget','energy','abortion','foreign-policy']
    subject_dict = {'health':0,'tax':1,'immigration':2,'election':3,'education':4,\
                'candidates-biography':5,'economy':6,'gun':7,'jobs':8,'federal-budget':9,'energy':10,'abortion':11,'foreign-policy':12}

    def map_subject(subject):
        if isinstance(subject, str):
            subject = subject.lower()
            matches = [s for s in subject_list if s in subject]
            if len(matches) > 0:
                return subject_dict[matches[0]] #Return index of first match
            else:
                return 13 #This maps any other subject to index 13
        else:
            return 13 #Nans or un-string data goes here.
    df['subject_id'] = df['subjects'].apply(map_subject)
    return df



def vect_venue(df):
    venue_list = ['news release','interview','tv','radio',
              'campaign','news conference','press conference','press release',
              'tweet','facebook','email']
    venue_dict = {'news release':0,'interview':1,'tv':2,'radio':3,
              'campaign':4,'news conference':5,'press conference':6,'press release':7,
              'tweet':8,'facebook':9,'email':10}
    def map_venue(venue):
        if isinstance(venue, str):
            venue = venue.lower()
            matches = [s for s in venue_list if s in venue]
            if len(matches) > 0:
                return venue_dict[matches[0]] #Return index of first match
            else:
                return 11 #This maps any other venue to index 11
        else:
            return 11 #Nans or un-string data goes here.
    df['venue_id'] = df['venue'].apply(map_venue)
    return df


column_names=['id','label','statement','subjects','speaker','job','state','party','barely true counts','false counts','half true counts','mostly true counts','pants on fire counts','venue','justification']
train_data=pd.read_csv('C:\\Users\\Rahul Verma\\Desktop\\liar_plus\\dataset\\train2.tsv',sep='\t',names=column_names).drop(['id'],'columns')
val_data=pd.read_csv('C:\\Users\\Rahul Verma\\Desktop\\liar_plus\\dataset\\val2.tsv',sep='\t',names=column_names).drop(['id'],'columns')
test_data=pd.read_csv('C:\\Users\\Rahul Verma\\Desktop\\liar_plus\\dataset\\test2.tsv',sep='\t',names=column_names).drop(['id'],'columns')



functions=[change_label,vect_speakers,vect_venue,vect_subject,vect_states,vect_party,vect_job]
dfs=[train_data,val_data,test_data]

for i,data in enumerate(dfs):
    
    for f in functions:
        d=f(data)
        temp=d
        data=d
  

train_data=train_data.drop([2142,9375],0)

train_data.to_csv('train_data_n.csv')
val_data.to_csv('val_data_n.csv')
test_data.to_csv('test_data_n.csv')

    
