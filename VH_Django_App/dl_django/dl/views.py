from django.shortcuts import render

from django.http import HttpResponse
import os
import math

import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, XLNetTokenizer, XLNetModel, XLNetLMHeadModel, XLNetConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import xlrd
import pandas as pd
import logging
import pickle
import random
import re
import threading
import warnings
from datetime import datetime
import json
import nltk
import numpy as np

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import json
import os
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
import tensorflow as tf
from google.cloud import bigquery
from google.oauth2 import service_account
from pandas.io import gbq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from firebase import firebase  
from datetime import datetime

#Preprocessing Functions

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have",
              "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
              "hasn't": "has not", "haven't": "have not","he'd": "he would","he'll": "he will", "he's": "he is",
              "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is","I'd": "I would", 
              "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", 
              "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", 
              "i've": "i have", "isn't": "is not", "it'd": "it would","it'd've": "it would have", "it'll": "it will", 
              "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
              "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", 
              "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
              "o'clock": "of the clock","oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", 
              "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", 
              "she'll": "she will", "she'll've": "she will have", "she's": "she is","should've": "should have", 
              "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
              "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", 
              "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is",
              "they'd": "they would", "they'd've": "they would have","they'll": "they will", "they'll've": "they will have",
              "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
              "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
              "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", 
              "what're": "what are","what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
              "where'd": "where did", "where's": "where is","where've": "where have", "who'll": "who will",
              "who'll've": "who will have", "who's": "who is", "who've": "who have","why's": "why is", "why've": "why have",
              "will've": "will have", "won't": "will not", "won't've": "will not have","would've": "would have", 
              "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all","y'all'd": "you all would",
              "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would",
              "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have","you're": "you are",
              "you've": "you have"}

def remove_trainingHash(txt):
    split = True
    while (split):
        x = txt.rsplit("#",1)
        try:
            if len(x[1].split(' '))>1:
                split = False
            else:
                txt=x[0].strip()
        except IndexError:
            split = False
    return(txt)

def Hashtext_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all handles
    2. Remove all links
    3. Remove all punctuation and special char
    4. Remove all stopwords
    5. Ensure words are unique
    6. Return the lemmatized cleaned text as a list of words
    '''
    
    #Remove Links
    cleanLink = re.sub(r'(?i)\b((?:https?://|www|pic\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}     /)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', text)

    #Apply Contractions
    cleanContractions = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in cleanLink.split(" ")])    
        
    #Removing handels, numbers and all special chars except #
    handleClean_text = re.sub("(@[A-Za-z0-9]+)|\d+|[^ #a-zA-Z0-9]","",cleanContractions.lower())
    #clean_text = re.sub("(#[A-Za-z0-9]+)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|\d+"," ",cleanContractions.lower())

    #Remove trailing hashtags
    hashClean_text = remove_trainingHash(handleClean_text.strip())
    
    #Remove special characters
    clean_text = re.sub("([^0-9A-Za-z \t])","",hashClean_text)
    
    return clean_text



# Connect to BigQuery
def read_bq(sessionID):
    credentials = service_account.Credentials.from_service_account_file('bigQuery-SA.json')
    project_id = 'w210-267619'
    client = bigquery.Client(credentials= credentials,project=project_id)
    
    query_survey = """SELECT * FROM `w210-267619.jonas_data.flowxo_chat` where UserID =@sessionID"""

    query_config = {
        'query': {
            'parameterMode': 'NAMED',
            'queryParameters': [
                {
                    'name': 'sessionID',
                    'parameterType': {'type': 'STRING'},
                    'parameterValue': {'value': sessionID}
                }
            
            ]
        }
    }
    
    results_df = gbq.read_gbq(query_survey, project_id=project_id, credentials=credentials, dialect = 'standard', configuration= query_config)
    surveydata = { 
                     'sessionID': sessionID,
                     'survey_type':results_df['Survey_Type'].iloc[0],
                     'Q1_Vaccine_Obstacles': results_df[results_df['Question']=='Q1 Vaccine Obstacles']['Message'].iloc[0],
                     'q1_len': len(results_df[results_df['Question']=='Q1 Vaccine Obstacles']['Message'].iloc[0]),
                     'Q2_Vaccine_Safety_Concerns': results_df[results_df['Question']=='Q2 Vaccine Safety Concerns']['Message'].iloc[0],
                     'q2_len':len(results_df[results_df['Question']=='Q2 Vaccine Safety Concerns']['Message'].iloc[0]),
                     'Q3_Vaccines_Mandatory': results_df[results_df['Question']=='Q3 Vaccines Mandatory']['Message'].iloc[0],
                     'q3_len':len(results_df[results_df['Question']=='Q3 Vaccines Mandatory']['Message'].iloc[0]),
                     'Q4_Govt_PH_Vaccines': results_df[results_df['Question']=='Q4 Govt/PH Vaccines']['Message'].iloc[0],
                     'q4_len': len(results_df[results_df['Question']=='Q4 Govt/PH Vaccines']['Message'].iloc[0]),
                     'Q5_Vaccine_Benefits': results_df[results_df['Question']=='Q5 Vaccine Benefits']['Message'].iloc[0],
                     'q5_len': len(results_df[results_df['Question']=='Q5 Vaccine Benefits']['Message'].iloc[0]),
                     'Patient_ID': results_df['PatientID'].iloc[0]
                     }
    return(surveydata)


    
#Insert to Firebase

def write_fb(surveydata, label_dict):
    from firebase import firebase  
    firebase = firebase.FirebaseApplication('https://w210-267619.firebaseio.com/', None)  
    data = label_dict
    data.update(surveydata)
    print(data)
    result = firebase.post('/survey_response_labels',data)  
    print(result)  
    

class XLNetForMultiLabelSequenceClassification(torch.nn.Module):
  
  def __init__(self, num_labels=2):
    super(XLNetForMultiLabelSequenceClassification, self).__init__()
    self.num_labels = num_labels
    self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
    self.classifier = torch.nn.Linear(768, num_labels)

    torch.nn.init.xavier_normal_(self.classifier.weight)

  def forward(self, input_ids, token_type_ids=None,\
              attention_mask=None, labels=None):
    # last hidden layer
    last_hidden_state = self.xlnet(input_ids=input_ids,\
                                   attention_mask=attention_mask,\
                                   token_type_ids=token_type_ids)
    # pool the outputs into a mean vector
    mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
    logits = self.classifier(mean_last_hidden_state)
        
    if labels is not None:
      loss_fct = BCEWithLogitsLoss()
      loss = loss_fct(logits.view(-1, self.num_labels),\
                      labels.view(-1, self.num_labels))
      return loss
    else:
      return logits
    
  def freeze_xlnet_decoder(self):
    """
    Freeze XLNet weight parameters. They will not be updated during training.
    """
    for param in self.xlnet.parameters():
      param.requires_grad = False
    
  def unfreeze_xlnet_decoder(self):
    """
    Unfreeze XLNet weight parameters. They will be updated during training.
    """
    for param in self.xlnet.parameters():
      param.requires_grad = True
    
  def pool_hidden_state(self, last_hidden_state):
    """
    Pool the output vectors into a single mean vector 
    """
    last_hidden_state = last_hidden_state[0]
    mean_last_hidden_state = torch.mean(last_hidden_state, 1)
    return mean_last_hidden_state

def tokenize_inputs(text_list, tokenizer, num_embeddings=512):
    """
    Tokenizes the input text input into ids. Appends the appropriate special
    characters to the end of the text to denote end of sentence. Truncate or pad
    the appropriate sequence length.
    """
    # tokenize the text, then truncate sequence to the desired length minus 2 for
    # the 2 special characters
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t)[:num_embeddings-2], text_list))
    # convert tokenized text into numeric ids for the appropriate LM
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # append special token "<s>" and </s> to end of sentence
    input_ids = [tokenizer.build_inputs_with_special_tokens(x) for x in input_ids]
    # pad sequences
    input_ids = pad_sequences(input_ids, maxlen=num_embeddings, dtype="long", truncating="post", padding="post")
    return input_ids

def create_attn_masks(input_ids):
    """
    Create attention masks to tell model whether attention should be applied to
    the input id tokens. Do not want to perform attention on padding tokens.
    """
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks

# write function that will spit out the labels given the model 
def generate_predictions_dict(model, feature_ids, attn_masks, num_labels, device="cpu", batch_size=32):
  
  pred_probs = np.array([]).reshape(0, num_labels)
  
  model.to(device)
  model.eval()
  
  X = feature_ids
  masks = attn_masks
  X = torch.tensor(X)
  masks = torch.tensor(masks, dtype=torch.long)
  X = X.to(device)
  masks = masks.to(device)
  with torch.no_grad():
      logits = model(input_ids=X, attention_mask=masks)
      logits = logits.sigmoid().detach().cpu().numpy()
      pred_probs = np.vstack([pred_probs, logits])

  result_ind = list(np.round(pred_probs)[0])

  output = {'Fear_of_Critical_Side_Effects__c': result_ind[1] ,
            'Logistic_Concerns__c': result_ind[4], 
            'Fear_of_Non_Critical_Side_Effects__c': result_ind[2], 
            'Fear_of_Toxic_Ingredients__c': result_ind[8], 
            'Holistic_or_Alternative_Medicine__c': result_ind[3], 
            'Religious_Beliefs_Preclude_Vaccinations__c': result_ind[6], 
            'Right_to_Choose__c': result_ind[7], 
            'Vaccines_are_a_Conspiracy__c': result_ind[0],
            'Vaccines_are_Ineffective_or_Unnecessary__c': result_ind[9],
            'Patient_is_Pro_Vaccination__c': result_ind[5],
            'Hesitancy_Classification__c': 0}
  
  return output
	
def eval_model_res(res1, res2):
    model_res = {'Fear_of_Critical_Side_Effects__c': max(res1['Fear_of_Critical_Side_Effects__c'],res2['Fear_of_Critical_Side_Effects__c']),
            'Logistic_Concerns__c': max(res1['Logistic_Concerns__c'],res2['Logistic_Concerns__c']),
            'Fear_of_Non_Critical_Side_Effects__c':max(res1['Fear_of_Non_Critical_Side_Effects__c'],res2['Fear_of_Non_Critical_Side_Effects__c']),
            'Fear_of_Toxic_Ingredients__c': max(res1['Fear_of_Toxic_Ingredients__c'],res2['Fear_of_Toxic_Ingredients__c']),
            'Holistic_or_Alternative_Medicine__c':max(res1['Holistic_or_Alternative_Medicine__c'],res2['Holistic_or_Alternative_Medicine__c']),
            'Religious_Beliefs_Preclude_Vaccinations__c': max(res1['Religious_Beliefs_Preclude_Vaccinations__c'],res2['Religious_Beliefs_Preclude_Vaccinations__c']),
            'Right_to_Choose__c': max(res1['Right_to_Choose__c'],res2['Right_to_Choose__c']),
            'Vaccines_are_a_Conspiracy__c':max(res1['Vaccines_are_a_Conspiracy__c'],res2['Vaccines_are_a_Conspiracy__c']),
            'Vaccines_are_Ineffective_or_Unnecessary__c': max(res1['Vaccines_are_Ineffective_or_Unnecessary__c'],res2['Vaccines_are_Ineffective_or_Unnecessary__c']),
            'Patient_is_Pro_Vaccination__c': max(res1['Patient_is_Pro_Vaccination__c'],res2['Patient_is_Pro_Vaccination__c']),
            'Hesitancy_Classification__c': max(res1['Hesitancy_Classification__c'],res2['Hesitancy_Classification__c']),
            'timestamp': str(datetime.now())}
    return model_res
   
checkpoint = torch.load("xlnet_vaccine.bin")
model_state_dict = checkpoint['state_dict']
model = XLNetForMultiLabelSequenceClassification(num_labels=model_state_dict["classifier.weight"].size()[0])
model.load_state_dict(model_state_dict)
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)


@api_view(['GET', 'POST'])
def predictXLNET(request):
    sessionID = str(request.GET.get('session'))
    print(sessionID)
    surveydata = read_bq(sessionID)
    print(surveydata)   
 
    label_cols = ['Conspiracy: Distrust of government, organizations, big pharma',
           'Fear of Critical side-effects (Autism, Brain Damage, SIDS/Death)',
           'Fear of Non-critical side-effects (Rash, Pain, Fever, GI problems, Bump on arm)',
           'Holistic or alternative medicine', 'Logistic Concerns', 'Pro-vax', 'Religious Beliefs',
           'Right to choose',
           'Toxic Ingredients, unclear origins of materials/manufacturer',
           'Vaccines ineffective/unnecessary']
    
    num_labels = len(label_cols)
    model_res = {'Fear_of_Critical_Side_Effects__c': 0,
            'Logistic_Concerns__c': 0,
            'Fear_of_Non_Critical_Side_Effects__c':0,
            'Fear_of_Toxic_Ingredients__c': 0,
            'Holistic_or_Alternative_Medicine__c':0,
            'Religious_Beliefs_Preclude_Vaccinations__c': 0,
            'Right_to_Choose__c': 0,
            'Vaccines_are_a_Conspiracy__c':0,
            'Vaccines_are_Ineffective_or_Unnecessary__c': 0,
            'Patient_is_Pro_Vaccination__c': 0,
            'Hesitancy_Classification__c': 0}

    questions = ['Q1_Vaccine_Obstacles', 'Q2_Vaccine_Safety_Concerns', 'Q3_Vaccines_Mandatory', 'Q4_Govt_PH_Vaccines','Q5_Vaccine_Benefits' ]

    for q in questions:
        testinput_ids = tokenize_inputs([Hashtext_process(surveydata[q])], tokenizer, num_embeddings=250).tolist()
        test_attention_masks = create_attn_masks(testinput_ids)
        model_res_q = generate_predictions_dict(model, testinput_ids, test_attention_masks, num_labels, device="cuda", batch_size=1)
        print(model_res_q)
        model_res = eval_model_res(model_res,model_res_q)
    
    write_fb(surveydata, model_res)
    return JsonResponse(model_res, safe=False)


def index(request):
    return HttpResponse(
        'Hello, World. This is Django running on Google App Engine')
