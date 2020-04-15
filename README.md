# Jonas_Capstone_Spring_2020
Capstone for UC Berkeley MIDS Spring 2020 - Aditi Hegde, Vivian Lu, TK Truong, Dili Wang

## I. Modeling 
* XLNet: See XLNetModeling 
    * Useful references: 
        * https://mlexplained.com/2019/06/30/paper-dissected-xlnet-generalized-autoregressive-pretraining-for-language-understanding-explained/
        * https://mccormickml.com/2019/09/19/XLNet-fine-tuning/
        * Training code heavily drawn from: https://towardsdatascience.com/multi-label-text-classification-with-xlnet-b5f5755302df
    * Training and validation notebook: Vaccine_XLNet_0402.ipynb 
    * Code for using model to do single prediction: singleprediction_test.ipynb
    * Prepping data for training: prepdata_0402
    * Error analysis: 0402_ExtraErrorAnalysis.ipynb

## II. FlowXO 
* The survey bot interface was built with Flow XO (https://flowxo.com/)
* We connected the Flow XO interface to our data storage (BigQuery and Firebase Real-time Database) with a Firebase Cloud Function as the webhook
    * This recorded user responses for each question
    * See FlowXO_Cloud_Func folder for the index.js and package.json

## III. Django 
* Django Application is hosted on the compute engine (GCP).
* The model is triggered using cloud function - TriggerModel, once the interaction between user and BOT have ended.
* Useful References to set up Django : 
    * https://dzone.com/articles/best-python-django-tutorial-for-beginners-with-pro

## IV. Salesforce
