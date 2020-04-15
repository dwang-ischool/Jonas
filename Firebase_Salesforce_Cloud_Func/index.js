// this version of the cloud function integrates model results from the model and formated survey bot responses to Salesforce
// The integration is between survey_response_labels table in Firebase and Vaccination_Sentiment__c object in Salesforce

'use strict';

// initialize firebase DB connection and jsforce
const functions = require('firebase-functions');
const admin = require('firebase-admin');
admin.initializeApp(functions.config().firebase);
const google = require('googleapis');
const jsforce = require('jsforce');

//credientials to connect to Salesforce.com
const SFDC_URL = "https://jonass-dev-ed.my.salesforce.com/";
const SFDC_LOGIN = "jonas@capstone.berkeley.edu";
const SFDC_PASSWORD = "w2102020";
const SFDC_TOKEN = "3ZI8Yv43qfzU8ZnVrvAWNjac";
console.log('initialization complete');


/*
 * updateVaccinationSentimentSFDC
 * Create new SFDC record for Vaccination_Sentiment__c object
 */
exports.updateVaccinationSentimentSFDC = functions.database.ref('/survey_response_labels/{survey_response_labelsID}').onCreate((snapshot, context) => {
  console.log('start cloud function');
  const key = snapshot.key;
  const vs = snapshot.val();

  var jsforce = require('jsforce');
  var conn = new jsforce.Connection();

  //[START conn_sfdc]
  conn = new jsforce.Connection({
    loginUrl : SFDC_URL
  });
  conn.login(SFDC_LOGIN, SFDC_PASSWORD + SFDC_TOKEN, function(err, res) {
  //[END conn_sfdc]
    if (err) {
      return console.error('SFDC ERROR', err);
    }

    console.log("started creating sfdc record");
    //[START create vaccine sentiment record]
    conn.sobject("Vaccination_Sentiment__c").create({
      Fear_of_Critical_Side_Effects__c: vs.Fear_of_Critical_Side_Effects__c,
      Logistic_Concerns__c: vs.Logistic_Concerns__c,
      Fear_of_Non_Critical_Side_Effects__c: vs.Fear_of_Non_Critical_Side_Effects__c,
      Fear_of_Toxic_Ingredients__c: vs.Fear_of_Toxic_Ingredients__c,
      Holistic_or_Alternative_Medicine__c: vs.Holistic_or_Alternative_Medicine__c,
      Patient_ID__c: vs.Patient_ID,
      Patient_is_Pro_Vaccination__c: vs.Patient_is_Pro_Vaccination__c,
      Question_1_Response__c: vs.Q5_Vaccine_Benefits,
      Question_2_Response__c: vs.Q2_Vaccine_Safety_Concerns,
      Question_3_Response__c: vs.Q3_Vaccines_Mandatory,
      Question_4_Response__c: vs.Q4_Govt_PH_Vaccines,
      Question_5_Response__c: vs.Q1_Vaccine_Obstacles,
      Question_1_Response_Length__c: vs.q5_len, 
      Question_2_Response_Length__c: vs.q2_len, 
      Question_3_Response_Length__c: vs.q3_len, 
      Question_4_Response_Length__c: vs.q4_len, 
      Question_5_Response_Length__c: vs.q1_len, 
      Response_for_Self_or_As_Parent_Guardian__c: vs.survey_type,
      Religious_Beliefs_Preclude_Vaccinations__c: vs.Religious_Beliefs_Preclude_Vaccinations__c,
      Right_to_Choose__c: vs.Right_to_Choose__c,
      Vaccines_are_a_Conspiracy__c: vs.Vaccines_are_a_Conspiracy__c,
      Vaccines_are_Ineffective_or_Unnecessary__c: vs.Vaccines_are_Ineffective_or_Unnecessary__c, 
      Session_ID__c: vs.sessionID,
      Testing_Integration_Description__c: vs.description 
    }, function(err, ret) {
    //[END create vaccine sentiment record]
      if (err || !ret.success) {
        return console.error(err, ret);
      }
      console.log("finished creating sfdc record");
      admin.database().ref(`/survey_response_labels/${key}/sfdc_key`).set(ret.id);
    });
  });
});


