const functions = require('firebase-functions');
const {BigQuery} = require('@google-cloud/bigquery');

// initialise firebase DB connection
const admin = require('firebase-admin');
admin.initializeApp({
  credential: admin.credential.applicationDefault(),
  databaseURL: 'ws://your-db.firebaseio.com/'
});


exports.flowxo = functions.https.onRequest((request, response) => {
  // log request for troubleshooting
  console.log('Request headers:' + JSON.stringify(request.headers));
  console.log('Request body:' + JSON.stringify(request.body));

  var userID = request.body.userID;
  var timestamp = request.body.timestamp;
  var message = request.body.message;
  var question = request.body.question;
  var patientID = request.body.patientID;
  var survey_type = request.body.survey_type;

  // for each webhook request from Flow XO, write data to BigQuery
  const projectId = 'project ID goes here';
  const datasetId = 'dataset ID goes here';
  const tableId = "table ID goes here";
  const bigquery = new BigQuery({
      projectId: projectId
    });
   const rows = [{UserID: userID, Convo_Timestamp: timestamp, Question: question,
                Message: message, PatientID: patientID, Survey_Type: survey_type}];

   bigquery
  .dataset(datasetId)
  .table(tableId)
  .insert(rows)
  .catch(err => {
   if (err && err.name === 'PartialFailureError') {
     if (err.errors && err.errors.length > 0) {
       console.log('Insert errors:');
       err.errors.forEach(err => console.error(err));
     }
   } else {
     console.error('ERROR:', err);
   }
 });

// After the last question, log the userID to a table in Firebase
// This will trigger the model to run
if (question === "Last Question") {
  return admin.database().ref('/flowxo_convos').push({userID: userID}).then((snapshot) => {
    console.log('firebase database write sucessful: ' + snapshot.ref.toString());
    response.status(200).send();
    return null;
  });
}

  response.status(200).send();


});
