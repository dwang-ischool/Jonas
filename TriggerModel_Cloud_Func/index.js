'use strict';


const functions = require('firebase-functions');
const admin = require('firebase-admin');
admin.initializeApp(functions.config().firebase);


exports.triggerModel = functions.database.ref('/flowxo_convos/{convoID}').onCreate((snapshot, context) => {
  console.log('start cloud function');
  const session = snapshot.val();
  console.log(session.userID);
  const request = require('request-promise');
  request('http://34.83.42.246:8000/predictXLNET?session='+session.userID, function (error, response, body) {
    if (!error && response.statusCode == 200) {
        console.log(response)
        console.log(body) // Print the google web page.
     }
         
     
})
 
  })







