/**
 * Created by pranjal on 11-09-2020.
 */

const mongoose = require('mongoose');

mongoose.connect('mongodb://localhost:27017/DLmodel_Database',{
    useNewUrlParser : true,
    useUnifiedTopology : true
},function(err){
    if(!err){
        console.log('connected to the AI params database');
    }
    else
    {
        console.log('Error connecting to database');
    }
});

require('./DLparams.model');