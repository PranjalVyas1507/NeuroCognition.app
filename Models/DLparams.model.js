/**
 * Created by pranjal on 15-09-2020.
 */
const mongoose = require('mongoose');

var NN_Schema = new mongoose.Schema({
    model_id : {type : String},
    fliename : {type : String},
    ptype : {type : String},
    layers : { type : Number},
    neurons : { type :Array},
    weights_biases : {type : Array}

});

mongoose.model('Neural_Model',NN_Schema);