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

var User_info = new mongoose.Schema({
   UserName : {type : String,
                required : true
   },

    googleId : {type : String,
        unique : true
    },

    firstname : {type : String,
        required : true
    },

    lastname : {type : String,
        required : true
    },

    image : {type : String,
        required : true
    },


    NeuralNet : { type : [NN_Schema]}
});


//mongoose.model('Neural_Model',NN_Schema);
mongoose.model('User',User_info);