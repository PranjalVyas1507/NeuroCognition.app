    /**
 * Created by PRANJAL VYAS on 5/29/2020.
 */
const express =     require('express');
const app = express() ;
const body_parser = require('body-parser');
const mongoose = require('mongoose');
const spawn = require('child_process').spawn;
const fs = require('fs');
const socket = require('socket.io');
//const dontenv = require('dotenv');

require('./Models/db');
    const NN_schema = mongoose.model('Neural_Model');

var params = [] ;
var params_rcvd = false ;
dataString = '' ;


//static files;
app.use(body_parser.json({ limit : '150mb' }));
app.use(body_parser.urlencoded({limit : '150mb' , extended : false}));
app.use(express.json({limit : '150mb', extended : true, parameterLimit: 50000 }));
app.use(express.urlencoded({limit : '150mb', extended : true, parameterLimit: 50000 }));
//app.use()
app.use(express.static(__dirname + '/frontend'));

//database initialization:

/*    const MongoClient = require('mongodb').MongoClient;
    const uri = "mongodb+srv://PranjalV1507:<password>@neuralpark0.xlkya.mongodb.net/<dbname>?retryWrites=true&w=majority";
    const client = new MongoClient(uri, { useNewUrlParser: true });
    client.connect(err => {
        const collection = client.db("test").collection("devices");
    // perform actions on the collection object
    client.close();
    }); */

//mongoose.connect("mongodb+srv://PranjalV1507:H7NkzGYHvjQcCgq@neuralpark0.xlkya.mongodb.net/AI_params?retryWrites=true&w=majority") ;



// GET request  : Neural Network parameters
app.get('/data' , function(req,res){
    console.log("Just received a GET request");

    var framework = req.query.framework ;
    var type = req.query.type ;
    var LR =  req.query.learning_rate ;
    var Test_Split = req.query.testsplit ;
    var Valid_Split = req.query.validsplit ;
    var Optimizer = req. query.optimization ;
    var batch_size = req.query.batch_size ;
    var Dropout = req.query.dropouts ;
    var filename = req.query.filename ;
    //var tabular_data = JSON.stringify(req.body.data) ;

    console.log("Framework:\t" + framework);
    console.log("Type:\t" + type) ;
    console.log("LR:\t" + LR) ;
    console.log("Train-Val Split:\t" + Valid_Split) ;
    console.log("Optimizer:\t" + Optimizer) ;
    console.log("Batch Size:\t" + batch_size) ;
    console.log("Layers:\t" + req.query.layers );
    console.log("Neurons:\t" + req.query.neurons );
    console.log("Activation:\t" + req.query.activation );
    console.log("Droputs :t" + Dropout);
    console.log("Headers:\t" + req.query.headers );
    //console.log("Excel_data:\t", tabular_data);

    params[0] = framework ;
    params[1] = type ;
    params[2] = LR ;
    params[3] = Test_Split ;
    params[4] = Optimizer ;
    params[5] = batch_size ;
    params[6] = req.query.layers ;
    params[7] = req.query.neurons ;
    params[8] = req.query.activation ;
    params[10] = req.query.headers ;
    params[11] = req.query.target ;
    params[12] = Valid_Split ;
    params[13] = Dropout ;
    params[14] = filename ;
   // console.log('selected headers:\t' + params[10]) ;
    console.log('target:\t' + params[11]) ;

    res.send("Received Params");
    params_rcvd = true ;
    // spawing python script
    /*
    py    = spawn('python', ['main.py']),
    py.stdout.on('data', function(data){
                dataString = '' + data.toString() ;
    });
    py.stdout.on('end', function(){
        console.log('python sent this',dataString);
        output = fs.readFileSync('result.json', 'utf8');
        var metrics = JSON.parse(output);
        metrics = res.json(metrics);


    });
    py.stdin.write(JSON.stringify(params));
    py.stdin.end();
    */

});



// POST method response : for the data file
app.post('/data' , function(req,res){
    tabular_data = req.body ;
    console.log(tabular_data);
    params[9] = tabular_data ;
    //console.log(typeof params[9]);
    res.send("Received data");

}) ;


// Starting the server
var server = app.listen(3000,function(){
    console.log("Server running on port 3000") ;

});

// socket setup
var io = socket(server);

// Making a web socket connection
io.on('connection', function(socket) {
    console.log('connected with a client:\t', socket.id);

    if(params_rcvd===true)
    {
        console.log('spawning python')
        py  = spawn('python', ['main.py']),
            py.stdout.on('data', function(data){
                dataString = '' + data.toString() ;
            });
        py.stdout.on('end', function(){
            console.log('python sent this',dataString);


            // results of the deep learning algorithm
            output = fs.readFileSync('result.json', 'utf8');
            var metrics = JSON.parse(output);
            // metrics = res.json(metrics);
            socket.emit('metrics', metrics);

            paramstodatabase();

            //generated python cde
            gen_code = fs.readFileSync('DL_code.py', 'utf8');
            var code = JSON.parse(JSON.stringify(gen_code));
            socket.emit('code',gen_code) ;

            fs.unlinkSync('DL_code.py');

        });
        py.stdin.write(JSON.stringify(params));
        py.stdin.end();

    }


}) ;
    /*
    params[0] = framework ;
    params[1] = type ;
    params[2] = LR ;
    params[3] = Test_Split ;
    params[4] = Optimizer ;
    params[5] = batch_size ;
    params[6] = req.query.layers ;
    params[7] = req.query.neurons ;
    params[8] = req.query.activation ;
    params[10] = req.query.headers ;
    params[11] = req.query.target ;
    params[12] = Valid_Split ;
    params[13] = Dropout ;
    params[14] = filename ;
    */

    function paramstodatabase()
{
    var NeuralNetworkmodel = new NN_schema() ;
    NeuralNetworkmodel.fliename = params[14] ;
    NeuralNetworkmodel.layers = params[6] ;
    NeuralNetworkmodel.neurons = params[7] ;
    NeuralNetworkmodel.ptype = params[1] ;
    NeuralNetworkmodel.save();

}




