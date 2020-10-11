            /**
 * Created by PRANJAL VYAS on 5/29/2020.
 */
const express = require('express');
const exphbs = require ('express-handlebars');
const app = express() ;
const body_parser = require('body-parser');
const cors = require('cors')
const mongoose = require('mongoose');
const spawn = require('child_process').spawn;
const fs = require('fs');
const socket = require('socket.io');
const passport = require('passport');
const session = require('express-session');
//const dontenv = require('dotenv');


require('./Models/db');
    const user = mongoose.model('User');

require('./passport')(passport);

var params = [] ;
var params_rcvd = false ;
var User ;
dataString = '' ;
var NeuralNet_history = {} ;


//static files;
app.use(body_parser.json({ limit : '150mb' }));
app.use(body_parser.urlencoded({limit : '150mb' , extended : false}));
app.use(express.json({limit : '150mb', extended : true, parameterLimit: 50000 }));
app.use(express.urlencoded({limit : '150mb', extended : true, parameterLimit: 50000 }));
app.use(cors());
app.engine('.hbs', exphbs({ defaultLayout : 'login', extname : '.hbs' })); app.set('view engine', '.hbs');


app.use(express.static(__dirname + '/frontend'));

    app.use(session({
        secret: 'keyboard cat',
        resave: false,
        saveUninitialized: false
    }));

app.use(passport.initialize());
app.use(passport.session());

    app.all('/*', function(req, res, next) {
        res.header("Access-Control-Allow-Origin", "*");
        next();
    });


app.use('/login',require('./routes/index'));
app.use('/auth',require('./routes/auth'));

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
    User = req.user ;
    console.log(req.user) ;

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
    User = req.query.user ;
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

app.post('/history',function(req, res){
    User_Name = req.body.body.user ;
    //console.log(User_Name);
    user.findOne({ UserName : User_Name },function(err, docs){
        //NeuralNet_history = docs.NeuralNet;
      //  console.log(docs);
       // console.log(docs.NeuralNet);
        for(i=0;i<docs.NeuralNet.length;++i)
        {
            NeuralNet_history['file' + (i+1)] = docs.NeuralNet[i] ;
        }
        //console.log(NeuralNet_history);
        res.json(NeuralNet_history) ;

    })
});

app.post('/deleterec',function(req,res){
   console.log(req.body.file_id);
   file_name = req.body.file ;
   date = req.body.date ;
   file_id = req.body.id ;
   //file_id = String(file_id);
   console.log(typeof file_id)

   user.updateOne(
       { UserName : User_Name },
    { $pull : { NeuralNet : { fliename : file_name, id : file_id }  } },function(error , success){
           if(error)
           {
               console.log('Error:\t' + error);
           }
           else
           {
               console.log(success);
           }
       }
   )

});



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
        console.log('spawning python');
        py  = spawn('python', ['main.py']),
            py.stdout.on('data', function(data){
                dataString = '' + data.toString() ;
            });
        py.stdout.on('end'  , function(){
            console.log('python sent this',dataString);


            // results of the deep learning algorithm
            output = fs.readFileSync('result.json', 'utf8');
            var metrics = JSON.parse(output);

            output = fs.readFileSync('weights.json','utf8');
            var w_n_b = JSON.parse(output) ;
            console.log(w_n_b);
         //   console.log(metrics);
            // metrics = res.json(metrics);
            socket.emit('metrics', metrics);

            paramstodatabase(User);

            //generated python cde
            gen_code = fs.readFileSync('DL_code.py', 'utf8');
            var code = JSON.parse(JSON.stringify(gen_code));
            socket.emit('code',gen_code) ;

            //fs.unlinkSync('DL_code.py');

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

    function paramstodatabase(User)
{

    var NeuralNetworkmodel =
    {
        fliename : params[14],
        layers : params[6] ,
        neurons : params[7],
        ptype : params[1],
        test_split : params[3],
        validation_split : params[12],
        learning_rate : params[2],
        Batch_size : params[5],
        Optimizer : params[4],
        dropouts : params[13],
        framework : params[0],
        date : Date.now()

    };
    user.updateOne(
        { UserName : User },
        {$push : { NeuralNet : NeuralNetworkmodel }},
        function(error , success){
            if(error)
            {
                console.log(error) ;
            }
            else
            {
                console.log(success);
            }
        }
    )
}




