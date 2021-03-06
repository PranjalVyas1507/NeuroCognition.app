// Code goes here

var default_neurons = [3, 5, 2] ;
var default_layers = 3;
var tabular_data ;

// Intial co-ordinates for 2D Rendering
var init_x = 80 , init_y = 20 ;

//Variables used in 2D rendering
//var grad_x, grad_y ;
var abssica = [];              // x co-ordinate for neurons
var ordinate = [];             // y co-ordinate lists for neurons
var layer_index = [];
var total_neurons = 0 ;
var neurons_list = default_neurons ;
var activation_list = ['relu', 'relu', 'relu', 'relu', 'relu'] ;
var final_headers = [] ;
var target ;
var droputs = [0.1, 0.1, 0.1, 0.1];
var socket ;
var login_counter = 0 ;
var toggle_nav = true ;

    angular.module('webapp', ['ngMaterial', 'ngCookies'])
    .controller('AppCtrl', function($scope, $http, $mdDialog, $cookies, $mdSidenav) {

        //List of Options for the NN Framework
        $scope.strlist_frmwrk = ['Keras', 'PyTorch'];
        //$scope.strlist_ptype = ['Classification' , 'Sequence Model' , 'Generative', 'Segmentation', 'Mapping'];
        $scope.strlist_ptype = ['Classification' , 'Time Series'] ;
        $scope.strlist_optype = ['SGD' , 'Adam', 'Adagrad','RMSProp', 'Adamax'];
        $scope.fltlist_layers = ['1','2','3','4','5','6'] ;

        $scope.percp = ['2','2','2','2','2','2'] ;
        $scope.activationfunc = ['relu', 'relu', 'relu', 'relu', 'relu'];
        $scope.headers = [] ;

        $scope.graph_datasets = [] ;
        $scope.pred_data = [] ;
        $scope.cm = [] ;

        $scope.images = [] ;   $scope.filename = '';
        $scope.excelURL = '';

            // Selected Variables from the list of functions
        $scope.optype = 'Adam' ;
        $scope.frmwrk = 'Keras' ;
        $scope.ptype = 'Classification';
        $scope.layer = default_layers ;
        //$scope.xl_file = new file();
        $scope.chart = null ;


        $scope.Upload_btn = 'Upload' ;
        $scope.Upload_btn_disable = false ;

        // Initial Hyperparameters
        $scope.flt_LR = 0.0003 ;
        $scope.int_Batch = 32 ;
        $scope.int_OpRate = 0.005 ;
        $scope.flt_testsplit = 0.1 ;
        $scope.flt_vldsplit = 0.1 ;


        // boolean for ui-control
        $scope.Isplaydisabled = false ;
        $scope.Ispausedisabled = true ;
        $scope.Isstopdisabled = true ;
        $scope.Isresetdisabled = true ;
        $scope.hyperpara_disabled = false ;
        $scope.showgraph = false ;
        $scope.progressivebar = true ;
        $scope.progressivebar2 = true ;


        $scope.play_tooltip = 'No File has been Uploaded' ;
        $scope.stop_tooltip = ' Stop Neural Network Tuning and Reset Parameters : Currently Not Tuning any network';
        $scope.code = ' ';
        $scope.FABOpen = true ;   $scope.Isguest = true ; $scope.showdrop = false ;
        $scope.username= 'Guest' ;
        $scope.profile = 'user.svg';
        $scope.NNmodels = {} ;

        var elem = document.getElementById('NN_park');
       // var elem = $("#NN-visualizer").getElementById("NeuralPark");
        var two = new Two({
            type: Two.Types.canvas,
             width: 1300, height: 700
            // fullscreen:
        }).appendTo(elem);

        var colors = [
            'rgb(255, 64, 64)',
            'rgb(0, 128, 64)',
            'rgb(0, 200, 255)',
            'rgb(135, 90, 68)',
            'rgb(153, 75, 55)',
            'rgb(255, 50, 0)'
        ];
        colors.index = 0;

        var radius = 20;
        var radialGradient = two.makeRadialGradient(
            0, 0,
            radius,
            new Two.Stop(0, 'rgba(255, 100, 74, 1)', 1),
            new Two.Stop(1, 'rgba(0, 0, 128, 250)', 0)
        );

        var linearGradient = two.makeLinearGradient(
            0, 0,
           1300, 700,
            new Two.Stop(0, colors[1]),
            new Two.Stop(1, colors[0]),
            new Two.Stop(1, colors[1])
        );
        var arrowGradient  = two.makeLinearGradient(
            0,0,1300,700,
            new Two.Stop(0, colors[1]),
            new Two.Stop(1, colors[5]),
            new Two.Stop(1, colors[0])
        ) ;
        addlayer(default_layers,default_neurons) ;

        two.update();
        //$cookies.put('user','name') ;
        Login();
        function addneuron(x,y,r)
        {
            var circle = two.makeCircle(x, y, r);
            // The object returned has many stylable properties:
            circle.fill = radialGradient ;//getRandomColor();
            circle.stroke = 'blue'; // Accepts all valid css color
            circle.linewidth = 2.5;

            /*   $(circle._renderer.elem)
                .css('cursor', 'pointer')
                .click(function(e) {
                    circle.fill = getRandomColor();
                });


             var theta = Math.PI * 2 * (frameCount / 60);

             grad_x = 0.75 * radius * Math.cos(theta);
             grad_y = 0.75 * radius * Math.sin(theta);

             gradient.focal.x = x;
             gradient.focal.y = y; */
           // var curve = two.makeLine(x, y, 120, 50);
           // curve.linewidth = 2;
         //   curve.scale = 1.75;
            //curve.rotation = Math.PI / 2; // Quarter-turn
           // curve.noFill();


        }

        function addlayer(layers,neurons)
        {
            //console.log(layers);

            neurons.splice(layers,0,1);             // adding an ouptut layer
            layers = layers + 1 ;
            two.clear() ;
            var i,j ;
            neurons_total(layers,neurons);

            if($scope.ptype === 'Classification')
            {
                for(i=0;i<layers;++i)
                {
                    //console.log(neurons[i]);
                    init_y = 20 ;


                    for(j=0;j<neurons[i];++j)
                    {
                        //console.log(neurons[i]);
                        if(neurons[i]>22)
                        {
                            init_y = 20 + ((j+1)*(700/(22+1))) ;
                        }

                        else
                        {
                            init_y = 20 + ((j+1)*(700/(neurons[i]+1))) ;
                        }


                        addneuron(init_x,init_y,20);

                        abssica.push(init_x) ;
                        ordinate.push(init_y) ;
                        layer_index.push(i);


                        connect_layers(init_x,init_y,i) ;


                    }
                    init_x = 100 + ((i+1)*(1300/(layers+1))) ;


                }
                abssica.splice(0,abssica.length);
                ordinate.splice(0,ordinate.length);
                layer_index.splice(0,layer_index.length);
                init_x = 80 ;
                init_y = 20 ;


            }

            else
            {
                var rectcent_x = 0,  height = 70, width = 70, rectcent_y = 580, diff ;
                //$scope.LSTM_graphic();
                two.clear() ;

                for(i=0;i<layers-1;++i)
                {
                    //rectcent_x = 50*neurons[i].length ;
                    for(j=0;j<neurons[i];++j)
                    {
                        rectcent_y =  (105*(layers-i));
                        neuron_subset = neurons.slice(0,i);
                        subset_max = Math.max.apply(Math,neuron_subset);
                        if(i!=0)
                        {
                            if(neurons[i-1]>neurons[i])
                            {
                                diff = subset_max - neurons[i] ;
                            }
                         /*   else if(neurons[i-1]<=neurons[i])
                            {
                               diff = neurons[i]-neurons[i-1] ;
                           } */
                        }
                        else
                        {
                            diff = 0 ;
                        }

                        rectcent_x = 50 + (110*(neurons[i]-j+diff));
                        var rect = two.makeRoundedRectangle(rectcent_x,rectcent_y, height, width,9);
                        rect.fill = '#f5f5f5';
                        rect.stroke = 'black'; // Accepts all valid css color
                        rect.linewidth = 0.5;
                        var text =  two.makeText("LSTM", rectcent_x, rectcent_y, {size : 20 , family : "Arial"});
                        text.stroke = 'black';
                        text.fill = 'rgb(255,180,120)';
                        if(i!=0)
                        {
                            if(neurons[i]>neurons[i-1])
                            {
                                if(j>=(neurons[i]-neurons[i-1]))
                                {
                                    var upward_arrow = two.makePath(rectcent_x,(rectcent_y+(height/2)),(rectcent_x+width/4),(rectcent_y+(height/2)+8),(rectcent_x-width/4),(rectcent_y+(height/2)+8),false)
                                    upward_arrow.fill = 'black';
                                    var upline = two.makeLine(rectcent_x,(rectcent_y+(height/2)+8),rectcent_x,(rectcent_y+(height/2)+35));
                                    upline.fill = 'black';
                                    upline.linewidth = 6.75 ;
                                }
                            }
                            else
                            {
                                var upward_arrow = two.makePath(rectcent_x,(rectcent_y+(height/2)),(rectcent_x+width/4),(rectcent_y+(height/2)+8),(rectcent_x-width/4),(rectcent_y+(height/2)+8),false)
                                upward_arrow.fill = 'black';
                                var upline = two.makeLine(rectcent_x,(rectcent_y+(height/2)+8),rectcent_x,(rectcent_y+(height/2)+35));
                                upline.fill = 'black';
                                upline.linewidth = 6.75 ;
                            }


                        }
                        if(j!=(neurons[i]-1))
                        {
                            var sidearrow = two.makePath(rectcent_x-(width/2),rectcent_y,rectcent_x-(width/2)-8,(rectcent_y+height/4),rectcent_x-(width/2)-8,(rectcent_y-height/4),true)
                            sidearrow.fill = 'rgb(255,180,120)';
                            sidearrow.stroke = 'rgb(255,180,120)' ;
                            var sideline = two.makeLine(rectcent_x-(width/2)-8,rectcent_y,(rectcent_x-(width/2)-40),rectcent_y);
                            sideline.fill = arrowGradient ;
                            sideline.stroke = 'rgb(255,180,120)' ;
                            sideline.linewidth = 6.75 ;

                        }
                    }

                }

            }
            two.update();
        }

        function connect_layers(x,y,layer_no) {
            var x1, y1, k;
            //   console.log(abssica,'x co-ordinate :') ;
            //  console.log(ordinate,'y co-ordinate :') ;

            if (layer_no != 0) {
                for (k = 0; k < abssica.length; ++k) {
                    // var curve = two.makeCurve(100,100,x,y, true);
                    if (layer_index[k] == (layer_no - 1)) {
                        //console.log('layer index', layer_index[k]);
                        x1 = abssica[k];
                      //  console.log('x co-ordinate :', x1);
                        y1 = ordinate[k];
                        //console.log('y co-ordinate :', y1);
                        var path = two.makeLine(x1, y1, x, y);
                        //  console.log(x1) ; console.log(y1) ;
                        path.linewidth = 1.2;
                        // path.fill = linearGradient;
                        path.stroke = linearGradient;
                    }
                }


            }


        }
            function line_chart_builder(id, data_sets, label_length)
        {
          //  var ctx = document.getElementById("myChart").getContext('2d');
            //console.log(Math.max.apply(Math,data_sets[0].data));
            var label_arr = [] ;
            for(i=0;i<label_length;++i)
            {
                label_arr[i] = i ;
            }
            var max =  0 ;
            for(i=0;i<data_sets.length;++i)
            {
                //console.log(Math.max.apply(Math,data_sets[i].data));
                if(Math.max.apply(Math,data_sets[i].data)>max)
                {
                    max = Math.max.apply(Math,data_sets[i].data) ;
                }
                //console.log(max)
            }
            $(document).ready(function()
            {
                var ctx = id;
                //console.log(ctx);
                var data = {
                    labels : label_arr,
                    datasets : data_sets
                };

                var options = {
                    title : {
                        display : "Display-metrics",
                        position : "top",
                        text : "Metrics",
                        fontSize : 18,
                        fontColor : "#111"
                    },
                    legend : {
                        display : true,
                        position : "bottom"
                    },
                    scales: {
                        yAxes: [{
                            //display : 'true',

                            ticks: {
                                maxTicksLimit : 25,
                                beginAtZero: true,
                                stepSize: Math.max.apply(Math,data_sets[0].data)/25,
                                max : max,
                                display : true
                                    },
                            gridLines : {
                                display : true
                            },
                            display : true
                                }],
                         xAxes : [{
                             ticks : {
                                 maxTicksLimit : 10,
                                //autoskip :true,
                                //maxTicksLimit : 100
                                display : true
                            },
                        gridLines : {
                            display : true
                        },
                       // display : 'true',
                             scaleLabel: {
                                 display: true,
                                 labelString: 'EPOCHS'
                             }
                        }]
                            },
                    responsive: true,
                    maintainAspectRatio: false,

                    plugins :
                    {
                        zoom :
                        {
                            pan :
                            {
                                enabled : true,
                                mode : 'xy',
                                speed : 20,
                                threshold : 10
                            },
                            zoom :
                            {
                              enabled : true,
                              drag : true,
                              mode : 'xy',
                              speed : 0.1,
                              threshold : 2,
                              sensitivity : 3
                            }
                        }

                    }
                };

                var chart = new Chart.Line( ctx, {
                    data : data,
                    options : options
                } );
            });

            //ctx.moveTo(100, 150);
            //ctx.lineTo(450, 50);
            //ctx.lineWidth = 10;

        };

         function confusionmatrix_builder(id, cm)
         {
             ctx = id ;
            $(document).ready(function(){
                $scope.chart =  new Chart(ctx, {
                    type: 'matrix',
                     data: {
                        datasets: [{
                            label: 'Confusion Matrix',
                            data: [
                                { x: 1, y: 1, v: cm[1][0] },
                                { x: 1, y: 2, v: cm[0][0] },
                                { x: 2, y: 1, v: cm[1][1] },
                                { x: 2, y: 2, v: cm[0][1] },

                            ],
                            backgroundColor: function(ctx) {
                                var x_val = ctx.dataset.data[ctx.dataIndex].x ;
                                var y_val = ctx.dataset.data[ctx.dataIndex].y ;

                                var value = ctx.dataset.data[ctx.dataIndex].v;
                                var alpha = (value) / 40;

                                if((x_val+y_val)===3)
                                {
                                    return Color('green').alpha(alpha).rgbString();
                                }
                                return Color('red').alpha(alpha).rgbString();
                            },
                            width: function(ctx) {
                                var a = ctx.chart.chartArea;
                                return (a.right - a.left) / 3.5;
                            },
                            height: function(ctx) {
                                var a = ctx.chart.chartArea;
                                return (a.bottom - a.top) / 3.5;
                            }
                        }]
                    },
                    options: {
                        title : {
                            display : "Confusion-Matrix ",
                            position : "top",
                            text : "Metrics",
                            fontSize : 18,
                            fontColor : "#111"
                        },
                        legend: {
                            display: false
                        },
                        tooltips: {
                            callbacks: {
                                title: function() { return null ;},
                                label: function(item, data) {
                                    var v = data.datasets[item.datasetIndex].data[item.index];
                                    return [ v.v];
                                }
                            }
                        },
                        scales: {
                            xAxes: [{
                                ticks: {
                                    display: true,
                                    min: 0.5,
                                    max: 3.5,
                                    stepSize: 1
                                },
                                gridLines: {
                                    display: false
                                },
                                afterBuildTicks: function(scale, ticks) {
                                    return ticks.slice(1, 2);
                                }
                            }],
                            yAxes: [{
                                ticks: {
                                    display: true,
                                    min: 0.5,
                                    max: 3.5,
                                    stepSize: 1
                                },
                                gridLines: {
                                    display: false
                                },
                                afterBuildTicks: function(scale, ticks) {
                                    return ticks.slice(1, 2);
                                }
                            }]
                        }
                    }
                })
            });
         }

        function neurons_total(layers,neurons)
        {
            var k;
            total_neurons = 0 ;
            for(k=0;k<layers;++k)
            {
                total_neurons = total_neurons + neurons[k];
            }
            //console.log('total neurons',total_neurons);

        }

         function play()
            {
           // console.log('play-clicked');
            $scope.Isplaydisabled = true;
            //console.log($scope.Isplaydisabled);
            $scope.Isstopdisabled = false ;
            $scope.hyperpara_disabled = true ;
            //console.log(final_headers);
            $scope.stop_tooltip = 'Stop Neural Network Tuning and Reset Parameters' ;
            $scope.Isresetdisabled = false ;
            $scope.progressivebar2 = false;
            $scope.Upload_btn_disable = true ;

          /*  if(login_counter>2)
            {
                $scope.logindialog() ;
            }*/


            $http.get('/data',
                {
                    params : {
                        framework : $scope.frmwrk,
                        type : $scope.ptype,
                        batch_size : $scope.int_Batch,
                        optimization : $scope.optype,
                        testsplit : $scope.flt_testsplit,
                        validsplit : $scope.flt_vldsplit,
                        learning_rate : $scope.flt_LR,
                        layers : $scope.layer,
                        neurons : neurons_list,
                        activation : activation_list,
                        headers : final_headers,
                        target : target,
                        dropouts : droputs,
                        filename : $scope.filename,
                        user : $scope.username
                    }

                  }).then(function (res){
                    //console.log(res);
                    socket = io.connect('http://localhost:3000');
                    socket.on('metrics',function(response){
                        $scope.$applyAsync(function(){
                            //console.log("Metrics",res);
                            $scope.progressivebar2 = true ;
                            var dataset_len ;
                            //console.log(response);
                            var loss = response.loss ;
                            var val_loss = response.val_loss ;
                            loss = str2numarr(loss);


                            val_loss = str2numarr(val_loss);
                            //console.log(loss);

                            loss = {
                                label : "Loss",
                                data : loss,
                                backgroundColor : "blue",
                                borderColor : "blue",
                                fill : false,
                                lineTension : 0.1,
                                pointRadius : 0
                            };
                            val_loss = {
                                label : "Val_loss",
                                data : val_loss,
                                backgroundColor : "green",
                                borderColor : "green",
                                fill : false,
                                lineTension : 0.1,
                                pointRadius : 0
                            };


                            $scope.graph_datasets.push(loss) ;
                            $scope.graph_datasets.push(val_loss);

                            //console.log(typeof loss);
                            if($scope.ptype==='Classification')
                            {
                                var accuracy = response.accuracy ;
                                var val_accuracy = response.val_accuracy ;
                                accuracy = str2numarr(accuracy);
                                val_accuracy = str2numarr(val_accuracy);


                                accuracy = {
                                    label : "Accuracy",
                                    data : accuracy,
                                    backgroundColor : "red",
                                    borderColor : "red",
                                    fill : false,
                                    lineTension : 0.1,
                                    pointRadius : 0
                                };

                                val_accuracy = {
                                    label : "Val_accuracy",
                                    data : val_accuracy,
                                    backgroundColor : "lightblue",
                                    borderColor : "lightblue",
                                    fill : false,
                                    lineTension : 0.1,
                                    pointRadius : 0
                                };

                                $scope.graph_datasets.push(accuracy);
                                $scope.graph_datasets.push(val_accuracy);

                                $scope.cm = response.confusion_matrix ;
                                confusionmatrix_builder($('#Prediction-Chart'),$scope.cm)
                            }
                            else
                            {
                                var trainset = response.y_train_inv[0] ;
                                var testset_1 = response.y_test_inv[0] ;
                                var op_predictions_1 = response.y_pred_inv ;
                                var testset = [], op_predictions = [] ;
                                for(i=0;i<trainset.length;++i)
                                {
                                    if(i==trainset.length-1)
                                    {
                                        testset[i] = trainset[i] ;
                                        op_predictions[i] = trainset[i] ;
                                    }
                                    testset[i] = null ;
                                    op_predictions[i] = null ;
                                }

                                for(i=0;i<testset_1.length;++i)
                                {
                                    testset[trainset.length + i] = testset_1[i] ;
                                    op_predictions[trainset.length + i] = op_predictions_1[i] ;
                                }


                                dataset_len = trainset.length + testset_1.length ;
                                trainset = {
                                    label : "Train Set",
                                    data : trainset,
                                    backgroundColor : "black",
                                    borderColor : "black",
                                    fill : false,
                                    lineTension : 0.9,
                                    pointRadius : 0
                                };

                                testset = {
                                    label : "Actual Value",
                                    data : testset,
                                    backgroundColor : function(){

                                    },
                                    borderColor : "blue",
                                    fill : false,
                                    lineTension : 0.9,
                                    pointRadius : 0
                                };

                                op_predictions = {
                                    label : "Predicted Value",
                                    data : op_predictions,
                                    backgroundColor : "red",
                                    borderColor : "red",
                                    fill : false,
                                    lineTension : 0.9,
                                    pointRadius : 0
                                };

                                $scope.pred_data.push(trainset);
                                $scope.pred_data.push(testset);
                                $scope.pred_data.push(op_predictions);
                                line_chart_builder($('#Prediction-Chart'),$scope.pred_data,dataset_len);
                            }
                            line_chart_builder($('#Metrics-Chart'), $scope.graph_datasets,100) ;
                            $scope.showgraph = true;
                            //console.log($scope.showgraph);
                            login_counter++
                        });

                    });
                    socket.on('code',function(code){
                        $scope.$applyAsync(function(){
                          //console.log(typeof code);
                         // console.log(code);
                          $scope.code = code ;
                        })
                    });

                });

        }

        $scope.navToggle = function () {
            $http.post('/history',{
                body : { user : $scope.username }
            }).then(function(res){
                console.log(res.data);
                $scope.NNmodels = res.data ;
                //console.log(Object.keys($scope.NNmodels).length)
                for(i=0;i<Object.keys($scope.NNmodels).length;++i)
                {
                    if($scope.NNmodels['file'+(i+1)].ptype =='Classification')
                    {
                        $scope.images[i] = 'binary.svg' ;
                    }
                    else
                    {
                        $scope.images[i] = 'timeseries.svg' ;
                    }
                   // console.log($scope.images[i]);
                }
            });

            if(toggle_nav===true)
            {
                document.getElementById('SideNav').style.width = "100%" ;
                document.getElementById('Main').style.marginLeft = "27%" ;
                toggle_nav = false ;
               // console.log(toggle_nav);
            }

            else
            {
                document.getElementById('SideNav').style.width = "0%";
                document.getElementById('Main').style.marginLeft = "0%";
                toggle_nav = true;
               // console.log(toggle_nav);
            }


            $mdSidenav('right').toggle() ;
        };


        $scope.reset = function()
        {
            Reset() ;
            };
            function Reset()
            {
                $scope.Isplaydisabled = false ;
                $scope.Ispausedisabled = true ;
                $scope.Isstopdisabled = true ;
                $scope.Isresetdisabled = true ;
                $scope.hyperpara_disabled = false ;
                $scope.Upload_btn_disable = false ;
                $scope.Upload_btn = 'Upload' ;

                $scope.graph_datasets.splice(0,$scope.graph_datasets.length);
                $scope.pred_data.splice(0,$scope.pred_data.length);
                $scope.cm.splice(0,$scope.cm.length)
                //console.log($scope.cm);

              //  var matrix_chart = document.getElementById('Prediction-Chart')
              //  const context = matrix_chart.getContext('2d');
              //  context.clearRect(0,0,matrix_chart.width,matrix_chart.height);

               // $scope.chart.clear();
                $scope.chart.destroy();

                $scope.code = null ;
                $scope.showgraph = false ;

            }

            function Login()
        {
            console.log($cookies.getAll());
            if($cookies.get('User'))
            {
                $scope.Isguest = false ;
                $scope.username = $cookies.get('User');
                console.log($scope.username);
                $scope.profile = $cookies.get('Image');
                console.log($scope.profile);
            }
        };

        $scope.addexcel = function()
        {
            $scope.$applyAsync(function()
            {
                $scope.progressivebar = false ;
                $scope.headers.splice(0,$scope.headers.length);

                // Import local excel files to the website
                    //// var regex = /^([a-zA-Z0-9\s_\\.\-:])+(.xlsx|.xls)$/;
                var xlsxflag = false; /*Flag for checking whether excel is .xls format or .xlsx format*/
                if ($("#xls_file").val().toLowerCase().indexOf(".xlsx") > 0) {
                    xlsxflag = true;
                }
                //  console.log("Xlsxflag",xlsxflag);
                var reader = new FileReader();
                reader.onload = function (e) {
                    //    console.log('action_triggered') ;
                    var data = e.target.result;
                    console.log('read data')
                    var data_xlsx= ''
                    data = new Uint8Array(data);
                    for(i=0;i<data.byteLength; ++i)
                    {
                        data_xlsx += String.fromCharCode(data[i]);
                    }
                    console.log('converted to data, Now converting to excel');
                    if (xlsxflag) {
                        var workbook = XLSX.read(data_xlsx, { type: 'binary' });
                    }
                    else {
                        var workbook = XLSX.read(data_xlsx, { type: 'binary' });
                    }
                    console.log("Adding worksheet")
                    var sheet_name_list = workbook.SheetNames;
                    //var cnt = 0;
                    sheet_name_list.forEach(function (y) { /*Iterate through all sheets*/

                        /*  if (xlsxflag) {
                         var exceljson = XLSX.utils.sheet_to_json(workbook.Sheets[y]);
                         //console.log("Hi");
                         console.log(exceljson) ;
                         }
                         else {
                         var exceljson = XLS.utils.sheet_to_row_object_array(workbook.Sheets[y]);
                         console.log(exceljson);
                         } */

                        var worksheet = workbook.Sheets[y];
                        var headers = [];
                        var row ;
                        var data = [];
                        for(z in worksheet)
                        {
                            if(z[0] === '!') continue;
                            //parse out the column, row, and value
                            var tt = 0;
                            for (var i = 0; i < z.length; i++) {
                                if (!isNaN(z[i])) {
                                    tt = i;
                                    break;
                                }
                            };
                            var col =  z.replace(/[0-9]/g, '');
                            const row = parseInt(z.replace(/\D/g,''));
                            var value = worksheet[z].v;

                            if(row == 1)
                            {
                                headers[col] = value;
                                $scope.headers.push(value);
                                continue;
                            }

                            if(!data[row]) data[row]={};
                            data[row][headers[col]] = value;
                        }
                       /* const result = data.reduce(function(r, e) {
                            return Object.keys(e).forEach(function(k) {
                                if(!r[k]) r[k] = [].concat(e[k]);
                                else r[k] = r[k].concat(e[k]);
                            }),r
                        }, {})  ; */
                       //tabular_data = data ;
                        $scope.progressivebar = true;
                        console.log('HTTP.POST');
                        $scope.play_tooltip = "Select Input Parameters" ;
                        //socket.emit("Excel Upload")
                        $http.post('/data',
                            {
                                body : data

                            });
                        $scope.Upload_btn = 'Uploaded' ;
                        $scope.Upload_btn_disable = true ;
                        console.log("Sent Data")

                    });

                };
                console.log("Reading the stream now");
                if (xlsxflag) {
                    reader.readAsArrayBuffer($("#xls_file")[0].files[0]);

                }
                else {
                    reader.readAsBinaryString($("#xls_file")[0].files[0]);
                }
                $scope.filename = $("#xls_file")[0].files[0].name ;

            })

        };

      //  $scope.adddataset = function()
       // {
            // Import local folders(data-sets) to the website

       // };

        $scope.LSTM_graphic = function()
            {
                addlayer(default_layers,default_neurons) ;
            };


        $scope.changelayergraphic = function()
        {
           // console.log("In CLG");
            //console.log($scope.layer);
            var l = Number($scope.layer);
            var n = neurons_list ;
            addlayer(l,n);
        };
        function str2numarr(arr)
        {
            for(i=0;i<arr.length;++i)
            {
                arr[i] = Number(arr[i]);
            }
            return arr ;
        }

        function DialogController($scope, $mdDialog, layers, ptype) {

           $scope.ActNeuronPara = [
                {
                    no_percp : 2,
                    Dropout : 0.1
                }
            ];

            for(i=0;i<layers;++i)
            {
                $scope.ActNeuronPara[i] = {} ;
                $scope.ActNeuronPara[i].no_percp = neurons_list[i];
                $scope.ActNeuronPara[i].Dropout = droputs[i] ;
            }
            //console.log(neurons_list) ;
            //console.log($scope.ActNeuronPara) ;

            $scope.hide = function() {
                $mdDialog.hide();
            };

            $scope.cancel = function() {
                Reset();
                $mdDialog.cancel();
            };

            $scope.answer = function() {
                while(neurons_list.length>0)
                {
                    neurons_list.pop();
                }
                for(i=0;i<layers;++i) {
                    neurons_list[i] = Number($scope.ActNeuronPara[i].no_percp);
                    droputs[i] = Number($scope.ActNeuronPara[i].Dropout);
                    console.log($scope.ActNeuronPara[i].no_percp);
                    console.log(neurons_list[i])
                }
                console.log(neurons_list);

                var n = neurons_list
                console.log(neurons_list);
                addlayer(Number(layers),n);
                console.log(neurons_list);
                //console.log(typeof layers);
                //console.log(neurons_list);

                $mdDialog.hide();
            };
        }

        $scope.neurons_dialog = function(ev)
        {
            $mdDialog.show({
                controller: DialogController,
                templateUrl: 'MLP_dialog.html',
                parent: angular.element(document.body),
                targetEvent: ev,
                clickOutsideToClose:false,
                locals : {
                   // percp : $scope.percp,
                    //activationfunc : $scope.activationfunc,
                    layers : $scope.layer,
                    ptype :$scope.ptype

                }
            })

        };

        $scope.header_dialog = function(ev)
        {
            $mdDialog.show({
                controller: IP_Header_Controller,
                templateUrl: 'Input-Headers.html',
                parent: angular.element(document.body),
                targetEvent: ev,
                clickOutsideToClose:false,
                locals : {
                    headers : $scope.headers
                }
            })
        };


            $scope.Reloadparams = function(i)
            {
                //console.log('index:\t' + index);
                $scope.layer = $scope.NNmodels['file'+(i+1)].layers ;
                neurons_list = $scope.NNmodels['file'+(i+1)].neurons ;
                $scope.ptype = $scope.NNmodels['file'+(i+1)].ptype ;
                droputs = $scope.NNmodels['file'+(i+1)].dropouts ;
                $scope.frmwrk = $scope.NNmodels['file'+(i+1)].framework ;
                neurons_list = neurons_list.map(function(v){
                    return +v ;
                });
                addlayer($scope.layer,neurons_list);

            };
        $scope.downloadreport = function(index)
        {
         // Excel Report for previous records
         var wb = XLSX.utils.book_new() ;
         wb.Props = {
             Title : $scope.NNmodels['file'+(index+1)].fliename,
             Subject : 'Code',
             Author : 'NeuroCognition.app' ,
             CreatedDate : new Date(Date.now())
         };
            const merged_cells = [ { s:{r:0,c:0},e:{r:0,c:10} }]


         Filename = {
            filename :  $scope.NNmodels['file'+(index+1)].fliename
        };

         parameters = {
           Optimizer : $scope.NNmodels['file'+(index+1)].Optimizer,
           layers : $scope.NNmodels['file'+(index+1)].layers,
           neurons : $scope.NNmodels['file'+(index+1)].neurons.toString(),
           test_split : $scope.NNmodels['file'+(index+1)].test_split,
           vald_split : $scope.NNmodels['file'+(index+1)].validation_split,
           Batch_Size : $scope.NNmodels['file'+(index+1)].Batch_size,
           EPOCH : 100,
           learning_rate : $scope.NNmodels['file'+(index+1)].learning_rate
         };

         data = {
             date : $scope.NNmodels['file'+(index+1)].date,
             Framework : $scope.NNmodels['file'+(index+1)].framework,
             Type : $scope.NNmodels['file'+(index+1)].ptype

         };

         layersname =  $scope.NNmodels['file'+(index+1)].layers_name ;



         var parametersWB = XLSX.utils.json_to_sheet([Filename]) ;
         parametersWB["!merges"] = merged_cells ;
         XLSX.utils.sheet_add_json(parametersWB,[data],{origin : "A4"});
            XLSX.utils.sheet_add_json(parametersWB,[parameters],{origin : "A7"});

            var counter = 0;
            var lname_char = "A", lname_num = 10, lname_pos = lname_char + lname_num ;
            var n_char = "B", n_num=12, n_pos = n_char + n_num ;
            var w_char = "B", w_num = 13, w_pos = w_char + w_num ;
            var b_char = "B", b_num = 13, b_pos = b_char + b_num ;
            var width_aoa = [[]] ;

            w = $scope.NNmodels['file'+(index+1)].weights ;
            // console.log(w[counter]);
            b = $scope.NNmodels['file'+(index+1)].biases ;

           /* XLSX.utils.sheet_add_aoa(parametersWB,[
                "SheetJS".split(""),
           /     [1,2,3,4,5,6,7],
                [2,3,4,5,6,7,8],
            ],{origin : "Z1"});*/
            //console.log(layersname);
            //console.log(layersname.length)
            for(i=0;i<parseInt(layersname.length);i++)
            {

              //  console.log(i+ ".\t"+layersname[i]);
              //  console.log(counter);
                if((layersname[i].includes("dense")===true) || (layersname[i].includes("Linear")===true) )
                {
                    /*console.log(lname_pos);
                    console.log(n_pos);
                    console.log(w_pos);
                    console.log(b_pos); */

                    XLSX.utils.sheet_add_aoa(parametersWB,[[layersname[i]]],{origin : lname_pos}) ;
                    //neuron_array =
                   // console.log("layer :\t"+layersname[i]);
                    //console.log(parameters.neurons,typeof parameters.neurons);
                    //no_neurons = Array.from(Array(Number(parameters.neurons.split(",")[counter])+1).keys()).splice(0);
                    //no_neurons = Array.from(Array(Number(parameters.neurons.split(",")[counter])+1).keys()).splice(0);
                   // console.log(w[counter]);
                    length1 = Array.from(Array(w[counter][0].length+1).keys()).slice(1) ;

                    /*width = Array.from(Array(w[counter].length+1).keys()).slice(1);
                    for(i=0;i<width.length;++i)
                    {
                      width_aoa[i] = [width[i]] ;
                    }*/
                    //console.log(length1);
                    width_aoa = getwidth(w[counter]);

                    XLSX.utils.sheet_add_aoa(parametersWB,[length1],{origin : n_pos});
                    XLSX.utils.sheet_add_aoa(parametersWB,[["Weights"]],{origin : "A"+n_num});
                    XLSX.utils.sheet_add_aoa(parametersWB,width_aoa,{origin : "A"+ w_num});
                    XLSX.utils.sheet_add_aoa(parametersWB,w[counter],{origin : w_pos});

                    b_num = parseInt(w_num) + parseInt(w[counter].length) + 2 ;

              /*      console.log("b_num\t"+b_num);
                    console.log("w_num\t"+w_num);
                    console.log("wlength \t:"+w[counter].length);
              */
                    b_pos = b_char + b_num ;
                   // o = str1 + str2 ;
                    length1 = Array.from(Array(b[counter].length+1).keys()).slice(1) ;
                    //width_aoa = getwidth(b[counter]);
                    XLSX.utils.sheet_add_aoa(parametersWB,[length1],{origin : "B"+(b_num-1)});
                    XLSX.utils.sheet_add_aoa(parametersWB,[["Biases"]],{origin : "A"+(b_num-1)});
                    //XLSX.utils.sheet_add_aoa(parametersWB,width_aoa,{origin : "A"+ b_num});
                    XLSX.utils.sheet_add_aoa(parametersWB,[b[counter]],{origin : b_pos});
                    //console.log("blength \t:"+b[counter].length);

                    lname_num = parseInt(b_num) +  5 ; //let b.length =1
                    n_num = parseInt(lname_num) +  2;
                    lname_pos = lname_char + lname_num ;

                //    console.log("n_num\t:"+n_num);

                    n_pos = n_char + n_num ;
                    w_num = parseInt(n_num) + 1 ;
                    w_pos = w_char + w_num ;
                    counter++ ;
                }

                else if((layersname[i].includes("lstm")===true) || (layersname[i].includes("LSTM")===true) )
                {
                //    console.log(layersname[i]);
                    var wh_num = 13  , wh_char = "B", wh_pos = wh_char+wh_num ;
                    var bh_num = 13, bh_char = "B", bh_pos = bh_char + bh_num ;
                    var temp_pos ;

                    weight_type = ["i","f","c","o"] ;
                    var bh, wh = $scope.NNmodels['file'+(index+1)].hidden_weights ;
                    // console.log(w[counter]);
                    if(data.Framework==="PyTorch")
                    {
                        bh = $scope.NNmodels['file'+(index+1)].hidden_biases ;
                    }
                  /*  console.log(lname_pos);
                    console.log(n_pos);
                    console.log(w_pos);
                    console.log(b_pos);
                  */
                    XLSX.utils.sheet_add_aoa(parametersWB,[[layersname[i]]],{origin : lname_pos}) ;

                    //no_neurons = Array.from(Array(Number(parameters.neurons.split(",")[counter])+1).keys()).splice(0);
                    //console.log("no_of neuron\t:"+ no_neurons);
                    XLSX.utils.sheet_add_aoa(parametersWB,[["Weights"]],{origin : "A" + (w_num-1)}) ;
                    XLSX.utils.sheet_add_aoa(parametersWB,[["Wi"]],{origin : "A" + w_num}) ;
                    temp_pos = w[counter].length/4 ;
                    XLSX.utils.sheet_add_aoa(parametersWB,[["Wf"]],{origin : "A" + (w_num + temp_pos)}) ;
                    XLSX.utils.sheet_add_aoa(parametersWB,[["Wc"]],{origin : "A" + (w_num + (temp_pos*2))}) ;
                    XLSX.utils.sheet_add_aoa(parametersWB,[["Wo"]],{origin : "A" + (w_num + (temp_pos*3))}) ;
                    XLSX.utils.sheet_add_aoa(parametersWB,w[counter],{origin : w_pos});



                    b_num = parseInt(w_num) + parseInt(w[counter].length) + 2 ;



                  /*  console.log("b_num\t"+b_num);
                    console.log("w_num\t"+w_num);
                    console.log("wlength \t:"+w[counter].length);
                  */
                    b_pos = b_char + b_num ;
                  //  console.log("blength \t:"+b[counter].length);
                    XLSX.utils.sheet_add_aoa(parametersWB,[["Biases"]],{origin : "A"+(b_num-1)});
                    XLSX.utils.sheet_add_aoa(parametersWB,[b[counter]],{origin : b_pos});


                    wh_num = parseInt(b_num) + 4 ;
                    wh_pos = wh_char + wh_num ;
                    XLSX.utils.sheet_add_aoa(parametersWB,[["Hidden Weights"]],{origin : "A" + (wh_num-1)}) ;
                    XLSX.utils.sheet_add_aoa(parametersWB,[["Whi"]],{origin : "A" + wh_num}) ;
                    temp_pos = w[counter].length/4 ;
                    XLSX.utils.sheet_add_aoa(parametersWB,[["Whf"]],{origin : "A" + (wh_num + temp_pos)}) ;
                    XLSX.utils.sheet_add_aoa(parametersWB,[["Whc"]],{origin : "A" + (wh_num + (temp_pos*2))}) ;
                    XLSX.utils.sheet_add_aoa(parametersWB,[["Who"]],{origin : "A" + (wh_num + (temp_pos*3))}) ;
                    XLSX.utils.sheet_add_aoa(parametersWB,wh[counter],{origin : wh_pos});

                    bh_num = parseInt(wh_num) + parseInt(wh[counter].length) + 4 ;
                    bh_pos = bh_char + bh_num ;

                    if(data.Framework==="PyTorch")
                    {
                        XLSX.utils.sheet_add_aoa(parametersWB,[["Hidden Biases"]],{origin : "A"+(bh_num-1)});
                        XLSX.utils.sheet_add_aoa(parametersWB,[bh[counter]],{origin : bh_pos});
                    }


                    lname_num = parseInt(bh_num) +  5 ; //let b.length =1
                    n_num = parseInt(lname_num) +  2;
                    lname_pos = lname_char + lname_num ;

                  //  console.log("n_num\t:"+n_num);

                    n_pos = n_char + n_num ;
                    w_num = parseInt(n_num) + 3 ;
                    w_pos = w_char + w_num ;
                    counter++ ;

                }
              //  console.log(i+ ".\t"+layersname[i]);
            }

       //  aoa1 = ["1"];
       //  aoa2 = ["A",'B','C',];


      //   XLSX.utils.sheet_add_aoa(parametersWB,[aoa1,aoa2]);
         //var aoaWB = XLSX.utils.json_to_sheet(array_of_arrays);
         XLSX.utils.book_append_sheet(wb,parametersWB,"parameters");
         //XLSX.utils.book_append_sheet(wb,aoaWB,'aoa');
         XLSX.writeFile(wb,'Report.xlsx');


        };

        function getwidth(parametermatrix)
        {
            var width_aoa = [[]];
            width = Array.from(Array(parametermatrix.length+1).keys()).slice(1);
            for(ii=0;ii<width.length;++ii)
            {
                width_aoa[ii] = [width[ii]] ;
            }
            return width_aoa ;
        }

        $scope.DeleteDoc = function(index)
        {
            console.log($scope.NNmodels['file'+(index+1)]._id);
            $http.post('/deleterec', {
                file : $scope.NNmodels['file'+(index+1)].fliename,
                date : $scope.NNmodels['file'+(index+1)].date,
                file_id : $scope.NNmodels['file'+(index+1)]._id


            }).then(function(res){

            })
        };

        function IP_Header_Controller($scope, $mdDialog, headers) {

           //    console.log(headers);
            final_headers.splice(0,final_headers.length);
            //console.log(final_headers);
            $scope.JSON_header = [
                {
                    header : '',
                    input  : true,
                    target: false,
                    disabled : false
                }
            ]  ;

            $scope.target = null ;

            for(i=0;i<headers.length;++i)
            {
                $scope.JSON_header[i] = {} ;
                $scope.JSON_header[i].header = headers[i] ;
                $scope.JSON_header[i].input = true ;
                $scope.JSON_header[i].target = false ;
                $scope.JSON_header[i].disabled = false ;
            }

            //console.log($scope.JSON_header) ;

           // console.log($scope.headers);

            $scope.hide = function() {
                $mdDialog.hide();
            };

            $scope.cancel = function() {

                $mdDialog.cancel();
            };

            $scope.answer = function()
            {
                for(i=0;i<$scope.JSON_header.length;++i)
                {
                    if($scope.JSON_header[i].input===true)
                    {
                        final_headers.push($scope.JSON_header[i].header) ;
                    }
                    if($scope.JSON_header[i].target===true)
                    {
                         target = $scope.JSON_header[i].header;
                    }

                }

                //target = $scope.target ;
               // console.log(final_headers);
               // console.log(target) ;
                $mdDialog.hide();
                play() ;
                //final_headers.splice(0);
            };
            $scope.isChecked = function(ip) {
                ip = !ip ;
                console.log(ip);
            };

            $scope.CheckboxDisable = function(ip,target)
            {
                ip = !ip ;
                console.log(target);
                for(i=0;i<$scope.JSON_header.length;++i)
                {
                    if ($scope.JSON_header[i].header!=target)
                    {
                        $scope.JSON_header[i].disabled = true;
                        $scope.JSON_header[i].target = false ;
                        //console.log($scope.JSON_header[i].disabled);

                    }

                }
            }
        }
        });
/*
 $scope.logindialog = function ()
 {
 $mdDialog.show({
 controller : Login_Controller,
 templateUrl : 'UserLogin.html',
 parent : angular.element(document.body),
 targetEvent: ev,
 clickOutsideToClose : true
 })
 };

 function Login_Controller()
 {
 $scope.hide = function() {
 $mdDialog.hide();
 };

 $scope.cancel = function() {
 $mdDialog.cancel();
 };

 $scope.answer = function()
 {
 $mdDialog.hide() ;

 }


 }





 */


/*
angular.module('webapp', ['ngMaterial']).factory('socket', function ($rootScope) {
    var socket = io.connect();
    return {
        on: function (eventName, callback) {
            socket.on(eventName, function () {
                var args = arguments;
                $rootScope.$apply(function () {
                    callback.apply(socket, args);
                });
            });
        },
        emit: function (eventName, data, callback) {
            socket.emit(eventName, data, function () {
                var args = arguments;
                $rootScope.$apply(function () {
                    if (callback) {
                        callback.apply(socket, args);
                    }
                });
            })
        }
    };
});
*/

