import sys, os
import json
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Dense

#from keras.models import Sequential
#from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


#import mxnet as mx
#from mxnet import nd, autograd, gluon
#from mxnet.gluon import nn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from statistics import  mean

loss_stats = {
"loss": [],
"val_loss": [],
"accuracy" : [],
"val_accuracy" : [],
"y_train_inv": [],
"y_test_inv": [],
"y_pred_inv" : [],
"confusion_matrix" : []
}

w_n_b = {
 "layers" : [],
 "weights" : [],
 "biases" : []
}

y_transformer = RobustScaler()
f_transformer = RobustScaler()


def coder(parameters):

    #Necessary Parameters :
    """
                parameters[2] = Learning Rate / alpha
                parameters[3] = Train-Test split
                parameters[4] = Optimizers
                parameters[5] = Mini batch size
                parameters[6] = layers
                parameters[7] = Neurons per layer
                parameters[8] =
                parameters[10] = Input Parameters
                parameters[11] = Output Target
                parameters[12] = Train-Val Split
                parameters[13] = Dropouts for each layers

    """
    alpha = parameters[2]
    layers = int(parameters[6])
    layers_1 = int(parameters[6]) + 1
    neurons = parameters[7]
    activationfunction = parameters[8]
    dropouts = parameters[13]
    try:
        code = open('DL_code.py', 'r+')
        code.truncate(0)
        #py_out = open("DL_code.py", "a")
        code_string = ''
        with open('DL_code.py', 'a') as py_out:
                    #py_out.truncate(0)
                    if(parameters[0]=='Keras'):
                        if(parameters[1]=='Classification'):
                            #Import Necessary libraries
                            py_out.write("import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.metrics import confusion_matrix\n\n\nimport keras\nfrom keras.models import Sequential\nfrom keras.layers import Dense\nfrom keras.layers import Dropout\n")

                            #Data Preprocessing
                            py_out.write("input_frame = pd.read_json(file)\ntarget = \"" +parameters[11]+"\"")
                            py_out.write("\nX = input_frame.drop(target, axis = 1)\ny = input_frame[target]\ncolumn_list =" +str(parameters[10]))
                            py_out.write("\nX = X.filter(column_list, axis=1)")
                            #py_out.write("\nprint(\"Input file\")")
                            py_out.write("\nprint(X)")
                            #py_out.write("\nprint(\"Target\")")
                            py_out.write("\nprint(y)")
                            py_out.write("\nfor column in column_list:\n\tif(X[column].dtype == 'object'):\n\t\tX[column] = X[column].astype('category')\n\t\tX[column] = X[column].cat.codes\nif(y.dtype == 'object'):\n\ty = y.astype('category')\n\ty = y.cat.codes")
                            py_out.write("\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size = " + parameters[3] + ", random_state = 2)\nsc = StandardScaler()\nX_train = sc.fit_transform(X_train)\nX_test = sc.transform(X_test)")


                            #Deep Learning Model
                            if(parameters[4]== 'SGD'):
                                py_out.write("\noptimizer = keras.optimizers.SGD(learning_rate="+ alpha+")\n")

                            elif(parameters[4]== 'Adam'):
                                py_out.write("\noptimizer = keras.optimizers.Adam(learning_rate="+ alpha+")\n")

                            elif(parameters[4]== 'Adagrad'):
                                py_out.write("\noptimizer = keras.optimizers.Adagrad(learning_rate="+ alpha+")\n")

                            elif(parameters[4]== 'RMSProp'):
                                py_out.write("\noptimizer = keras.optimizers.RMSprop(learning_rate="+ alpha+")\n")

                            elif(parameters[4]== 'Adamax'):
                                py_out.write("\noptimizer = keras.optimizers.Adamax(learning_rate="+ alpha+")\n")


                            py_out.write("classifier = Sequential()\n")

                            for i in range(layers+1):
                                if(i == 0):
                                    py_out.write("classifier.add(Dense(units ="+str(neurons[i]) +", kernel_initializer = 'uniform', activation = 'relu', input_dim = X_test.shape[1]))\n")
                                    py_out.write("classifier.add(Dropout("+str(dropouts[i])+"))\n")
                                elif(i == layers):
                                    py_out.write("classifier.add(Dense(units ="+str(neurons[i]) +", kernel_initializer = 'uniform', activation = 'sigmoid'))\n")
                                else:
                                    py_out.write("classifier.add(Dense(units ="+str(neurons[i]) +", kernel_initializer = 'uniform', activation = 'relu'))\n")
                                    py_out.write("classifier.add(Dropout("+str(dropouts[i])+"))\n")
                            # Model Training and evaluation

                            py_out.write("classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])\n")
                            py_out.write("history = classifier.fit(X_train, y_train, batch_size = " +str(parameters[5])+", epochs = 100, validation_split="+str(parameters[12])+")\n")
                            py_out.write("y_pred = classifier.predict(X_test)\ny_pred = (y_pred > 0.5)\ncm = confusion_matrix(y_test, y_pred)")
                            py_out.write("\nprint(\'confusion matrix:\')")
                            py_out.write("\nprint(cm)")

                        if(parameters[1]=='Time Series'):
                            #Import Necessary libraries
                            py_out.write("import numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import RobustScaler\n\nimport keras\nfrom keras.models import Sequential\nfrom keras.layers import LSTM\nfrom keras.layers import Dropout\nfrom keras.layers import Dense\n")

                            #data_preprocessing
                            py_out.write("def create_dataset(X, y, time_steps=1):\n\tXs, ys = [], []\n\tfor i in range(len(X) - time_steps):\n\t\tv = X.iloc[i:(i + time_steps)].values\n\t\tXs.append(v)\n\t\tys.append(y.iloc[i + time_steps])\n\treturn np.array(Xs), np.array(ys)")
                            py_out.write("\ninput_frame = pd.read_excel(Enter file address here)")
                            #py_out.write("\n
                            py_out.write("\ntarget = " + "\'" +parameters[11]+ "\'")
                            py_out.write("\nX = input_frame\ny = input_frame[target]\ncolumn_list =" +str(parameters[10]))
                            py_out.write("\nX = X.filter(column_list, axis=1)\n") #change this
                            py_out.write("for column in column_list:\n\tif(X[column].dtype == 'object'):\n\t\tX[column] = X[column].astype('category')\n\t\tX[column] = X[column].cat.codes\n")
                            py_out.write("test_size = int(len(X) * "+parameters[3]+")\ntrain_size = len(X) - test_size\ntrain, test = X.iloc[0:train_size], X.iloc[train_size:len(X)]\n")
                            py_out.write("y_transformer = RobustScaler()\nf_transformer = RobustScaler()\n")
                            py_out.write("y_transformer = y_transformer.fit(train[[target]])\ny_trn = y_transformer.transform(train[[target]])\ny_tst = y_transformer.transform(test[[target]])\nf_transformer = f_transformer.fit(train[column_list].to_numpy())\n")
                            py_out.write("train.loc[:, column_list] = f_transformer.transform(train[column_list].to_numpy())\ntest.loc[:, column_list] = f_transformer.transform(test[column_list].to_numpy())\ntime_steps = 10\ny_trn = pd.DataFrame(y_trn)\ny_tst = pd.DataFrame(y_tst)\nX_train, y_train = create_dataset(train, y_trn, time_steps)\nX_test, y_test = create_dataset(test, y_tst, time_steps)\n")

                            #Deep Learning Model
                            if(parameters[4]== 'SGD'):
                                py_out.write("optimizer = keras.optimizers.SGD(learning_rate="+ alpha+")\n")

                            elif(parameters[4]== 'Adam'):
                                py_out.write("optimizer = keras.optimizers.Adam(learning_rate="+ alpha+")\n")

                            elif(parameters[4]== 'Adagrad'):
                                py_out.write("optimizer = keras.optimizers.Adagrad(learning_rate="+ alpha+")\n")

                            elif(parameters[4]== 'RMSProp'):
                                py_out.write("optimizer = keras.optimizers.RMSprop(learning_rate="+ alpha+")\n")

                            elif(parameters[4]== 'Adamax'):
                                py_out.write("optimizer = keras.optimizers.Adamax(learning_rate="+ alpha+")\n")

                            py_out.write("regressor = Sequential()\n")
                            for i in range(layers):
                                if(i == 0):
                                    if(layers-1 == 0):
                                        py_out.write("regressor.add(LSTM(units = "+neurons[i]+",return_sequences = False, input_shape = (X_train.shape[1], X_train.shape[2])))\n")
                                    else:
                                        py_out.write("regressor.add(LSTM(units = "+neurons[i]+",return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))\n")
                                elif(i == layers-1):
                                    py_out.write("regressor.add(LSTM(units = "+neurons[i]+", return_sequences = False))\n")
                                    py_out.write("regressor.add(Dropout("+dropouts[i]+"))\n")

                                else:
                                    # Adding remaining hidden layers
                                    py_out.write("regressor.add(LSTM(units = "+neurons[i]+",return_sequences = True ))\n")
                            #Model Training and evaluation
                            py_out.write("regressor.add(Dense(units = 1))\n")
                            py_out.write("regressor.compile(optimizer = optimizer, loss = 'mean_squared_error')\n")
                            py_out.write("history = regressor.fit(X_train, y_train, validation_split="+parameters[12]+", epochs=100, batch_size =" +parameters[5]+")\n")
                            py_out.write("y_pred = regressor.predict(X_test)\ny_train_inv = y_transformer.inverse_transform(y_train.reshape(1, -1))\ny_test_inv = y_transformer.inverse_transform(y_test.reshape(1, -1))\ny_pred_inv = y_transformer.inverse_transform(y_pred)\n")

                    if(parameters[0]=='PyTorch'):
                        #Import Libraries
                        code_string += "import numpy as np\nimport pandas as pd\n"
                        code_string += "\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\nfrom torch.utils.data import Dataset, DataLoader\nfrom torch.autograd import Variable\n"

                        if(parameters[1]=='Classification'):
                            code_string += ("\nclass Dataset(Dataset):")

                            code_string += ("\n\tdef __init__(self, X_data, y_data):")
                            code_string += ("\n\t\tself.X_data = X_data")
                            code_string += ("\n\t\tself.y_data = y_data")

                            code_string += ("\n\tdef __getitem__(self, index):")
                            code_string += ("\n\t\treturn self.X_data[index], self.y_data[index]")

                            code_string += ("\n\tdef __len__ (self):")
                            code_string += ("\n\t\treturn len(self.X_data)")


                            code_string += ("\nfrom sklearn.preprocessing import StandardScaler\n")
                            code_string += ("\nfrom sklearn.model_selection import train_test_split\n")
                            #code_string += ("from sklearn.preprocessing import RobustScaler
                            code_string += ("from sklearn.metrics import confusion_matrix\n")


                            #Data Preprocessing
                            code_string += ("input_frame = pd.read_json(file)\ntarget = \'" +parameters[11]+"\'")
                            code_string += ("\nX = input_frame.drop(target, axis = 1)\ny = input_frame[target]\ncolumn_list =" +str(parameters[10]))
                            code_string += ("\nX = X.filter(column_list, axis=1)\n")
                            code_string += ("for column in column_list:\n\tif(X[column].dtype == 'object'):\n\t\tX[column] = X[column].astype('category')\n\t\tX[column] = X[column].cat.codes\nif(y.dtype == 'object'):\n\ty = y.astype('category')\n\ty = y.cat.codes")
                            code_string += ("\nX_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size = " + parameters[3] + ", random_state = 2)\n")
                            code_string += "\nX_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size="+parameters[12]+", random_state = 0)\n"
                            code_string += "sc = StandardScaler()\nX_train = sc.fit_transform(X_train)\nX_val = sc.transform(X_val)\nX_test = sc.transform(X_test)\n\nX_train , y_train = np.array(X_train), np.array(y_train)\nX_val, y_val = np.array(X_val), np.array(y_val)\nX_test, y_test = np.array(X_test), np.array(y_test)"


                            #Deep Learning model
                            code_string +="\nEPOCHS = 100\n"
                            code_string +=("\nBATCH_SIZE = " + parameters[5])

                            code_string +="\ntrain_dataset = Dataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())\n"
                            code_string +="\nval_dataset = Dataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())"
                            code_string +="\ntest_dataset = Dataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())"

                            code_string +="\ntrain_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)"
                            code_string +="\nval_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)"
                            code_string +="\ntest_loader = DataLoader(dataset=test_dataset, batch_size=1)"

                            code_string +="\nlayers = []"
                            code_string +="\nlayers.append(nn.Linear(X_train.shape[1],"+neurons[0]+",bias=True))"

                            for i in range(layers_1):
                                #if(i == 0):
                                if(i == layers_1-1):
                                    code_string +="\nlayers.append(nn.Linear("+neurons[i]+",1,bias=True))"
                                    code_string +="\nlayers.append(nn.Sigmoid())"
                                else:
                                    code_string +="\nlayers.append(nn.Linear("+neurons[i]+","+neurons[i+1]+",bias=True))"
                                    code_string +="\nlayers.append(nn.Dropout("+dropouts[i]+"))"

                            code_string +="\nmodel = nn.Sequential(*layers)"
                            code_string +="\nlayers.clear()"

                            #Tuning, Training and validation
                            code_string +="\ndevice = torch.device(\"cuda:0\" if torch.cuda.is_available() else \"cpu\")"
                            code_string +="\nloss_func = nn.BCELoss()"

                            if(parameters[4]== 'SGD'):
                                code_string +="\noptimizer = torch.optim.SGD(model.parameters(), lr="+alpha+", weight_decay=0.00008)"

                            elif(parameters[4]== 'Adam'):
                                code_string +="\noptimizer = torch.optim.Adam(model.parameters(), lr="+alpha+", weight_decay=0.00008)"

                            elif(parameters[4]== 'RMSProp'):
                                code_string +="\noptimizer = torch.optim.RMSprop(model.parameters(), lr="+alpha+", weight_decay=0.00008)"

                            elif(parameters[4]== 'Adagrad'):
                                code_string +="\noptimizer = torch.optim.Adagrad(model.parameters(), lr="+alpha+", weight_decay=0.00008)"

                            elif(parameters[4]== 'Adamax'):
                                code_string +="\noptimizer = torch.optim.Adamax(model.parameters(), lr="+alpha+", weight_decay=0.00008)"

                            code_string +="\ny_pred = []"
                            code_string +="\ny_actual = []"
                            code_string +="\nfor e in range(EPOCHS):"
                            code_string +="\n\ttrain_epoch_loss = 0"
                            code_string +="\n\tval_epoch_loss = 0"
                            code_string +="\n\ttrain_epoch_acc = 0"
                            code_string +="\n\tval_epoch_acc = 0"

                            code_string +="\n\tmodel.train()"
                            code_string +="\n\tfor X_train_batch, y_train_batch in train_loader:"
                            code_string +="\n\t\tX_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)"
                            code_string +="\n\t\ty_train_pred = model(X_train_batch)"
                            code_string +="\n\t\toptimizer.zero_grad()"
                            code_string +="\n\t\ty_actual.append(y_train_batch.unsqueeze(1))"
                            code_string +="\n\t\ttrain_loss = loss_func(y_train_pred, y_train_batch.unsqueeze(1))"
                            code_string +="\n\t\ttrain_loss.backward()"
                            code_string +="\n\t\toptimizer.step()"
                            code_string +="\n\t\ty_pred_tag = (y_train_pred > 0.5).float()"
                            code_string +="\n\t\tacc = ((y_pred_tag == y_train_batch.unsqueeze(1)).sum().float())/y_train_batch.shape[0]"
                            code_string +="\n\t\ttrain_epoch_loss += train_loss.item()"
                            code_string +="\n\t\ttrain_epoch_acc += acc.item()"
                            code_string +="\n\twith torch.no_grad():"
                            code_string +="\n\t\tmodel.eval()"
                            code_string +="\n\t\tfor X_val_batch, y_val_batch in val_loader:"
                            code_string +="\n\t\t\tX_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)"
                            code_string +="\n\t\t\ty_val_pred = model(X_val_batch)"
                            code_string +="\n\t\t\tval_loss = loss_func(y_val_pred, y_val_batch.unsqueeze(1))"
                            code_string +="\n\t\t\tval_epoch_loss += val_loss.item()"
                            code_string +="\n\t\t\ty_pred_tag = (y_val_pred > 0.5).float()"
                            code_string +="\n\t\t\tacc = ((y_pred_tag == y_val_batch.unsqueeze(1)).sum().float())/y_val_batch.shape[0]"
                            code_string +="\n\t\t\tval_epoch_acc += acc.item()"
                            code_string +="\n\tprint(train_epoch_loss/len(train_loader))"
                            code_string +="\n\tprint(val_epoch_loss/len(val_loader))"
                            code_string +="\n\tprint(train_epoch_acc/len(train_loader))"
                            code_string +="\n\tprint(val_epoch_acc/len(val_loader))"

                            # Test-set Prediction
                            code_string +="\ny_pred_list = []"
                            code_string +="\ny_test_list = []"
                            code_string +="\nmodel.eval()"
                            code_string +="\nfor X_test_batch, y_test_batch in test_loader:"
                            code_string +="\n\tX_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)"
                            code_string +="\n\ty_pred = model(X_test_batch)"
                            code_string +="\n\ty_pred_tag = (y_pred > 0.5).float()"
                            code_string +="\n\ty_pred_list.append(y_pred)"
                            code_string +="\n\ty_test_list.append(y_test_batch)"
                            code_string +="\ny_pred_list = [a.squeeze().tolist() for a in y_pred_list ]"
                            code_string +="\ny_pred_list = [round(a) for a in y_pred_list]"
                            code_string +="\ny_test_list = [a.squeeze().tolist() for a in y_test_list ]"

                            code_string +="\ncm = confusion_matrix(y_test_list, y_pred_list)"
                            code_string +="\nprint(cm)"

                            py_out.write(code_string)

                        if(parameters[1]=='Time Series'):
                                layers_1 = layers_1 - 1
                                #code_string += ("\nfrom sklearn.preprocessing import StandardScaler\n")
                                code_string += ("\nfrom sklearn.preprocessing import RobustScaler\n")
                                #code_string += ("from sklearn.metrics import confusion_matrix\n")
                                code_string += ("\nclass Dataset(Dataset):")

                                code_string += ("\n\tdef __init__(self, X_data, y_data):")
                                code_string += ("\n\t\tself.X_data = X_data")
                                code_string += ("\n\t\tself.y_data = y_data")

                                code_string += ("\n\tdef __getitem__(self, index):")
                                code_string += ("\n\t\treturn self.X_data[index], self.y_data[index]")

                                code_string += ("\n\tdef __len__ (self):")
                                code_string += ("\n\t\treturn len(self.X_data)")
                                code_string+= "\ndef create_dataset(X, y, time_steps=1):"
                                code_string+= "\n\tXs, ys = [], []"
                                code_string+= "\n\tfor i in range(len(X) - time_steps):"
                                code_string+= "\n\t\tv = X.iloc[i:(i + time_steps)].values"
                                code_string+= "\n\t\tXs.append(v)"
                                code_string+= "\n\t\tys.append(y.iloc[i + time_steps])"
                                code_string+= "\n\treturn np.array(Xs), np.array(ys)"

                                code_string+=""
                                code_string+= "\nclass Regressor_LSTM(nn.Module):"
                                code_string+="\n\tdef __init__(self,input_dim, seq_len):"
                                code_string+="\n\t\t"+"super(Regressor_LSTM, self).__init__()"
                                code_string+="\n\t\tself.input_dim = input_dim"
                                code_string+="\n\t\tself.seq_length = seq_len"
                                code_string+="\n\t\tself.IP_Layer = nn.LSTM(input_size=self.input_dim,hidden_size="+neurons[0]+",dropout="+dropouts[0]+")"
                                code_string+="\n\t\tself.Out_Layer = nn.Linear(" +neurons[layers_1-1]+",1,bias=True)"

                                code_string+="\n\tdef forward(self,X, batch):"
                                for i in range(layers_1):
                                    code_string+="\n\t\th_x_"+str(i)+" = Variable(torch.zeros(1, batch," + neurons[i] + "))"
                                    code_string+="\n\t\tc_x_"+str(i)+" = Variable(torch.zeros(1, batch," + neurons[i] + "))"
                                    if(i==0):
                                        code_string+="\n\t\tout"+str(i)+", (h_x_"+str(i+1)+", c_x_"+str(i+1)+") = self.IP_Layer(X.view(self.seq_length,len(X),-1), (h_x_"+str(i)+",c_x_"+str(i)+"))"
                                    else:
                                        code_string+="\n\t\tself.regress"+str(i)+" = nn.LSTM(input_size="+neurons[i-1]+",hidden_size="+neurons[i]+",dropout="+dropouts[i]+")"
                                        code_string+="\n\t\tout"+str(i)+", (h_x_"+str(i+1)+", c_x_"+str(i+1)+") = self.regress"+str(i)+"(out"+str(i)+",(h_x_"+str(i)+", c_x_"+str(i)+"))"
                                code_string+="\n\t\tout = self.Out_Layer(out"+str(layers_1-1)+"[-1].view(batch,-1))"

                                code_string+="\n\t\treturn out.view(-1)\n"

                                #Data Preprocessing
                                code_string += ("\ninput_frame = pd.read_json(file)\ntarget = \'" +parameters[11]+"\'")
                                code_string += ("\nX = input_frame\ny = input_frame[target]\ncolumn_list =" +str(parameters[10]))
                                code_string += ("\nX = X.filter(column_list, axis=1)")

                                code_string += ("\ntest_size = int(len(X) *"+parameters[3]+")")
                                code_string += ("\ntrain_size = len(X) - test_size")
                                code_string += ("\ntrain, test = X.iloc[0:train_size], X.iloc[train_size:len(X)]")
                                code_string += ("\nval_size = int(train_size *"+parameters[12]+")")
                                code_string += ("\nval, train = train.iloc[0:val_size], train.iloc[val_size:train_size]")

                                code_string += ("\ny_transformer = RobustScaler()")
                                code_string += ("\nf_transformer = RobustScaler()")
                                code_string += ("\ny_transformer = y_transformer.fit(train[[target]])")
                                code_string += ("\ny_trn = y_transformer.transform(train[[target]])")
                                code_string += ("\ny_tst = y_transformer.transform(test[[target]])")
                                code_string += ("\ny_val = y_transformer.transform(val[[target]])")
                                code_string += ("\nf_transformer = f_transformer.fit(train[column_list].to_numpy())")
                                code_string += ("\ntrain.loc[:, column_list] = f_transformer.transform(train[column_list].to_numpy())")
                                code_string += ("\ntest.loc[:, column_list] = f_transformer.transform(test[column_list].to_numpy())")
                                code_string += ("\nval.loc[:, column_list] = f_transformer.transform(val[column_list].to_numpy())")
                                code_string += ("\ny_train_inv = y_transformer.inverse_transform(y_val.reshape(1,-1)).tolist()")
                                code_string += ("\nytrain_inv = y_transformer.inverse_transform(y_trn.reshape(1,-1)).tolist()")
                                code_string += ("\ny_train_inv[0].extend(ytrain_inv[0])")
                                code_string += ("\ny_test_inv = y_transformer.inverse_transform(y_tst.reshape(1,-1)).tolist()")

                                code_string += ("\ntime_steps = 10")
                                code_string += ("\ny_trn = pd.DataFrame(y_trn)")
                                code_string += ("\ny_tst = pd.DataFrame(y_tst)")
                                code_string += ("\ny_val = pd.DataFrame(y_val)")
                                code_string += ("\nX_train, y_train = create_dataset(train, y_trn, time_steps)")
                                code_string += ("\nX_test, y_test = create_dataset(test, y_tst, time_steps)")
                                code_string += ("\nX_val, y_val = create_dataset(val, y_val, time_steps)")

                                code_string += ("\nEPOCHS = 100")
                                code_string += ("\nBATCH_SIZE = "+parameters[5])

                                code_string += ("\ntrain_dataset = Dataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())")
                                code_string += ("\nval_dataset = Dataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())")
                                code_string += ("\ntest_dataset = Dataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())")

                                code_string += ("\ntrain_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)")
                                code_string += ("\nval_loader = DataLoader(dataset=val_dataset, batch_size=16, drop_last=True)")
                                code_string += ("\ntest_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False )")

                                code_string += ("\ndevice = torch.device(\"cuda:0\" if torch.cuda.is_available() else \"cpu\")")
                                code_string += ("\nloss_func = nn.MSELoss()")
                                code_string += ("\nregressor_model = Regressor_LSTM(X_train.shape[2], 10)")

                                if(parameters[4]== 'SGD'):
                                    code_string += ("\noptimizer = torch.optim.SGD(regressor_model.parameters(), lr="+alpha+", weight_decay=0.00008)")

                                elif(parameters[4]== 'Adam'):
                                    code_string += ("\noptimizer = torch.optim.Adam(regressor_model.parameters(), lr="+alpha+", weight_decay=0.00008)")

                                elif(parameters[4]== 'RMSProp'):
                                    code_string += ("\noptimizer = torch.optim.RMSprop(regressor_model.parameters(), lr="+alpha+", weight_decay=0.00008)")

                                elif(parameters[4]== 'Adagrad'):
                                    code_string +=("\noptimizer = torch.optim.Adagrad(regressor_model.parameters(), lr="+alpha+", weight_decay=0.00008)")

                                elif(parameters[4]== 'Adamax'):
                                    code_string +=("\noptimizer = torch.optim.Adamax(regressor_model.parameters(), lr="+alpha+", weight_decay=0.00008)")

                                code_string +=("\nfor e in range(EPOCHS):")
                                code_string +=("\n\ttrain_epoch_loss = 0")
                                code_string +=("\n\tregressor_model.train()")
                                code_string +=("\n\tfor X_train_batch, y_train_batch in train_loader:")
                                code_string +=("\n\t\tX_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)")
                                code_string +=("\n\t\toptimizer.zero_grad()")
                                code_string +=("\n\t\ty_train_pred = regressor_model(X_train_batch,BATCH_SIZE)")

                                code_string +=("\n\t\ttrain_loss = loss_func(y_train_pred, y_train_batch.unsqueeze(1))")
                                code_string +=("\n\t\ttrain_loss.backward()")
                                code_string +=("\n\t\toptimizer.step()")

                                code_string +=("\n\t\ttrain_epoch_loss += train_loss.item()")
                                code_string +=("\n\twith torch.no_grad():")
                                code_string +=("\n\t\tval_epoch_loss = 0")
                                code_string +=("\n\t\tregressor_model.eval()")
                                code_string +=("\n\t\tfor X_val_batch, y_val_batch in val_loader:")
                                code_string +=("\n\t\t\tX_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)")
                                code_string +=("\n\t\t\ty_val_pred = regressor_model(X_val_batch,16)")


                                code_string +=("\n\t\t\tval_loss = loss_func(y_val_pred, y_val_batch.unsqueeze(1))")
                                code_string +=("\n\t\t\tval_epoch_loss += val_loss.item()")
                                code_string +=("\n\tprint(train_epoch_loss/len(train_loader))")
                                code_string +=("\n\tprint(val_epoch_loss/len(val_loader))")

                                code_string +=("\ny_pred_list = []")
                                code_string +=("\nfor X_test_batch, y_test_batch in test_loader:")
                                code_string +=("\n\tX_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)")
                                code_string +=("\n\ty_test_pred = regressor_model(X_test_batch,1)")
                                code_string +=("\n\ty_pred_list.append(y_test_pred)")
                                code_string +=("\ny_pred_inv = y_transformer.inverse_transform(pd.DataFrame(y_pred_list))")

                                py_out.write(code_string)

        #py_out.close()
        #with open('debug.json', 'w') as fp:
            #json.dump(str(code_string), fp)

    except Exception as e:
        with open('debug.json', 'w') as fp:
            json.dump(str(e), fp)



class Regressor_LSTM(nn.Module):
    def __init__(self, layers, neurons, dropouts, input_dim, seq_len):
        super(Regressor_LSTM, self).__init__()
        self.layers = int(layers)
        self.neurons = neurons
        self.dropouts = dropouts
        self.input_dim = input_dim
        #self.batch_size = batch
        self.seq_length = seq_len
        self.out = 0

        self.IP_Layer = nn.LSTM(input_size=self.input_dim,hidden_size=int(self.neurons[0]),dropout=float(dropouts[0]))

        self.Out_Layer = nn.Linear(int(self.neurons[self.layers-1]),1,bias=True)
        #self.relu = nn.ReLU()

    #def hidden_states(b):
        #self.hidden_state = [each.detach() for each in self.hidden_state]


    def forward(self,X, batch):

        #X = X.view(self.seq_length,self.batch_size,self.input_dim)
        #hidden_states(batch)
        self.hidden_state=[]
        for i in range(self.layers):
            h_x = Variable(torch.zeros(1,batch, int(self.neurons[i])))
            c_x = Variable(torch.zeros(1, batch, int(self.neurons[i])))
            hidden = (h_x,c_x)
            hidden = [each.detach() for each in hidden]
            self.hidden_state.append(hidden)

        for i in range(self.layers):
            if(i==0):
                self.out, self.hidden_state[i] = self.IP_Layer(X.view(self.seq_length,len(X),-1),self.hidden_state[i])
            else:
                self.regress = nn.LSTM(input_size=int(self.neurons[i-1]),hidden_size=int(self.neurons[i]),dropout=float(self.dropouts[i]))
            #h_1 = Variable(torch.zeros(1, self.batch_size, self.neurons[i]))
            #c_1 = Variable(torch.zeros(1, self.batch_size, self.neurons[i]))
                self.out, self.hidden_state[i] = self.regress(self.out.view(self.seq_length,len(X),-1), self.hidden_state[i])
            self.hidden_state[i] = [each.detach() for each in self.hidden_state[i]]
         #self.relu(out)
        #out = out.view(-1, int(self.neurons[self.layers-1]))
        self.out = self.Out_Layer(self.out.view(self.seq_length,len(X),int(self.neurons[self.layers-1]))[-1])
        #self.out = self.ReLU(self.out)

        return self.out


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def data_preprocessing(file, parameters):
    # Importing the dataset
    try:
        input_frame = pd.read_json(file)
        target = parameters[11]

        if(type(parameters[10])=='str'):
            temp = parameters[10]
            parameters[10] = list()
            parameters[10].append(temp)

    except Exception as e:
        with open('debug1.json', 'w') as fp:
            json.dump(str(e), fp)
    with open('debug.json', 'w') as fp:
        json.dump(str(target), fp)

    ##print(input_frame.info())
        #column_list = list(self.dataset.columns)
    #input_frame.drop(target,1)
    #print(parameters[1])
    if(parameters[1]=='Time Series'):

        try:
            X = input_frame
            y = input_frame[target]

            #print(input_frame.head())
            #print(X,y)
            inx = list(X.columns)
            column_list = parameters[10]
            #print(column_list)
            drop = True
            #print(column_list)
            for column1 in inx:
                for column2 in column_list:
                    if(column1==column2):
                        drop = False
                        break
                if(drop==True):
                    X = X.drop(column1, axis=1)
                drop = True

            test_size = int(len(X) * float(parameters[3]))
            train_size = len(X) - test_size
            train, test = X.iloc[0:train_size], X.iloc[train_size:len(X)]

            global y_transformer
            global f_transformer

            y_transformer = y_transformer.fit(train[[target]])
            y_trn = y_transformer.transform(train[[target]])
            #print(y_trn.shape)
            y_tst = y_transformer.transform(test[[target]])
            #print(y_tst.shape)

            #f_transformer = RobustScaler()
            f_transformer = f_transformer.fit(train[column_list].to_numpy())
            train.loc[:, column_list] = f_transformer.transform(train[column_list].to_numpy())
            test.loc[:, column_list] = f_transformer.transform(test[column_list].to_numpy())

            time_steps = 10
            # reshape to [samples, time_steps, n_features]
            y_trn = pd.DataFrame(y_trn)
            y_tst = pd.DataFrame(y_tst)
            X_train, y_train = create_dataset(train, y_trn, time_steps)
            X_test, y_test = create_dataset(test, y_tst, time_steps)
            #print(X_train.shape,y_train.shape)

            return(X_train, X_test, y_train, y_test)

        except Exception as e:
            with open('debug.json', 'w') as fp:
                json.dump(str(e), fp)


    if(parameters[1]=='Classification'):
        try:
            X = input_frame.drop(target, axis = 1)
            y = input_frame[target]
            #print(input_frame.head())
            #print(X,y)
            inx = list(X.columns)
                    #print(type(parameters[10]))
                    #print(inx)
            column_list = parameters[10]
            for name in column_list:
                if(name==target):
                    column_list.remove(target)
            #with open('debug.json', 'w') as fp:
                #json.dump(str(column_list), fp)

                    #print(column_list)
            drop = True
                    #print(column_list)
            for column1 in inx:
                for column2 in column_list:
                    if(column1==column2):
                        drop = False
                        break
                if(drop==True):
                    X = X.drop(column1, axis=1)
                            #print(type(column1))
                            #print(input_frame.head())
                            #print(type(column1))
                drop = True

            for column in column_list:
                if(X[column].dtype == 'object'):
                    X[column] = X[column].astype('category')
                    X[column] = X[column].cat.codes
            if(y.dtype == 'object'):
                y = y.astype('category')
                y = y.cat.codes
                    #print(parameters[1])
                    #print(target)
                    #print(y)
                    #print(parameters[3])
                    #print(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = float(parameters[3]), random_state = 2)

                    #print(X_train)
            sc = StandardScaler()
                    #print(X_train)
            X_train = sc.fit_transform(X_train)
                    #print(X_train)
            X_test = sc.transform(X_test)
                    #print(X_test)
                    #print(X_train.shape, y_train.shape)
            return(X_train, X_test, y_train, y_test)

        except Exception as e:
            with open('debug1.json', 'w') as fp:
                json.dump(str(e), fp)

def tf_ann(parameters):

    global loss_stats
    global w_n_b
    #print(type(parameters[10]))
    X_train, X_test, y_train, y_test = data_preprocessing('data.json', parameters)
    #print(type(X_train),type(y_train))
    with open('debug.json', 'w') as fp:
        json.dump(str(y_train.shape), fp)

    try:
        alpha = float(parameters[2])
        #print(alpha)

        layers = int(parameters[6])
        #print(layers)

        neurons = parameters[7]
        #print(neurons)

        activationfunction = parameters[8]
        #print(activationfunction)

        dropouts = parameters[13]
        #print(droputs)

        if(parameters[4]== 'SGD'):
            optimizer = keras.optimizers.SGD(learning_rate=alpha)

        elif(parameters[4]== 'Adam'):
            optimizer = keras.optimizers.Adam(learning_rate=alpha)

        elif(parameters[4]== 'Adagrad'):
            optimizer = keras.optimizers.Adagrad(learning_rate=alpha)

        elif(parameters[4]== 'RMSProp'):
            optimizer = keras.optimizers.RMSprop(learning_rate=alpha)

        elif(parameters[4]== 'Adamax'):
            optimizer = keras.optimizers.Adamax(learning_rate=alpha)

        #print(type(layers))

        # Initialising the ANN
        classifier = Sequential()

        for i in range(layers+1):
            #print(i)
            # Adding the input layer and the first hidden layer
            if(i == 0):
                #print(neurons[i])
                classifier.add(Dense(units = int(neurons[i]), kernel_initializer = 'uniform', activation = 'relu', input_dim = X_test.shape[1]))

                #print(i)
                #print(neurons[i])
            elif(i == layers):
                classifier.add(Dense(units = int(neurons[i]), kernel_initializer = 'uniform', activation = 'sigmoid'))
                #print(neurons[i])
            else:
                # Adding remaining hidden layers
                classifier.add(Dense(units = int(neurons[i]), kernel_initializer = 'uniform', activation = 'relu'))
                classifier.add(Dropout(float(dropouts[i])))
                #print(neurons[i])
            #classifier.add(Dropout(float(dropouts[i])))
        #print(classifier.summary())

                # Compiling the ANN
        classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        #print(optimizer)
        #Fitting the ANN to the Training set
        history = classifier.fit(X_train, y_train, batch_size = int(parameters[5]), epochs = 100, validation_split=float(parameters[12]))


    except Exception as e:
             with open('debug1.json', 'w') as fp:
                 json.dump(str(e), fp)
    #history = classifier.evaluate(X_test,)

    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_test, y_pred)
    #with open('predict.json', 'w') as fp:
        #json.dump(str(y_pred), fp)
    try:
        i=0
        for layer in classifier.layers:
            w_n_b['layers'].append(layer.name)
            if(layer.name.find("dropout")==-1):
                w_n_b['weights'].append(layer.get_weights()[0].tolist())
                w_n_b['biases'].append(layer.get_weights()[1].tolist())
            i= i + 1
        with open('weights.json','w') as fp :
            json.dump(w_n_b,fp)
    except Exception as e:
        with open('debug.json', 'w') as fp:
            json.dump(str(e), fp)

    loss_stats["loss"] = history.history['loss']
    loss_stats["val_loss"] = history.history['val_loss']
    loss_stats["accuracy"] = history.history['accuracy']
    loss_stats["val_accuracy"] = history.history['val_accuracy']
    loss_stats["confusion_matrix"] = np.array(pd.DataFrame(cm)).tolist()

    with open('result.json', 'w') as fp:
        json.dump(loss_stats, fp)

def tf_rnn(parameters):
    #print(X_train.shape,y_train.shape,type(parameters[5]))
    try:
        #print(parameters[1])
        #dataset_train = pd.read_json('data.json')
        X_train, X_test, y_train, y_test = data_preprocessing('data.json', parameters)

        #print('Backfromprocessing')

        global loss_stats
        alpha = float(parameters[2])
        #print(alpha)

        layers = int(parameters[6])
        #print(layers)

        neurons = parameters[7]
        #print(neurons)

        activationfunction = parameters[8]
        #print(activationfunction)

        dropouts = parameters[13]

        if(parameters[4]== 'SGD'):
            optimizer = keras.optimizers.SGD(learning_rate=alpha)

        elif(parameters[4]== 'Adam'):
            optimizer = keras.optimizers.Adam(learning_rate=alpha)

        elif(parameters[4]== 'Adagrad'):
            optimizer = keras.optimizers.Adagrad(learning_rate=alpha)

        elif(parameters[4]== 'RMSProp'):
            optimizer = keras.optimizers.RMSprop(learning_rate=alpha)

        elif(parameters[4]== 'Adamax'):
            optimizer = keras.optimizers.Adamax(learning_rate=alpha)

        #print(optimizer)
        # Part 2 - Building the RNN
        # Initialising the RNN
        regressor = Sequential()

        for i in range(layers):
            #print(i)
            if(i == 0):
                #print(neurons[i])
                # Adding the first LSTM layer and some Dropout regularisation
                # datapoint,timesteps,dimensions
                #print(X_train.shape)
                if(layers-1 == 0):
                    regressor.add(LSTM(units = int(neurons[i]), return_sequences = False, input_shape = (X_train.shape[1], X_train.shape[2])))
                else :
                    regressor.add(LSTM(units = int(neurons[i]), return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))

                regressor.add(Dropout(float(dropouts[i])))
                #print(neurons[i])
            elif(i == layers-1):
                regressor.add(LSTM(units = int(neurons[i]), return_sequences = False))
                regressor.add(Dropout(float(dropouts[i])))
                #print(neurons[i])
            else:
                # Adding remaining hidden layers
                regressor.add(LSTM(units = int(neurons[i]), return_sequences = True ))
                regressor.add(Dropout(float(dropouts[i])))
                #regressor.add(Dropout(0.2))
                #print(neurons[i]
        #print(dropouts)

        # Adding the output layer
        regressor.add(Dense(units = 1))
        #print(regressor)
        # Compiling the RNN
        regressor.compile(optimizer = optimizer, loss = 'mean_squared_error')
        #print(regressor)
        # Fitting the RNN to the Training set

        history = regressor.fit(X_train, y_train, validation_split=float(parameters[12]), epochs=100, batch_size = int(parameters[5]))
    except Exception as e:
        with open('debug.json', 'w') as fp:
            json.dump(str(e), fp)
        #print(parameters[5])
    try:
        i=0
        for layer in regressor.layers:
            w_n_b['layers'].append(layer.name)
            if(layer.name.find("dropout")==-1):
                w_n_b['weights'].append(layer.get_weights()[0].tolist())
                w_n_b['biases'].append(layer.get_weights()[1].tolist())
            i= i + 1
        with open('weights.json','w') as fp :
            json.dump(w_n_b,fp)
    #print(json.dumps(str(history.history)))
        loss_stats["loss"] = history.history['loss']
        loss_stats["val_loss"] = history.history['val_loss']
        #loss_stats['accuracy'] = history.history['accuracy']
        #loss_stats['val_accuracy'] = history.history['val_accuracy']

        global y_transformer
        y_pred = regressor.predict(X_test)
        loss_stats["y_train_inv"] = y_transformer.inverse_transform(y_train.reshape(1, -1)).tolist()
        loss_stats["y_test_inv"] = y_transformer.inverse_transform(y_test.reshape(1, -1)).tolist()
        loss_stats["y_pred_inv"] = y_transformer.inverse_transform(y_pred).tolist()
    except Exception as e:
        with open('debug1.json', 'w') as fp:
            json.dump(str(e), fp)
    with open('result.json', 'w') as fp:
        json.dump(loss_stats, fp)


def pyt_preprocessing(file, parameters):
    try:
        input_frame = pd.read_json(file)
        target = parameters[11]

        if(type(parameters[10])=='str'):
            temp = parameters[10]
            parameters[10] = list()
            parameters[10].append(temp)


        X = input_frame.drop(target, axis = 1)
        y = input_frame[target]
        inx = list(X.columns)
        #print(type(parameters[10]))
        #print(inx)
        column_list = parameters[10]
        for name in column_list:
            if(name==target):
                column_list.remove(target)
        #print(column_list)
        drop = True
        #print(column_list)
        for column1 in inx:
            for column2 in column_list:
                if(column1==column2):
                    drop = False
                    break
            if(drop==True):
                X = X.drop(column1, axis=1)
            drop = True

        for column in column_list:
            if(X[column].dtype == 'object'):
                X[column] = X[column].astype('category')
                X[column] = X[column].cat.codes
        if(y.dtype == 'object'):
            y = y.astype('category')
            y= y.cat.codes

        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=float(parameters[3]), random_state = 0)
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=float(parameters[12]), random_state = 0)

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_val = sc.transform(X_val)
        X_test = sc.transform(X_test)

        X_train , y_train = np.array(X_train), np.array(y_train)
        X_val, y_val = np.array(X_val), np.array(y_val)
        X_test, y_test = np.array(X_test), np.array(y_test)

        return(X_train , y_train , X_val, y_val, X_test, y_test)

    except Exception as e:
        with open('debug.json', 'w') as fp:
            json.dump(str(e), fp)


class Dataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)

def pyt_ANN(parameters):
    #print(parameters[0],parameters[1])
    target = parameters[11]
    alpha = float(parameters[2])
    #print(alpha)

    layers_1 = int(parameters[6]) + 1
    #print(layers)

    neurons = parameters[7]
    #print(neurons)

    activationfunction = parameters[8]
    #print(activationfunction)

    dropouts = parameters[13]

    global loss_stats

    X_train , y_train , X_val, y_val, X_test, y_test = pyt_preprocessing('data.json', parameters)

    EPOCHS = 100
    BATCH_SIZE = int(parameters[5])

    try:
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        y_val = y_val.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.float32)
        train_dataset = Dataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        val_dataset = Dataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
        test_dataset = Dataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)

        #try :
        layers = []
        layers.append(nn.Linear(X_train.shape[1],int(neurons[0]),bias=True))
        #layers.append(nn.ReLU())
        for i in range(layers_1):
            #if(i == 0):
            if(i == layers_1-1):
                layers.append(nn.Linear(int(neurons[i]),1,bias=True))
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.Linear(int(neurons[i]),int(neurons[i+1]),bias=True))
                #layers.append(nn.ReLU())
                layers.append(nn.Dropout(float(dropouts[i])))
        model = nn.Sequential(*layers)

        layers.clear()


        #except Exception as e:


        #NUM_FEATURES = len(X.columns)
        #try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss_func = nn.BCELoss()
        #loss_func = nn.BCEWithLogitsLoss()
            #loss_func = nn.MSELoss()
            #learning_rate = 0.0015

        if(parameters[4]== 'SGD'):
            optimizer = torch.optim.SGD(model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters[4]== 'Adam'):
            optimizer = torch.optim.Adam(model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters[4]== 'RMSProp'):
            optimizer = torch.optim.RMSprop(model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters[4]== 'Adagrad'):
            optimizer = torch.optim.Adagrad(model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters[4]== 'Adamax'):
            optimizer = torch.optim.Adamax(model.parameters(), lr=alpha, weight_decay=0.00008)


        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        y_pred = []
        y_actual = []
        for e in range(EPOCHS):
            train_epoch_loss = 0
            val_epoch_loss = 0
            train_epoch_acc = 0
            val_epoch_acc = 0

            model.train()
            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)

                y_train_pred = model(X_train_batch)
                optimizer.zero_grad()
                y_actual.append(y_train_batch.unsqueeze(1))
                train_loss = loss_func(y_train_pred, y_train_batch.unsqueeze(1))
                train_loss.backward()
                optimizer.step()

                #y_pred_tag = torch.round(torch.sigmoid(y_train_pred)
                y_pred_tag = (y_train_pred > 0.5).float()
                acc = ((y_pred_tag == y_train_batch.unsqueeze(1)).sum().float())/y_train_batch.shape[0]
                train_epoch_loss += train_loss.item()
                train_epoch_acc += acc.item()

                # VALIDATION
            with torch.no_grad():

                model.eval()
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                    y_val_pred = model(X_val_batch)

                    val_loss = loss_func(y_val_pred, y_val_batch.unsqueeze(1))
                    val_epoch_loss += val_loss.item()

                    #y_pred_tag = torch.round(torch.sigmoid(y_val_pred))
                    y_pred_tag = (y_val_pred > 0.5).float()
                    acc = ((y_pred_tag == y_val_batch.unsqueeze(1)).sum().float())/y_val_batch.shape[0]
                    val_epoch_acc += acc.item()



            loss_stats["loss"].append(train_epoch_loss/len(train_loader))
            loss_stats["val_loss"].append(val_epoch_loss/len(val_loader))

            loss_stats["accuracy"].append(train_epoch_acc/len(train_loader))
            loss_stats["val_accuracy"].append(val_epoch_acc/len(val_loader))


        y_pred_list = []
        y_test_list = []
        model.eval()
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            y_pred = model(X_test_batch)
            #y_pred = torch.round(torch.sigmoid(y_pred))
            y_pred_tag = (y_pred > 0.5).float()
            y_pred_list.append(y_pred)
            y_test_list.append(y_test_batch)


        y_pred_list = [a.squeeze().tolist() for a in y_pred_list ]
        y_pred_list = [round(a) for a in y_pred_list]
        y_test_list = [a.squeeze().tolist() for a in y_test_list ]

        cm = confusion_matrix(y_test_list, y_pred_list)
        #check the data-types and convert lists to numpy arrays

        loss_stats["confusion_matrix"] = np.array(pd.DataFrame(cm)).tolist()


    except Exception as e:
        with open('debug1.json', 'w') as fp:
            json.dump(str(e), fp)

    with open('result.json', 'w') as fp:
        json.dump(loss_stats, fp)

def pyt_RNN(parameters):
    input_frame = pd.read_json('data.json')
    target = parameters[11]
    if(type(parameters[10])=='str'):
        temp = parameters[10]
        parameters[10] = list()
        parameters[10].append(temp)

    X = input_frame
    y = input_frame[target]
    alpha = float(parameters[2]) # alpha : learning_rate
    #print(alpha)

    layers_1 = int(parameters[6])
    #print(layers)

    neurons = parameters[7]
    #print(neurons)

    activationfunction = parameters[8]
    #print(activationfunction)

    dropouts = parameters[13]
    global loss_stats

    #print(input_frame.head())
    #print(X,y)
    with open('debug2.json', 'a') as fp:
        json.dump(str(X.columns), fp)

        inx = list(X.columns)
    #print(type(parameters[10]))
    #print(inx)
        column_list = parameters[10]
        json.dump(str(column_list), fp)
    #print(column_list)
    drop = True
    #print(column_list)
    for column1 in inx:
        for column2 in column_list:
            if(column1==column2):
                drop = False
                break
        if(drop==True):
            X = X.drop(column1, axis=1)
            #print(type(column1))
            #print(input_frame.head())
            #print(type(column1))
        drop = True
    #X = X.filter(column_list, axis=1)
    #with open('debug.json', 'w') as fp:
        #json.dump(str(drop), fp)
    with open('debug2.json', 'a') as fp:
        json.dump(str(X.columns), fp)

    test_size = int(len(X) * float(parameters[3]))
    train_size = len(X) - test_size
    train, test = X.iloc[0:train_size], X.iloc[train_size:len(X)]
    val_size = int(train_size * float(parameters[12]))
    train_size = train_size - val_size
    train,val = train.iloc[0:train_size], train.iloc[train_size: (train_size+val_size)]

    #print(test.shape)
    #print(type(target))
    try:
        #y_transformer = RobustScaler()
        global y_transformer
        global f_transformer
        y_transformer = y_transformer.fit(train[[target]])
        y_trn = y_transformer.transform(train[[target]])
        #print(y_trn.shape)
        y_tst = y_transformer.transform(test[[target]])
        #print(y_tst.shape)
        y_val = y_transformer.transform(val[[target]])

        #f_transformer = RobustScaler()
        f_transformer = f_transformer.fit(train[column_list].to_numpy())
        train.loc[:, column_list] = f_transformer.transform(train[column_list].to_numpy())
        test.loc[:, column_list] = f_transformer.transform(test[column_list].to_numpy())
        val.loc[:, column_list] = f_transformer.transform(val[column_list].to_numpy())
        loss_stats["y_train_inv"] = y_transformer.inverse_transform(y_trn.reshape(1,-1)).tolist()
        ytrain_inv = y_transformer.inverse_transform(y_val.reshape(1,-1)).tolist()
        loss_stats["y_train_inv"][0].extend(ytrain_inv[0])
        loss_stats["y_test_inv"] = y_transformer.inverse_transform(y_tst.reshape(1,-1)).tolist()

        time_steps = 10
        # reshape to [samples, time_steps, n_features]
        y_trn = pd.DataFrame(y_trn)
        y_tst = pd.DataFrame(y_tst)
        y_val = pd.DataFrame(y_val)
        X_train, y_train = create_dataset(train, y_trn, time_steps)
        X_test, y_test = create_dataset(test, y_tst, time_steps)
        X_val, y_val = create_dataset(val, y_val, time_steps)
    except Exception as e:
        with open('debug.json', 'w') as fp:
            json.dump(str(e), fp)


    EPOCHS = 100
    BATCH_SIZE = int(parameters[5])

    train_dataset = Dataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_dataset = Dataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    test_dataset = Dataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE,shuffle=False, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False )
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #loss_func = nn.BCELoss()
        loss_func = nn.MSELoss()
        #learning_rate = 0.0015
        #model = nn.ModuleList()

        #for i in range(layers_1):
            #if(i==0):
                #model.append(nn.LSTM(input_size=X_train.shape[2],hidden_size=int(neurons[0]),dropout=float(dropouts[i])))

        regressor_model = Regressor_LSTM(layers_1, neurons, dropouts, X_train.shape[2], 10)
        with open('debug.json', 'w') as fp:
            json.dump(str(X_train.shape), fp)

        if(parameters[4]== 'SGD'):
            optimizer = torch.optim.SGD(regressor_model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters[4]== 'Adam'):
            optimizer = torch.optim.Adam(regressor_model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters[4]== 'RMSProp'):
            optimizer = torch.optim.RMSprop(regressor_model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters[4]== 'Adagrad'):
            optimizer = torch.optim.Adagrad(regressor_model.parameters(), lr=alpha, weight_decay=0.00008)

        elif(parameters[4]== 'Adamax'):
            optimizer = torch.optim.Adamax(regressor_model.parameters(), lr=alpha, weight_decay=0.00008)
        with open('model.json', 'w') as fp:
            json.dump(str(regressor_model.parameters()), fp)
    except Exception as e:
        with open('debug_hyp.json', 'w') as fp:
            json.dump(str(e), fp)

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        for e in range(EPOCHS):
            train_epoch_loss = 0
            #hidden_state = [each.detach() for each in hidden]
            #global regressor_model
            regressor_model.train()
            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
                optimizer.zero_grad()
                #X_train_batch = X_train_batch.permute(1,0,2)
                with open('debug.json', 'w') as fp:
                    json.dump(str(X_train_batch.shape), fp)

                y_train_pred = regressor_model(X_train_batch,BATCH_SIZE)

                #hidden = [each.detach() for each in hidden]

                #hidden = Variable(hidden.data, requires_grad=True)
                #hidden = hidden.detach()

                train_loss = loss_func(y_train_pred, y_train_batch.unsqueeze(1))
                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item()
                #n_correct += (t2orch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
                #n_total += batch.batch_size
                #train_acc = n_correct/n_total

                # VALIDATION
            with torch.no_grad():
                val_epoch_loss = 0

                regressor_model.eval()
                for X_val_batch, y_val_batch in val_loader:
                    #h_x = Variable(torch.zeros(1, BATCH_SIZE, int(neurons[0])))
                    #c_x = Variable(torch.zeros(1, BATCH_SIZE, int(neurons[0])))
                    #hidden = (h_x,c_x)

                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                    #X_val_batch = X_val_batch.permute(1,0,2)
                    y_val_pred = regressor_model(X_val_batch,BATCH_SIZE)


                    val_loss = loss_func(y_val_pred, y_val_batch.unsqueeze(1))
                    val_epoch_loss += val_loss.item()
            loss_stats["loss"].append(train_epoch_loss/len(train_loader))
            loss_stats["val_loss"].append(val_epoch_loss/len(val_loader))

        y_pred_list = []
        #y_test_list = []
        for X_test_batch, y_test_batch in test_loader:
            #h_x = Variable(torch.zeros(1, 1, int(neurons[0])))
            #c_x = Variable(torch.zeros(1, 1, int(neurons[0])))
            #hidden = (h_x,c_x)

            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            #X_test_batch = X_test_batch.permute(1,0,2)
            y_test_pred = regressor_model(X_test_batch,1)
            y_pred_list.append(y_test_pred)
            #y_test_list.append(y_test_batch)
        loss_stats["y_pred_inv"] = y_transformer.inverse_transform(pd.DataFrame(y_pred_list)).tolist()
        #loss_stats["y_test_inv"] = y_transformer.inverse_transform(y_test_inv.reshape(1,-1)).tolist()

    except Exception as e:
        with open('debug1.json', 'w') as fp:
            json.dump(str(e), fp)

    with open('result.json', 'w') as fp:
        json.dump(loss_stats, fp)

#Read data from stdin
def read_in():
    lines = sys.stdin.readlines()
    return json.loads(lines[0])

def main():
    #get our data as an array from read_in()

    #recieving data from the nodejs server and storing it in a JSON file
    lines = read_in()
    dict = lines[9]
    data = json.dumps(dict)
    f = open("data.json","w")
    f.write(data)
    f.close()

    #parameter[0] = Framework/libraries
    #parameter[1] = NN_type
    #parameter[2] = learning_rate
    #parameter[3] & parameter[4] = test_split and optimizer
    #parameter[5] = training_batch_size
    #parameter[6] = no of layers
    #parameter[7] = no of neurons in every layer




    if(lines[0]=='Keras'):
        if(lines[1]=='Classification'):
            tf_ann(lines)
            #print(lines[5])
        if(lines[1]=='Time Series'):
            tf_rnn(lines)
    #if(lines[0]=='Gluon'):
        #if(lines[1]=='Classification'):
            #tf_ann(lines)
        #if(lines[1]=='Time Series'):
            #tf_rnn(lines)
    if(lines[0]=='PyTorch'):
        if(lines[1]=='Classification'):
            pyt_ANN(lines)
        if(lines[1]=='Time Series'):
            pyt_RNN(lines)

    coder(lines)

#start process
if __name__ == '__main__':
    main()
