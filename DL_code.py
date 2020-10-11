import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

class Dataset(Dataset):
	def __init__(self, X_data, y_data):
		self.X_data = X_data
		self.y_data = y_data
	def __getitem__(self, index):
		return self.X_data[index], self.y_data[index]
	def __len__ (self):
		return len(self.X_data)
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
input_frame = pd.read_json(file)
target = 'TARGET_5Yrs'
X = input_frame.drop(target, axis = 1)
y = input_frame[target]
column_list =['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3P Made', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV']
X = X.filter(column_list, axis=1)
for column in column_list:
	if(X[column].dtype == 'object'):
		X[column] = X[column].astype('category')
		X[column] = X[column].cat.codes
if(y.dtype == 'object'):
	y = y.astype('category')
	y = y.cat.codes
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size = 0.1, random_state = 2)

X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)

X_train , y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)
EPOCHS = 100

BATCH_SIZE = 20
train_dataset = Dataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())

val_dataset = Dataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
test_dataset = Dataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)
layers = []
layers.append(nn.Linear(X_train.shape[1],9,bias=True))
layers.append(nn.Linear(9,1,bias=True))
layers.append(nn.Dropout(0.1))
layers.append(nn.Linear(1,1,bias=True))
layers.append(nn.Sigmoid())
model = nn.Sequential(*layers)
layers.clear()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.00008)
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
		y_pred_tag = (y_train_pred > 0.5).float()
		acc = ((y_pred_tag == y_train_batch.unsqueeze(1)).sum().float())/y_train_batch.shape[0]
		train_epoch_loss += train_loss.item()
		train_epoch_acc += acc.item()
	with torch.no_grad():
		model.eval()
		for X_val_batch, y_val_batch in val_loader:
			X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
			y_val_pred = model(X_val_batch)
			val_loss = loss_func(y_val_pred, y_val_batch.unsqueeze(1))
			val_epoch_loss += val_loss.item()
			y_pred_tag = (y_val_pred > 0.5).float()
			acc = ((y_pred_tag == y_val_batch.unsqueeze(1)).sum().float())/y_val_batch.shape[0]
			val_epoch_acc += acc.item()
	print(train_epoch_loss/len(train_loader))
	print(val_epoch_loss/len(val_loader))
	print(train_epoch_acc/len(train_loader))
	print(val_epoch_acc/len(val_loader))
y_pred_list = []
y_test_list = []
model.eval()
for X_test_batch, y_test_batch in test_loader:
	X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
	y_pred = model(X_test_batch)
	y_pred_tag = (y_pred > 0.5).float()
	y_pred_list.append(y_pred)
	y_test_list.append(y_test_batch)
y_pred_list = [a.squeeze().tolist() for a in y_pred_list ]
y_pred_list = [round(a) for a in y_pred_list]
y_test_list = [a.squeeze().tolist() for a in y_test_list ]
cm = confusion_matrix(y_test_list, y_pred_list)
print(cm)