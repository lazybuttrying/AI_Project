#%%
import torch
from lstm import LSTM
from environment import Lotto

from torch import nn, optim

import pandas as pd
import numpy as np
import yaml

lotto = Lotto()
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


df = pd.read_csv('data.csv', index_col="date")
df.drop(columns=["num"], inplace=True)

data = df.values - 1
train = data[50:]
test = data[:50]

X_train = [train[i-config["WINDOW_SIZE"]:i, :]
   for i in range(config["WINDOW_SIZE"], len(train))]
y_train = [train[i, :] for i in range(config["WINDOW_SIZE"], len(train))]



inputs = data[data.shape[0] - test.shape[0] - config["WINDOW_SIZE"] :]

X_test = [ inputs[i-config["WINDOW_SIZE"]:i, :] for i in range(config["WINDOW_SIZE"], len(inputs))]


X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = test

#%%

model = LSTM(input_size=config["WINDOW_SIZE"],
           hidden_layer_size=7, output_size=7)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
# df.head()
# df.shape
#%%
for e in range(100):

  count = 0
  old = 0
  for i in range(X_train.shape[0]-1):
    # input = [0]*101
    # for d in df.iloc[i,2:].values:
    #   input[d] = 1

    # input = torch.from_numpy(df.iloc[i,2:].values.astype(np.float32))
    input = torch.from_numpy(X_train[i].astype(np.float32)).T


    result = torch.round(model(input)*100).abs()
    # print(result)




      
    score = lotto.reward(result, X_train[i+1][-1])
    if score:
      # print(score, result, X_train[i+1][-1])
      count+=1


    # loss = criterion(result, torch.from_numpy(df.iloc[i+1,2:].values))

    # if (e + 1) % 100 == 0:
    #   print('Epoch:', '%04d' % (e + 1), 'cost =', '{:.6f}'.format(loss))

    optimizer.zero_grad()
    # loss.backward()
    optimizer.step()

  if old != count:
    print()
    print("EPOCH",e )
    print("COUNT", count )
  count =old


#%%



