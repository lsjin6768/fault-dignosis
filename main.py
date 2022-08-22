import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import io
from sklearn.model_selection import train_test_split
torch.manual_seed(777)
mat_file=io.loadmat('report_train.mat')
#mat_file2=io.loadmat('Bspec_1000_100_train.mat')
#data2=mat_file2['train_data']
data=mat_file['td_data']
y=data[:,7]
data=np.delete(data,7,axis=1)
#data2=np.delete(data2,50,axis=1)
#sum_data=np.hstack((data,data2))
#print(sum_data.shape)
x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
x_train=torch.FloatTensor(x_train)
x_test=torch.FloatTensor(x_test)
y_train=torch.FloatTensor(y_train)
y_test=torch.FloatTensor(y_test)
y_train=y_train.reshape(-1,1)
model = nn.Sequential(
    nn.Linear(7,50),
    nn.ReLU(),
    nn.Linear(50,80),
    nn.ReLU(),
    nn.Linear(80,120),
    nn.ReLU(),
    nn.Linear(120,120),
    nn.ReLU(),
    nn.Linear(120,120),
    nn.ReLU(),
    nn.Linear(120,80),
    nn.ReLU(),
    nn.Linear(80,50),
    nn.ReLU(),
    nn.Linear(50,1),
    nn.Sigmoid()
)
print(model)
criterion=nn.BCEWithLogitsLoss()
optimiser=optim.Adam(model.parameters(),lr=1e-4)
nb_epochs=10000

for epoch in range(nb_epochs+1):
    hx=model(x_train)
    cost=criterion(hx, y_train)
    optimiser.zero_grad()
    cost.backward()
    optimiser.step()
    if epoch % 1000 == 0:
        prediction = hx >= torch.FloatTensor([0.5]) # 예측값이 0.5를 넘으면 True로 간주
        correct_prediction = prediction.float() == y_train # 실제값과 일치하는 경우만 True로 간주
        accuracy = correct_prediction.sum().item() / len(correct_prediction) # 정확도를 계산
        print('Epoch {:4d}/{} Cost: {:.6f}'.format( # 각 에포크마다 정확도를 출력
            epoch, nb_epochs, cost.item()
        ))
pred=(model(x_test)).int()
cnt=0
for i in range(len(pred)):
    if pred[i]==y_test[i]:
        cnt=cnt+1
acc=cnt/len(pred)
print('Test data acc={:.2f}%'.format( acc*100))