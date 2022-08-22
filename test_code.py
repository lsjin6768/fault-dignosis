

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import io
from sklearn.model_selection import train_test_split


# torch.manual_seed(777)
# #mat_file=io.loadmat('report_train.mat')
# #data=mat_file['td_data']
# mat_file_2=io.loadmat('Bspec_1000_100_train')
# data_2=mat_file_2['train_data']
# y=data_2[:,50]
# #data=np.delete(data,7,axis=1)
# data_2=np.delete(data_2,50,axis=1)
#
# #sum_data=np.hstack((data,data_2))

# mat_file=io.loadmat('YYYY (1).mat')
# data=mat_file['sum_data']
# data=np.delete(data,204,axis=1)
# y=data[:,204]
# data=np.delete(data,204,axis=1)

mat_file=io.loadmat('DDDD1.mat')
data=mat_file['train_data']
y=data[:,204]
data=np.delete(data,204,axis=1)

x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.3,random_state=10)

x_train=torch.FloatTensor(x_train)
x_test=torch.FloatTensor(x_test)
y_train=torch.FloatTensor(y_train)
y_test=torch.FloatTensor(y_test)

y_train=y_train.reshape(-1,1)

model = nn.Sequential(
    nn.Linear(204,350),
    nn.ReLU(),
    nn.Linear(350,560),
    nn.ReLU(),
    nn.Linear(560,840),
    nn.ReLU(),
    nn.Linear(840,840),
    nn.ReLU(),
    nn.Linear(840,840),
    nn.ReLU(),
    nn.Linear(840,560),
    nn.ReLU(),
    nn.Linear(560,350),
    nn.ReLU(),
    nn.Linear(350,7),
    nn.ReLU(),
    nn.Linear(7, 1),
    nn.Sigmoid()
)
optimiser=optim.Adam(model.parameters(),lr=1e-4)

nb_epochs=10000
for epoch in range(nb_epochs+1):

    hx=model(x_train)

    cost=F.binary_cross_entropy(hx, y_train)

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