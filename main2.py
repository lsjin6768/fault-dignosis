import torch
import numpy as np
from scipy import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

torch.manual_seed(1)

# mat_file=io.loadmat('YYYY (1).mat')
# data=mat_file['sum_data']
# data=np.delete(data,204,axis=1)
# y=data[:,204]
# data=np.delete(data,204,axis=1)


# mat_file=io.loadmat('../../Downloads/data_for_train_1000_1000.mat')
# data=mat_file['train_data']
# y=data[:,204]
# data=np.delete(data,204,axis=1)

mat_file=io.loadmat('data_for_train_1000_1000.mat')
data=mat_file['train_data']
y=data[:,50]
data=np.delete(data,50,axis=1)

x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2,random_state=1)

sc=MinMaxScaler()
sc.fit(x_train)
x_train=sc.transform(x_train)
x_test=sc.transform(x_test)

svm_model=SVC(kernel='poly' ,C=10,gamma=1)
svm_model.fit(x_train,y_train)

y_pred = svm_model.predict(x_test)
print(y_pred)
print(y_test)
print("prediction accuracy: {:.2f}".format(np.mean(y_pred == y_test)))