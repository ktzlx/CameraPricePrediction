#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

df=pd.read_csv('cam_data.csv')
df.drop(columns=['Unnamed: 0'],inplace=True)
print(df.Price.value_counts().plot(kind='bar'))

#20% testing and 80% training 
x=df.drop('Price',axis=1)
y=df['Price']

testSet = [[1,1,16,1]]
test = pd.DataFrame(testSet)

x.isna().any()
y=y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)

Price=pred[0]
if Price ==0:
    print('price= $300 - $1,233')
if Price==1:
    print('price= $1,234 - $2,466')
else:
    print('price= $2,467 - $3,000')

print('ACCURACY',accuracy_score(y_test,pred))
print(classification_report(pred,y_test,zero_division=0))
# %%
