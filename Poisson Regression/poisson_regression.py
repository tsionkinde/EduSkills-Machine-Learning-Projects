import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error,mean_poisson_deviance
data={
    "advertising_budget":[10,20,30,40,50,60,70,80,90,100],
    "customers":[12,18,25,32,41,48,57,65,73,85]
}
df=pd.DataFrame(data)
print(df)
X=df[["advertising_budget"]]
y=df["customers"]
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)
model=PoissonRegressor()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Actual" , y_test.values)
print("Predicted" ,y_pred)
mae=mean_absolute_error(y_test,y_pred)
deviance=mean_poisson_deviance(y_test,y_pred)
print("Mean Absolute Error",mae)
print("Mean Poisson Deviance ",deviance)
plt.scatter(X_test,y_test,label="Actual")
plt.scatter(X_test,y_pred,label="Predicted")
plt.xlabel("Advertising Budget")
plt.ylabel("Number of Customers")
plt.title("Poisson Regression")
plt.legend()
plt.show()
new_data=pd.DataFrame({
    "advertising_budget":[110]
})
prediction=model.predict(new_data)
print("Expected customers",prediction[0])

