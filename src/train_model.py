# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import seaborn as sns
#import data
df = pd.read_csv('data/student_data.csv')
# check data
print(df.head())
# select feature
x = df[['G2','failures','studytime']]
y = df['G1']
# split data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(f" training example : {len(x_train)}")
print(f" test example : {len(x_test)}")
# train model
model = LinearRegression()
model.fit(x_train,y_train)
# predict model
pred_y = model.predict(x_test)
# plot model graph
plt.scatter(y_test, pred_y, c = 'y', marker='x')
plt.xlabel('actual')
plt.ylabel('predicted')
plt.savefig('image/actual_vs_predicted.png')
plt.show()
# evaluate model
mse = mean_squared_error(y_test, pred_y)
print(f"mean squared error : {mse}")
# evaluate model
r2 = r2_score(y_test, pred_y)
print(f"r2 score : {r2}")
# check correlation matrix  
corr = df.select_dtypes(include='number').corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('image/correlation_matrix.png')
plt.show()
plt.scatter(df['studytime'],df['G1'])
plt.xlabel('studytime')
plt.ylabel('Grade')
plt.title('Study Time vs Grade')
plt.savefig('image/studytime_vs_grade.png')
plt.show()