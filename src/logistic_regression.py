# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
#import data
df = pd.read_csv('data/student_data.csv')
# check data
print(df.head())
df['result'] = df['G1'].apply(lambda x: 1 if x >= 10 else 0) 
# select feature
X = df[['G2', 'failures', 'studytime']]
y = df['result']
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f" training example : {len(X_train)}")
print(f" test example : {len(X_test)}")
# train model
model = LogisticRegression()
model.fit(X_train, y_train)
# predict model
pred_y = model.predict(X_test)
# evaluate model
acc = accuracy_score(y_test, pred_y)
print(f"accuracy : {acc}")
# check confusion matrix
cm = confusion_matrix(y_test, pred_y)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('image/confusion_matrix.png')
plt.show()
# plot model graph
plt.scatter(y_test, pred_y)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Logistic Regression Classification")
plt.savefig('image/logistic_regression_classification.png')
plt.show()
# check classification report
print(classification_report(y_test, pred_y))