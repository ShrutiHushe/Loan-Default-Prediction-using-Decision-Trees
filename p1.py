#import lib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

#load the data
data = pd.read_csv("loan_march24.csv")
print(data)

#feature and target
features = data[["GENDER", "OCCUPATION"]]
target = data["DEFAULT"]

#handle cat data
cfeatures = pd.get_dummies(features)
print(features)
print(cfeatures)

#model
model=DecisionTreeClassifier()
mf = model.fit(cfeatures.values, target)

#predict
gender = int(input("1 for female and 2 for male"))
if gender == 1:
	d1 = [1, 0]
else :
	d1 = [0, 1]

occ = int(input("1 for business and 2 for salary "))
if occ == 1:
	d2 = [1, 0]
else:
	d2 = [0, 1]

d = [d1 + d2]
ans = model.predict(d)
print(ans)

#internal working
plt.figure(figsize=(12, 5))
plot_tree(mf, feature_names=["gender", "gender", "occupation", "occupation"], filled=True, class_names=["loan de de", "loan mat de"], fontsize=12)
plt.show()