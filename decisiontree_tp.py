import pandas as pd
from sklearn import tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

laptop = pd.read_csv('laptop_price - dataset.csv')
print(laptop.head())
laptop_data = laptop[['Company','Inches','CPU_Frequency (GHz)','RAM (GB)','Weight (kg)', 'Price (Euro)']]
print(laptop_data.head())

# predict which company made the laptop with inches, cpu parameters, ram, weight and price
X = laptop_data.iloc[:,1:].values
# companies
y = laptop_data.iloc[:,0].values

clf =  tree.DecisionTreeClassifier()

clf = clf.fit(X,y)

# check the possibilities of all companies
print(laptop_data['Company'].unique())
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=laptop_data.columns, class_names=laptop_data['Company'].unique(), filled=True)
plt.show()

# predict the company who made the new laptop

# It can predict laptop with some accuracy
new_laptop = [[15.6,2.5,8,1.86,575]]
predicted_company = clf.predict(new_laptop)
print(predicted_company)