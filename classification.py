import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

'''
splittato dataset rispetto a led==1 e led==2
'''
data = pd.read_csv(r"C:\Users\Giovanni\Desktop\Classification\dataset\led1.csv",
                      names=["Sample",
                             "Repetition",
                             "Led",
                             "Data",
                             "Quality",
                             "FAEES",
                             "K232",
                             "K270",
                             "Acidity",
                             "Peroxide Index"],
                      skiprows=1)

bg = pd.read_csv(r"C:\Users\Giovanni\Desktop\Classification\dataset\background.csv",
                         names=["background"],
                         skiprows=1)

len_data = len(data)

background = np.array([x for x in bg["background"]])
len_background = len(background)

#nomi dei campioni
name_of_samples = []
name_of_samples = set([x for x in data["Sample"]])

features = np.empty(shape=(0,len_background))

# Sottrazione del background
index = 0;
for i in data["Data"]:
    temp_list = [float(x) for x in i.strip('][').split(',')]

    for j in range(len_background):
        temp_list[j] = temp_list[j] - background[j]
    features = np.append(features,[temp_list],axis=0)
    
targets = np.array([x for x in data["Quality"]])

# Encoding della classe target contenente i nomi delle classi

enc = LabelEncoder()
encoded_targets= enc.fit(targets)

print(encoded_targets.classes_)
target_names=encoded_targets.classes_
encoded_targets= encoded_targets.transform(targets)

# Normalizzo i dati affinchè abbia media==1 e deviazione standard==0
scaler = StandardScaler()

X = scaler.fit_transform(features)
y = encoded_targets

#Split del dataset da rivedere
train_data = X[:400]
train_targets = y[:400]

test_data = X[400:]
test_targets = y[400:]

score = []
'''
10 fold cross validation sul test set
'''
index=1
k_fold_10 = KFold(n_splits=10)
k_fold_10.get_n_splits(train_data)

for train_index, validation_index in k_fold_10.split(train_data):
    X_train, X_validation = train_data[train_index], train_data[validation_index]
    y_train, y_validation = train_targets[train_index], train_targets[validation_index]
  
    decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
    decision_tree = decision_tree.fit(X_train, y_train)
    
    score.append(decision_tree.score(X_validation, y_validation))
    
    cm = confusion_matrix(y_validation, decision_tree.predict(X_validation))
    print(f"confusion matrix cross val {index}\nscore:{decision_tree.score(X_validation, y_validation)}\n{cm}")
    index +=1
    
    print(classification_report(y_validation, decision_tree.predict(X_validation)))
    
    
score.append(decision_tree.score(test_data, test_targets))    
cm = confusion_matrix(test_targets, decision_tree.predict(test_data))
print(f"confusion matrix test data\n{cm}")

print(classification_report(y_validation, decision_tree.predict(X_validation)))


  
    
