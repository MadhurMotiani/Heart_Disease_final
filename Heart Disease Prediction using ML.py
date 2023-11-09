#!/usr/bin/env python
# coding: utf-8

# # Classification of a specific heart disease using machine learning techniques.

# #### Build a machine learning model(s), that can detect between a subject afflicted with heart disease and someone who is normal. Problems such as this are common in the healthcare field where such medical diagnoses can be made with the aid of machine learning and AI techniques, usually with much better accuracy. Hospitals and medical enterprises often employ specialists such as machine learning engineers and data scientists to carry out these tasks.

# #### Attribute Information: 
# 
# Using the 13 attributes which are already extracted, in the heart disease dataset, you are 
# expected to detect either the presence of or the absence of the heart disease in human 
# subjects. 
# There are 13 attributes: 
# 1.  age: age in years 
# 2. sex: sex (1 = male; 0 = female) 
# 3. cp: chest pain type (Value 0: typical angina, Value 1: atypical angina, Value 2: non-anginal pain, Value 3: asymptomatic) 
# 4. trestbps: resting blood pressure (in mm Hg on admission to the hospital) 
# 5. chol: serum cholesterol in mg/dl 
# 6. fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) 
# 7. restecg: resting electrocardiographic results (Value 0: normal Value, 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria)
# 8. thalach: maximum heart rate achieved 
# 9. exang: exercise induced angina (1 = yes; 0 = no) 
# 10.  oldpeak = ST depression induced by exercise relative to rest 
# 11. slope: the slope of the peak exercise ST segment (Value 0: upsloping, Value 1: flat, Value 2: downsloping) 
# 12. ca: number of major vessels (0-3) colored by flourosopy 
# 13. thal: 0 = normal; 1 = fixed defect; 2 = reversable defect and the label 
# 14.  condition: 0 = no disease, 1 = disease

# In[67]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score


# In[68]:


#Loading the CSV data to a pandas dataframe
heart_data = pd.read_csv("D:\\NMIMS\Devops bootcamp\Heart_disease\heartdisease_data.csv")


# In[69]:


heart_data.head()


# In[70]:


heart_data.tail()


# In[71]:


heart_data.shape


# ### Data Preprocessing:

# In[72]:


#Extracting information about the attributes of the data
heart_data.info()


# In[73]:


#checking for missing values
heart_data.isnull().sum()


# In[74]:


#Statistical measures about the data
heart_data.describe()


# ### Data Exploration and Visualization:

# In[75]:


#Distribution of Target Variable
heart_data['condition'].value_counts()


# In[76]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(data=heart_data, x='condition')
plt.title('Heart Disease Presence (1) vs. Absence (0)')
plt.show()


# ## Testing Multicolinearity 

# In[77]:


correlation_matrix = heart_data.corr()
print(correlation_matrix)


# In[78]:



plt.figure(figsize=(20,12))
sns.set_context('notebook',font_scale = 1)
sns.heatmap(heart_data.corr(),annot=True,linewidth =2)
plt.tight_layout()


# ### Conclusion: The indipendent variables are not colinear

# In[79]:


plt.figure(figsize=(18,9))
sns.set_context('notebook',font_scale = 1)
sns.countplot(heart_data['age'],hue=heart_data['condition'])
plt.tight_layout()


# In[80]:


plt.figure(figsize=(18,9))
sns.set_context('notebook',font_scale = 1)
sns.countplot(heart_data['cp'],hue=heart_data['condition'])
plt.tight_layout()


# In[81]:


sns.relplot(
    data=heart_data,
    x="trestbps", y="chol", col="cp", style="condition", kind='line'
)


# In[23]:


# Pairplot to visualize relationships between numerical variables
image=sns.pairplot(heart_data, hue='condition', diag_kind='kde')
plt.show()


# ## Splitting the features and the target variable

# In[82]:


X = heart_data.drop(columns='condition',axis=1)
Y = heart_data['condition']


# In[83]:


X.shape


# In[84]:


X


# In[85]:


Y.shape


# In[86]:


Y


# In[87]:


#normalizing the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(X)

# Create a DataFrame with the normalized data
normalized_X = pd.DataFrame(data_normalized, columns=X.columns)
normalized_X.head()


# In[88]:


normalized_X.describe()


# ### Splitting the data into training data and test data

# In[89]:


#Split ratio - 80:20, stratify will split data in equal proportions

X_train, X_test, Y_train, Y_test = train_test_split(normalized_X,Y, test_size=0.2, stratify=Y, random_state=2 )


# In[90]:


print(normalized_X.shape,X_train.shape, X_test.shape)


# ### FEATURE IMPORTANCE USING DECISION TREE MODEL

# In[91]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score


# In[92]:


dtclf = DecisionTreeClassifier()


# In[93]:


dtclf.fit(X_train, Y_train)


# In[94]:


#important features
importances = dtclf.feature_importances_
importances


# In[95]:


indices = np.argsort(importances)[::-1]
ind_attr_names = X_train.columns
pd.DataFrame([ind_attr_names[indices], np.sort(importances)[::-1]])


# In[96]:


# Dropping the least important columns (restecg,fbs) from the normalized_X

new_normalized_X = normalized_X.drop(axis=1, columns= ['restecg', 'fbs'])
print(new_normalized_X)


# ### Splitting the normalized and feature reduced data into training data and test data for model building

# In[97]:


#Split ratio - 80:20, stratify will split data in equal proportions

X_train, X_test, Y_train, Y_test = train_test_split(new_normalized_X,Y, test_size=0.2, stratify=Y, random_state=2 )


# In[98]:


print(new_normalized_X.shape,X_train.shape, X_test.shape)


# In[115]:


new_normalized_X.columns


# In[99]:


def evaluate_model(act, pred):
    print("Confusion Matrix \n", confusion_matrix(act, pred))
    print("Accurcay : ", accuracy_score(act, pred))
    print("Recall   : ", recall_score(act, pred))
    print("Precision: ", precision_score(act, pred))   


# ##  Model Training

# ### LOGISTIC REGRESSION

# In[100]:


model =LogisticRegression()


# In[101]:


#training the logistic regression model with Training data
model.fit(X_train,Y_train)


# ### LOGISTIC REGRESSION MODEL EVALUATION

# In[102]:


X_train_prediction = model.predict(X_train)


# In[103]:


X_test_prediction = model.predict(X_test)


# In[104]:


print("\n--Accuracy on Training Data LOGISTIC REGRESSION--\n")
evaluate_model(Y_train, X_train_prediction)
print("\n--Accuracy on Test Data LOGISTIC REGRESSION--\n")
evaluate_model(Y_test, X_test_prediction)


# ### DECISION TREE MODEL EVALUATION

# In[105]:


#training Decision Tree Classifier with new normalized data

dtclf = DecisionTreeClassifier()
dtclf.fit(X_train, Y_train)

#Predict
train_pred = dtclf.predict(X_train)
test_pred = dtclf.predict(X_test)


# In[106]:


print("\n--Accuracy on Training Data--\n")
evaluate_model(Y_train, train_pred)
print("\n--Accuracy on Test Data--\n")
evaluate_model(Y_test, test_pred)


# #### Recall is low

# In[107]:


# Display the decision tree
plt.figure(figsize=(50, 40))

from sklearn.tree import plot_tree

plot_tree(dtclf, filled=True, feature_names=ind_attr_names, class_names=["NO","YES"])
plt.show()


# #### K-Fold Cross Validation with SVM (K=10):

# In[108]:


from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Defining the number of folds (k) for cross-validation
num_folds = 10

# Initializing a KFold object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initializing SVM model
svm_classifier = SVC(kernel="linear", C=1.0)
svm_classifier.fit(X_train, Y_train)
y_pred = svm_classifier.predict(X_test)

# Lists to store accuracy scores for each fold
accuracy_scores = []

# Performing k-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Fit the model on the training data
    model.fit(X_train, Y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate accuracy and store it in the list
    accuracy = accuracy_score(Y_test, y_pred)
    accuracy_scores.append(accuracy)

# Calculate and print the average accuracy
average_accuracy = sum(accuracy_scores) / num_folds
print("K-Fold Crossvalidation  k=10 using SVM")
print(f"Accuracy: {average_accuracy}")
print('F1 Score: %.3f' % f1_score(Y_test, y_pred))


# #### K-Fold Cross Validation with SVM (K=2):

# In[109]:


# Defining the number of folds (k) for cross-validation
num_folds = 2

# Initializing a KFold object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initializing SVM model
svm_classifier = SVC(kernel="linear", C=1.0)
svm_classifier.fit(X_train, Y_train)
y_pred = svm_classifier.predict(X_test)

# Lists to store accuracy scores for each fold
accuracy_scores = []

# Performing k-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Fit the model on the training data
    model.fit(X_train, Y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate accuracy and store it in the list
    accuracy = accuracy_score(Y_test, y_pred)
    accuracy_scores.append(accuracy)

# Calculate and print the average accuracy
average_accuracy = sum(accuracy_scores) / num_folds
print("K-Fold Crossvalidation  k=2 using SVM")
print(f"Accuracy: {average_accuracy}")
print('F1 Score: %.3f' % f1_score(Y_test, y_pred))


# ## RANDOM FOREST CLASSIFIER

# In[110]:


from sklearn.ensemble import RandomForestClassifier

# Create and train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, Y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy}")


# ### Random Forest Classifier evaluation using ROC curve

# ##### The ROC curve is generated by plotting the TPR on the y-axis and the FPR on the x-axis for various threshold values. Each point on the curve represents the trade-off between sensitivity and specificity at a particular threshold. A diagonal line (the "no information" or random classifier line) is often shown on the ROC plot, and good classification models should be above this line.

# In[111]:


from sklearn.metrics import roc_curve, roc_auc_score, auc
fpr, tpr, thresholds = roc_curve(Y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# ## Logistic regression is the most optimum model with the given parameters

# # Creating Pickle File

# In[113]:


import pickle
pickle.dump(model,open('D:\\NMIMS\\Devops bootcamp\\heart_disease_prediction_model\\heart_disease_ML.pkl','wb'))


# In[ ]:




