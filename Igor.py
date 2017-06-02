
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import gzip
import seaborn as sns
from pandas.io.json import json_normalize
get_ipython().magic(u'matplotlib inline')


# # Data Cleaning

# ## Open Catalog dataset

# In[2]:


catalog = pd.read_csv('data/catalog.gz')


# In[3]:


catalog.head()


# ## Open dataset with a JSON object per line with NaN

# In[4]:


get_ipython().run_cell_magic(u'time', u'', u"with gzip.open('data/test.gz', 'r') as f:\n    json_records = []\n    for line in f:\n        record = json.loads(line)\n        json_records.append(record)\ndata = pd.DataFrame(json_records)")


# In[5]:


data.head()


# ## Open only purchase events from data dataset

# In[6]:


get_ipython().run_cell_magic(u'time', u'', u"with open('data/purchase_data', 'r') as f:\n    json_records = []\n    for line in f:\n        record = json.loads(line)\n        json_records.append(record)\npurchase = pd.DataFrame(json_records)")


# In[7]:


purchase.head()


# ## Unnest products

# In[8]:


get_ipython().run_cell_magic(u'time', u'', u"with open('data/purchase_data', 'r') as f:\n    json_records = []\n    for line in f:\n        record = json.loads(line)\n        products = record.pop('products')\n        record.pop('source')\n        record.pop('event_type')\n        for obj in products:\n            new_record = record.copy()\n            new_record.update(obj)\n            json_records.append(new_record)\npurchase = pd.DataFrame(json_records)")


# In[9]:


purchase.head()


# ### Using json_normalize

# In[10]:


get_ipython().run_cell_magic(u'time', u'', u"with open('data/purchase_data', 'r') as f:\n    json_records = []\n    for line in f:\n        record = json.loads(line)\n        json_records.append(record)\npurchase = json_normalize(json_records, 'products', ['uid','date','gender'])")


# In[11]:


purchase.head()


# In[12]:


purchase = purchase.join(catalog.set_index('pid'), on='pid')


# In[13]:


purchase.info()


# In[14]:


categorical = ['category', 'sub_category', 'sub_sub_category','gender']
new_purchase = pd.get_dummies(purchase, columns=categorical)
new_purchase.info()


# In[15]:


new_purchase.head()


# In[16]:


dummy_cols = ['category_c1bd5fd999bd577743936898ada96496b547af3c',
'sub_category_f08770a96fb546673053ab799f5ea9cada06c06a',
'sub_sub_category_2d2c44a2d8f18a6271f0e8057313af68a46d0f24',
'gender_F']
purchase2 = new_purchase.drop(dummy_cols, 1)
purchase2.info()


# ---
# # Machine Learning Algorithms

# In[40]:


from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve


# ## Separate train and test data

# In[18]:


features = purchase2.columns[6:-1]
target = purchase2.columns[-1]


# ## Gambiarra : nem todas as features tão no target

# In[19]:


get_ipython().run_cell_magic(u'time', u'', u"with open('data/purchase_target', 'r') as f:\n    json_records = []\n    for line in f:\n        record = json.loads(line)\n        json_records.append(record)\npurchase3 = json_normalize(json_records, 'products', ['uid','date'])\npurchase3 = purchase3.join(catalog.set_index('pid'), on='pid')\ncategorical = ['category', 'sub_category', 'sub_sub_category']\nnew_purchase2 = pd.get_dummies(purchase3, columns=categorical)\ndummy_cols = ['category_c1bd5fd999bd577743936898ada96496b547af3c',\n'sub_category_f08770a96fb546673053ab799f5ea9cada06c06a',\n'sub_sub_category_2d2c44a2d8f18a6271f0e8057313af68a46d0f24']\npurchase4 = new_purchase2.drop(dummy_cols, 1)\nfeatures2 = purchase4.columns[6:]")


# In[20]:


new_features = list(set(features)&set(features2))


# In[21]:


X = purchase2[new_features]
Y = purchase2[target]


# In[22]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)


# In[23]:


len(x_train.columns)


# ## Naive Bayes

# In[24]:


get_ipython().run_cell_magic(u'time', u'', u'gnb = GaussianNB()\ngnb.fit(x_train, y_train)\ny_pred = gnb.predict(x_test)\nscore = metrics.accuracy_score(y_test, y_pred)')


# In[25]:


score


# ## KNeighborsClassifier

# In[19]:


get_ipython().run_cell_magic(u'time', u'', u'knn = KNeighborsClassifier(n_neighbors=1)\nknn.fit(x_train, y_train)\ny_pred = knn.predict(x_test)\nscore = metrics.accuracy_score(y_test, y_pred)')


# In[20]:


score


# ## Decision Tree

# In[26]:


get_ipython().run_cell_magic(u'time', u'', u'tree = DecisionTreeClassifier()\ntree.fit(x_train, y_train)\ny_pred = tree.predict(x_test)\nscore = metrics.accuracy_score(y_test, y_pred)')


# In[27]:


score


# ## Forest Tree

# In[43]:


get_ipython().run_cell_magic(u'time', u'', u'forest = RandomForestClassifier(n_estimators=100)\nforest.fit(x_train, y_train)\ny_pred = tree.predict(x_test)\nscore = metrics.accuracy_score(y_test, y_pred)')


# In[44]:


score


# ## xgboost

# In[35]:


get_ipython().run_cell_magic(u'time', u'', u'boost = GradientBoostingClassifier()\nboost.fit(x_train, y_train)\ny_pred = boost.predict(x_test)\nscore = metrics.accuracy_score(y_test, y_pred)')


# In[36]:


score


# ## SGDClassifier

# In[31]:


clf = linear_model.SGDClassifier()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
score = metrics.accuracy_score(y_test, y_pred)


# In[32]:


score


# ## Logistic Regression

# In[38]:


get_ipython().run_cell_magic(u'time', u'', u'clf = linear_model.LogisticRegression(C=1e5)\nclf.fit(x_train, y_train)\ny_pred = clf.predict(x_test)\nscore = metrics.accuracy_score(y_test, y_pred)')


# In[39]:


score


# ## SVM

# In[49]:


get_ipython().run_cell_magic(u'time', u'', u'svc = LinearSVC(C=1.0)\nsvc.fit(x_train, y_train)\ny_pred = svc.predict(x_test)\nscore = metrics.accuracy_score(y_test, y_pred)')


# In[50]:


score


# ---
# # Generate output

# In[25]:


X = purchase4[new_features]
answer = tree.predict(X)


# In[27]:


len(answer)


# In[32]:


users = purchase4.uid.values
len(users)


# In[33]:


len(set(users))


# In[37]:


ans = []
for i,u in enumerate(users):
    if answer[i]:
        g = 'M'
    else:
        g = 'F'
    obj = {'a':u, 'b':g}
    ans.append(obj)


# In[39]:


import csv

with open('ans.csv', 'wb') as f:
    w = csv.DictWriter(f, fieldnames=['a','b'])
    for obj in ans:
        w.writerow(obj)


# In[ ]:


with open('ans79.csv', 'r') as f:
    w = csv.DictWriter(f, fieldnames=['a','b'])
    for obj in ans:
        w.writerow(obj)

