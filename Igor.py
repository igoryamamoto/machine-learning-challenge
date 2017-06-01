
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import json
import gzip
import seaborn as sns
from pandas.io.json import json_normalize
get_ipython().magic(u'matplotlib inline')


# # Data Cleaning

# ## Open Catalog dataset

# In[15]:


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

# In[42]:


get_ipython().run_cell_magic(u'time', u'', u"with open('data/purchase_data', 'r') as f:\n    json_records = []\n    for line in f:\n        record = json.loads(line)\n        products = record.pop('products')\n        record.pop('source')\n        record.pop('event_type')\n        for obj in products:\n            new_record = record.copy()\n            new_record.update(obj)\n            json_records.append(new_record)\npurchase = pd.DataFrame(json_records)")


# In[43]:


purchase.head()


# ### Using json_normalize

# In[14]:


get_ipython().run_cell_magic(u'time', u'', u"with open('data/purchase_data', 'r') as f:\n    json_records = []\n    for line in f:\n        record = json.loads(line)\n        json_records.append(record)\npurchase = json_normalize(json_records, 'products', ['uid','date','gender'])")


# In[49]:


purchase.head()


# In[16]:


purchase = purchase.join(catalog.set_index('pid'), on='pid')


# In[18]:


purchase.info()


# In[28]:


categorical = ['category', 'sub_category', 'sub_sub_category','gender']
new_purchase = pd.get_dummies(purchase, columns=categorical)
new_purchase.info()


# In[23]:


new_purchase.head()


# In[29]:


dummy_cols = ['category_c1bd5fd999bd577743936898ada96496b547af3c',
'sub_category_f08770a96fb546673053ab799f5ea9cada06c06a',
'sub_sub_category_2d2c44a2d8f18a6271f0e8057313af68a46d0f24',
'gender_F']
purchase2 = new_purchase.drop(dummy_cols, 1)
purchase2.info()


# ---
# # Machine Learning Algorithms

# In[65]:


from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# ## Separate train and test data

# In[45]:


features = purchase2.columns[6:-1]
target = purchase2.columns[-1]


# In[50]:


X = purchase2[features]
Y = purchase2[target]


# In[51]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)


# ## Naive Bayes

# In[66]:


get_ipython().run_cell_magic(u'time', u'', u'gnb = GaussianNB()\ngnb.fit(x_train, y_train)\ny_pred = gnb.predict(x_test)\nscore = metrics.accuracy_score(y_test, y_pred)')


# In[67]:


score


# ## KNeighborsClassifier

# In[68]:


get_ipython().run_cell_magic(u'time', u'', u'knn = KNeighborsClassifier(n_neighbors=1)\nknn.fit(x_train, y_train)\ny_pred = knn.predict(x_test)\nscore = metrics.accuracy_score(y_test, y_pred)')


# In[69]:


score


# ## Decision Tree

# In[71]:


get_ipython().run_cell_magic(u'time', u'', u'tree = DecisionTreeClassifier()\ntree.fit(x_train, y_train)\ny_pred = tree.predict(x_test)\nscore = metrics.accuracy_score(y_test, y_pred)')


# In[72]:


score


# ## Forest Tree

# In[73]:


get_ipython().run_cell_magic(u'time', u'', u'forest = RandomForestClassifier()\nforest.fit(x_train, y_train)\ny_pred = tree.predict(x_test)\nscore = metrics.accuracy_score(y_test, y_pred)')


# In[ ]:




