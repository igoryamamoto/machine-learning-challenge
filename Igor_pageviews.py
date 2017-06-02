
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import json
import gzip
import seaborn as sns
from pandas.io.json import json_normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
get_ipython().magic(u'matplotlib inline')


# In[8]:


catalog = pd.read_csv('data/catalog.gz')


# In[9]:


get_ipython().run_cell_magic(u'time', u'', u"with open('data/purchase_data', 'r') as f:\n    json_records = []\n    for line in f:\n        record = json.loads(line)\n        json_records.append(record)\npurchase = json_normalize(json_records, 'products', ['uid','date','gender'])\npurchase = purchase.join(catalog.set_index('pid'), on='pid')\ncategorical = ['category', 'sub_category', 'sub_sub_category','gender']\npurchase = pd.get_dummies(purchase, columns=categorical)\ndummy_cols = ['category_c1bd5fd999bd577743936898ada96496b547af3c',\n'sub_category_f08770a96fb546673053ab799f5ea9cada06c06a',\n'sub_sub_category_2d2c44a2d8f18a6271f0e8057313af68a46d0f24',\n'gender_F']\npurchase = purchase.drop(dummy_cols, 1)\nfeatures = purchase.columns[6:-1]\ntarget = purchase.columns[-1]")


# In[10]:


get_ipython().run_cell_magic(u'time', u'', u"with open('data/purchase_target', 'r') as f:\n    json_records = []\n    for line in f:\n        record = json.loads(line)\n        json_records.append(record)\npurchase2 = json_normalize(json_records, 'products', ['uid','date'])\npurchase2 = purchase2.join(catalog.set_index('pid'), on='pid')\ncategorical = ['category', 'sub_category', 'sub_sub_category']\npurchase2 = pd.get_dummies(purchase2, columns=categorical)\ndummy_cols = ['category_c1bd5fd999bd577743936898ada96496b547af3c',\n'sub_category_f08770a96fb546673053ab799f5ea9cada06c06a',\n'sub_sub_category_2d2c44a2d8f18a6271f0e8057313af68a46d0f24']\npurchase2 = purchase2.drop(dummy_cols, 1)\nfeatures2 = purchase2.columns[6:]")


# In[11]:


new_features = list(set(features)&set(features2))
X = purchase[new_features]
Y = purchase[target]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)


# In[17]:


get_ipython().run_cell_magic(u'time', u'', u'forest = RandomForestClassifier(n_estimators=100)\nforest.fit(x_train, y_train)\ny_pred = forest.predict(x_test)\nscore = metrics.accuracy_score(y_test, y_pred)')


# In[18]:


score


# ---
# 
# Pageviews
# 
# ---

# In[22]:


get_ipython().run_cell_magic(u'time', u'', u"with open('data/products_data', 'r') as f:\n    json_records = []\n    for line in f:\n        record = json.loads(line)\n        json_records.append(record)\npageview = pd.DataFrame(json_records)")


# In[23]:


pageview.info()


# In[ ]:




