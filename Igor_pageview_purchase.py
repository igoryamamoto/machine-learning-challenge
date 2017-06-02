
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
import json
import gzip
import seaborn as sns
from datetime import datetime
from pandas.io.json import json_normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
get_ipython().magic(u'matplotlib inline')


# In[46]:


def open_json(data_file):    
    with open(data_file, 'r') as f:
        json_records = []
        for line in f:
            record = json.loads(line)
            json_records.append(record)
    return json_records

def gender_to_bin(g):
    if g == 'M':
        return 1
    else:
        return 0

def bin_to_gender(g):
    if g == 1:
        return 'M'
    else:
        return 'F'

def to_weekday(timestamp):
    try:
        weekday = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").weekday()
    except ValueError:
        weekday = datetime.strptime(timestamp, "%Y-%m-%d").weekday()
    return weekday


# In[39]:


get_ipython().run_cell_magic(u'time', u'', u"catalog = pd.read_csv('data/catalog.gz')\n\npurchase_data = open_json('data/purchase_data')           \npurchase = json_normalize(purchase_data, 'products', ['uid','date','gender'])\n\npageview_data = open_json('data/products_data')   \npageview = pd.DataFrame(pageview_data, columns=['uid', 'productId', 'timestamp', 'gender'])\npageview.columns = ['pid','uid','date','gender']")


# In[50]:


purchase.date = list(map(to_weekday, purchase.date.values))
pageview.date = list(map(to_weekday, pageview.date.values))


# In[36]:


categorical = ['category', 'sub_category', 'sub_sub_category']

purchase = purchase.join(catalog.set_index('pid'), on='pid')
purchase = pd.get_dummies(purchase, columns=categorical)

pageview = pageview.join(catalog.set_index('pid'), on='pid')
pageview = pd.get_dummies(pageview, columns=categorical)


# In[37]:


purchase.head()


# In[11]:


new_features = list(set(features)&set(features2))
X = purchase[new_features]
Y = purchase[target]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

