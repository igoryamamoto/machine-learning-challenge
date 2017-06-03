
# coding: utf-8

# In[21]:


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
from sklearn.externals import joblib
get_ipython().magic(u'matplotlib inline')


# In[2]:


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
    except TypeError:
        weekday = 1
    return weekday


# ## Create 5 DataFrames: catalog, purchase, pageview, target_purchase, target_pageview

# In[5]:


get_ipython().run_cell_magic(u'time', u'', u"catalog = pd.read_csv('data/catalog', usecols=['pid', 'current_price', 'category', 'sub_category', 'sub_sub_category'])\nfor i,val in enumerate(catalog.current_price.values):\n    if np.isnan(val):\n        catalog.current_price.values[i] = 0\n\npurchase_data = open_json('data/sub_data/purchase')           \npurchase = json_normalize(purchase_data, 'products', ['date','gender','uid'])\n\npageview_data = open_json('data/sub_data/products')   \npageview = pd.DataFrame(pageview_data, columns=['productId', 'timestamp', 'gender', 'uid'])\npageview.columns = ['pid', 'date', 'gender', 'uid']\n\n\npurchase_data = open_json('data/sub_target/purchase')           \ntarget_purchase = json_normalize(purchase_data, 'products', ['date','uid'])\n\npageview_data = open_json('data/sub_target/purchase')   \ntarget_pageview = pd.DataFrame(pageview_data, columns=['productId', 'timestamp', 'uid'])\ntarget_pageview.columns = ['pid', 'date', 'uid']")


# ## Transform gender and date into numbers

# In[6]:


get_ipython().run_cell_magic(u'time', u'', u'purchase.gender = list(map(gender_to_bin, purchase.gender.values))\npurchase.date = list(map(to_weekday, purchase.date.values))\n\npageview.gender = list(map(gender_to_bin, pageview.gender.values))\npageview.date = list(map(to_weekday, pageview.date.values))\n\n\ntarget_purchase.date = list(map(to_weekday, target_purchase.date.values))\ntarget_pageview.date = list(map(to_weekday, target_pageview.date.values))')


# ## Set dummies vars for categorical features

# In[7]:


get_ipython().run_cell_magic(u'time', u'', u"categorical = ['category', 'sub_category', 'sub_sub_category', 'date']\n\npurchase = purchase.join(catalog.set_index('pid'), on='pid')\npurchase = pd.get_dummies(purchase, columns=categorical).iloc[:,1:]\n\npageview = pageview.join(catalog.set_index('pid'), on='pid')\npageview = pd.get_dummies(pageview, columns=categorical).iloc[:,1:]\n\ntarget_purchase = target_purchase.join(catalog.set_index('pid'), on='pid')\ntarget_purchase = pd.get_dummies(target_purchase, columns=categorical).iloc[:,1:]\n\ntarget_pageview = target_pageview.join(catalog.set_index('pid'), on='pid')\ntarget_pageview = pd.get_dummies(target_pageview, columns=categorical).iloc[:,1:]")


# ## Merge users with same uid (sum entries) and join DFs

# In[8]:


len(target_purchase.uid.values)


# In[9]:


get_ipython().run_cell_magic(u'time', u'', u"#quantity = purchase.pop('quantity')\npurchase = purchase.groupby(['uid','gender'], as_index=False).sum()\npageview = pageview.groupby(['uid','gender'], as_index=False).sum()\ndata = purchase.join(pageview.set_index(['uid','gender']), on=['uid','gender'], rsuffix='_v')\n\n\ntarget_purchase = target_purchase.groupby(['uid'], as_index=False).sum()\ntarget_pageview = target_pageview.groupby(['uid'], as_index=False).sum()\ntarget_data = target_purchase.join(target_pageview.set_index('uid'), on='uid', rsuffix='_v')")


# In[10]:


data = data.fillna(value=0)
target_data = target_data.fillna(value=0)


# In[11]:


features = data.columns[2:]
target = data.columns[1]
features2 = target_data.columns[1:]
features = list(set(features)&set(features2))


# In[94]:


X = data[features]
Y = data[target]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)


# In[78]:


get_ipython().run_cell_magic(u'time', u'', u"forest = RandomForestClassifier(n_estimators=100)\nforest.fit(x_train, y_train)\ny_pred = forest.predict(x_test)\nscore1 = metrics.accuracy_score(y_test, y_pred)\nscore2 = metrics.f1_score(y_test, y_pred, average='binary')\nscore3 = metrics.roc_auc_score(y_test, y_pred)\n#joblib.dump(forest, '83.pkl')")


# In[79]:


print score1, score2, score3


# In[95]:


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1,random_state=0)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
score1 = metrics.accuracy_score(y_test, y_pred)
score2 = metrics.f1_score(y_test, y_pred, average='binary')
score3 = metrics.roc_auc_score(y_test, y_pred)
print score1, score2, score3


# ## Output

# In[96]:


X = target_data[features]
answer = clf.predict(X)
users = target_data.uid.values
ans = []
for i,u in enumerate(users):
    if answer[i]:
        g = 'M'
    else:
        g = 'F'
    obj = {'a':u, 'b':g}
    ans.append(obj)
    
import csv

with open('ans8.csv', 'wb') as f:
    w = csv.DictWriter(f, fieldnames=['a','b'])
    for obj in ans:
        w.writerow(obj)


# In[19]:


data.head()


# In[ ]:




