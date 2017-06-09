
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import gzip
import seaborn as sns
from datetime import datetime
from pandas.io.json import json_normalize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.externals import joblib
get_ipython().magic(u'matplotlib inline')


# In[42]:


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

def to_weekday(timestamp):
    try:
        weekday = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").weekday()
    except ValueError:
        weekday = datetime.strptime(timestamp, "%Y-%m-%d").weekday()
    except TypeError:
        weekday = 9
    return weekday

def convert_date(timestamp):
    try:
        res = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        res = datetime.strptime(timestamp, "%Y-%m-%d")
    except TypeError:
        res = 9
    return res

def get_data_pageview(page_type):
    json_records = open_json('data/sub_data/{}'.format(page_type))
    df = pd.DataFrame(json_records, columns=['uid', 'gender', 'page_type','timestamp'])
    df.columns = ['uid', 'gender', '{}'.format(page_type), 'date']
    df.gender = list(map(gender_to_bin, df.gender.values))
    df.date = list(map(to_weekday, df.date.values))
    df = pd.get_dummies(df, columns=['{}'.format(page_type),'date'])
    return df.groupby(['uid','gender'], as_index=False).sum()

def get_target_pageview(page_type):
    json_records = open_json('data/sub_target/{}'.format(page_type))
    df = pd.DataFrame(json_records, columns=['uid', 'page_type','timestamp'])
    df.columns = ['uid', '{}'.format(page_type), 'date']
    df.date = list(map(to_weekday, df.date.values))
    df = pd.get_dummies(df, columns=['{}'.format(page_type),'date'])
    return df.groupby(['uid'], as_index=False).sum()


# ## Create 5 DataFrames: catalog, purchase, pageview, target_purchase, target_pageview

# In[43]:


get_ipython().run_cell_magic(u'time', u'', u"catalog = pd.read_csv('data/catalog', usecols=['pid', 'current_price','original_price', 'category', 'sub_category', 'sub_sub_category'])\ncatalog = catalog.fillna(value=0)\ncatalog = catalog[catalog.current_price!=0]\ncatalog['original_price'] = np.where(catalog['original_price'] == 0, catalog['current_price'], catalog['original_price'])\ncatalog['diff_price'] = catalog['original_price'] - catalog['current_price']\n\npurchase_data = open_json('data/sub_data/purchase')           \npurchase = json_normalize(purchase_data, 'products', ['date','gender','uid'])\n\npageview_data = open_json('data/sub_data/products')   \npageview = pd.DataFrame(pageview_data, columns=['productId', 'timestamp', 'gender', 'uid'])\npageview.columns = ['pid', 'date', 'gender', 'uid']\n\n\npurchase_data = open_json('data/sub_target/purchase')           \ntarget_purchase = json_normalize(purchase_data, 'products', ['date','uid'])\n\npageview_data = open_json('data/sub_target/purchase')   \ntarget_pageview = pd.DataFrame(pageview_data, columns=['productId', 'timestamp', 'uid'])\ntarget_pageview.columns = ['pid', 'date', 'uid']")


# In[ ]:





# ## Transform gender and date into numbers

# In[44]:


get_ipython().run_cell_magic(u'time', u'', u"purchase.gender = list(map(gender_to_bin, purchase.gender.values))\npurchase['hour'] = purchase['date']\npurchase['month'] = purchase['date']\n#purchase['day'] = purchase['date']\n#pageview['hour'] = pageview['date']\n#pageview['month'] = pageview['date']\n#pageview['day'] = pageview['date']\n#purchase.day = list(map(lambda t:convert_date(t).day, purchase.day.values))\npurchase.month = list(map(lambda t:convert_date(t).month, purchase.month.values))\npurchase.hour = list(map(lambda t:convert_date(t).hour, purchase.hour.values))\npurchase.date = list(map(to_weekday, purchase.date.values))\npageview.gender = list(map(gender_to_bin, pageview.gender.values))\n#pageview.day = list(map(lambda t:convert_date(t).day, pageview.day.values))\n#pageview.month = list(map(lambda t:convert_date(t).month, pageview.month.values))\n#pageview.hour = list(map(lambda t:convert_date(t).hour, pageview.hour.values))\npageview.date = list(map(to_weekday, pageview.date.values))\n\ntarget_purchase['hour'] = target_purchase['date']\ntarget_purchase['month'] = target_purchase['date']\n#target_purchase['day'] = target_purchase['date']\n#target_pageview['hour'] = target_pageview['date']\n#target_pageview['month'] = target_pageview['date']\n#target_pageview['day'] = target_pageview['date']\n#target_purchase.day = list(map(lambda t:convert_date(t).day, target_purchase.day.values))\ntarget_purchase.month = list(map(lambda t:convert_date(t).month, target_purchase.month.values))\ntarget_purchase.hour = list(map(lambda t:convert_date(t).hour, target_purchase.hour.values))\ntarget_purchase.date = list(map(to_weekday, target_purchase.date.values))\n#target_pageview.day = list(map(lambda t:convert_date(t).day, target_pageview.day.values))\n#target_pageview.month = list(map(lambda t:convert_date(t).month, target_pageview.month.values))\n#target_pageview.hour = list(map(lambda t:convert_date(t).hour, target_pageview.hour.values))\ntarget_pageview.date = list(map(to_weekday, target_pageview.date.values))")


# ## Set dummies vars for categorical features

# In[45]:


get_ipython().run_cell_magic(u'time', u'', u"categorical_p = ['category', 'sub_category', 'sub_sub_category', 'date','hour','month']\ncategorical_v = ['category', 'sub_category', 'sub_sub_category', 'date']\n\npurchase = purchase.join(catalog.set_index('pid'), on='pid')\npurchase = pd.get_dummies(purchase, columns=categorical_p).iloc[:,1:]\n\npageview = pageview.join(catalog.set_index('pid'), on='pid')\npageview = pd.get_dummies(pageview, columns=categorical_v).iloc[:,1:]\n\ntarget_purchase = target_purchase.join(catalog.set_index('pid'), on='pid')\ntarget_purchase = pd.get_dummies(target_purchase, columns=categorical_p).iloc[:,1:]\n\ntarget_pageview = target_pageview.join(catalog.set_index('pid'), on='pid')\ntarget_pageview = pd.get_dummies(target_pageview, columns=categorical_v).iloc[:,1:]")


# ## Merge users with same uid (sum entries) and join DFs

# In[46]:


purchase.current_price = purchase.current_price.values*purchase.quantity
purchase.original_price = purchase.original_price.values*purchase.quantity
purchase.diff_price = purchase.diff_price.values*purchase.quantity


# In[47]:


get_ipython().run_cell_magic(u'time', u'', u"purchase = purchase.groupby(['uid','gender'], as_index=False).sum()\npageview = pageview.groupby(['uid','gender'], as_index=False).sum()\ndata = purchase.join(pageview.set_index(['uid','gender']), on=['uid','gender'], rsuffix='_v')\n\n\ntarget_purchase = target_purchase.groupby(['uid'], as_index=False).sum()\ntarget_pageview = target_pageview.groupby(['uid'], as_index=False).sum()\ntarget_data = target_purchase.join(target_pageview.set_index('uid'), on='uid', rsuffix='_v')")


# In[48]:


purchase.to_csv('grouped_purchase_data.csv')
pageview.to_csv('grouped_product_pageview_data.csv')
target_purchase.to_csv('grouped_target_purchase_data.csv')
target_pageview.to_csv('grouped_target_product_pageview_data.csv')


# ## Additional features

# In[97]:


search = get_data_pageview('search')
target_search = get_target_pageview('search')
data2 = data.join(search.set_index(['uid','gender']), on=['uid','gender'], rsuffix='_s')
target_data2 = target_data.join(target_search.set_index('uid'), on='uid', rsuffix='_s')

home = get_data_pageview('home')
target_home = get_target_pageview('home')
data2 = data2.join(home.set_index(['uid','gender']), on=['uid','gender'], rsuffix='_s')
target_data2 = target_data2.join(target_home.set_index('uid'), on='uid', rsuffix='_s')


# In[8]:


len(search.uid.values)


# In[49]:


data = data.fillna(value=0)
target_data = target_data.fillna(value=0)
features = data.columns[2:]
target = data.columns[1]
features2 = target_data.columns[1:]
features = list(set(features)&set(features2))


# In[98]:


data2 = data2.fillna(value=0)
target_data2 = target_data2.fillna(value=0)
features = data2.columns[2:]
target = data2.columns[1]
features2 = target_data2.columns[1:]
features = list(set(features)&set(features2))


# In[94]:


X = data[features]
Y = data[target]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=32)


# In[146]:


X = data2[features]
Y = data2[target]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.35, random_state=42)


# In[147]:


get_ipython().run_cell_magic(u'time', u'', u"forest = RandomForestClassifier(n_estimators=100)\nforest.fit(x_train, y_train)\ny_pred = forest.predict(x_test)\nscore1 = metrics.accuracy_score(y_test, y_pred)\nscore2 = metrics.f1_score(y_test, y_pred, average='binary')\nscore3 = metrics.roc_auc_score(y_test, y_pred)\n#joblib.dump(forest, '83.pkl')\nprint score1, score2, score3")


# In[148]:


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1,random_state=0)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
score1 = metrics.accuracy_score(y_test, y_pred)
score2 = metrics.f1_score(y_test, y_pred, average='binary')
score3 = metrics.roc_auc_score(y_test, y_pred)
print score1, score2, score3


# ## Output

# In[57]:


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

with open('ans12.csv', 'wb') as f:
    w = csv.DictWriter(f, fieldnames=['a','b'])
    for obj in ans:
        w.writerow(obj)


# In[160]:


forest.max


# In[ ]:




