
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import gzip
import seaborn as sns


# In[2]:


catalog = pd.read_csv('data/catalog.gz')


# In[3]:


catalog.head()


# In[66]:


get_ipython().run_cell_magic(u'time', u'', u"with gzip.open('data/test.gz', 'r') as f:\n    json_records = []\n    for line in f:\n        record = json.loads(line)\n        json_records.append(record)\ndata = pd.DataFrame(json_records)")


# In[67]:


data.head()


# In[98]:


get_ipython().run_cell_magic(u'time', u'', u"with open('data/purchase_data', 'r') as f:\n    json_records = []\n    for line in f:\n        record = json.loads(line)\n        json_records.append(record)\npurchase = pd.DataFrame(json_records)")


# In[99]:


purchase.head()


# In[ ]:




