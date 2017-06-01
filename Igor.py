
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np
import json
import gzip
import seaborn as sns
from pandas.io.json import json_normalize


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

# In[42]:


get_ipython().run_cell_magic(u'time', u'', u"with open('data/purchase_data', 'r') as f:\n    json_records = []\n    for line in f:\n        record = json.loads(line)\n        products = record.pop('products')\n        record.pop('source')\n        record.pop('event_type')\n        for obj in products:\n            new_record = record.copy()\n            new_record.update(obj)\n            json_records.append(new_record)\npurchase = pd.DataFrame(json_records)")


# In[43]:


purchase.head()


# ### Using json_normalize

# In[48]:


get_ipython().run_cell_magic(u'time', u'', u"with open('data/purchase_data', 'r') as f:\n    json_records = []\n    for line in f:\n        record = json.loads(line)\n        json_records.append(record)\npurchase = json_normalize(json_records, 'products', ['uid','date','gender'])")


# In[49]:


purchase.head()


# In[ ]:




