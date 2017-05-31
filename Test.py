
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import json
import seaborn as sns


# In[23]:


catalog = pd.read_csv('data/catalog.gz')


# In[24]:


catalog.head()


# In[35]:


with open('data/test', 'r') as f:
    json_records = []
    for line in f:
        record = json.loads(line)
        json_records.append(record)


# In[36]:


json_records


# In[37]:


pd.DataFrame(json_records)


# In[39]:


oi = [{'col1':'oi', 'col2':'ola'},{'col1':'oi2','col2':'ola2'}]


# In[42]:


pd.DataFrame(oi)


# In[38]:


get_ipython().magic(u'pinfo pd.read_json')


# In[ ]:




