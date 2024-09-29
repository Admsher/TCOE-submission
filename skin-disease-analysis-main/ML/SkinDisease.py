#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import json
url="https://centralindia.api.cognitive.microsoft.com/customvision/v3.0/Prediction/82462bcc-5616-4f82-98ea-8d4fda55a9e6/classify/iterations/Iteration1/url"
headers={'content-type':'application/json','Prediction-Key':'24ab65398bb943768ed6aea74b9a073e'}
body={"Url": "https://i.imgur.com/cYzaOkV.jpg"}
r =requests.post(url,json=body,headers=headers)


# In[2]:


print(r.json())


# In[14]:


response=r.json()

for prediction in response["predictions"]:
    if prediction["probability"]>0.60:
        print(prediction["tagName"]+":",end="")
        print(prediction["probability"]*100)
        #print(prediction["probability"])


# In[ ]:




