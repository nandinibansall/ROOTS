#!/usr/bin/env python
# coding: utf-8

# In[9]:


import warnings,logging
warnings.simplefilter('ignore')
logging.disable(logging.WARNING)


# In[11]:


import sys
get_ipython().system('{sys.executable} -m pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu')
get_ipython().system('{sys.executable} -m pip install --quiet transformers pillow ipywidgets huggingface_hub')


# In[12]:


get_ipython().system('jupyter nbextension enable --py widgetsnbextension')


# In[13]:


import torch
print(torch.__version__)


# In[14]:


from transformers import pipeline


# In[15]:


ocr=pipeline('image-to-text', model="microsoft/trocr-base-handwritten")


# In[16]:


from PIL import Image


# In[23]:


image_path = "C:/Users/peehu/OneDrive/Desktop/imagee.jpg"
image = Image.open(image_path)


# In[24]:


result = ocr(image)
print("Recognized text:\n")
print(result[0]['generated_text'])


# In[ ]:




