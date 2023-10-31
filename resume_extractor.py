#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyresparser import ResumeParser
import os
from docx import Document
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
stopw  = set(stopwords.words('english'))
df =pd.read_csv('job_final.csv') 


# In[2]:


df['test']=df['Job_Description'].apply(lambda x: ' '.join([word for word in str(x).split() if len(word)>2 and word not in (stopw)]))
df['test']


# In[8]:


import pdfplumber
import re

def extract_skills_from_resume(pdf_file):
    skills = set()  # Using a set to ensure unique skills
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            # Define a regular expression pattern to match skill-related keywords
            skill_pattern = r"(\bPython\b|\bJava\b|\bSQL\b|\bMachine Learning\b|\bData Analysis\b|\bData Visualization\b)"  # Add more skills as needed
            matches = re.findall(skill_pattern, text, flags=re.IGNORECASE)
            skills.update(matches)
    return skills


# In[9]:


if __name__ == "__main__":
    pdf_resume = "CV (1).pdf"
    extracted_skills = extract_skills_from_resume(pdf_resume)
    print("Skills extracted from the PDF resume:")
    print(extracted_skills)


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[11]:


vectorizer = TfidfVectorizer(min_df=1, analyzer="word", lowercase=False)
tfidf = vectorizer.fit_transform(list(extracted_skills))


# In[12]:


from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
test = (df['test'].values.astype('U'))


# In[13]:


def getNearestN(query):
  queryTFIDF_ = vectorizer.transform(query)
  distances, indices = nbrs.kneighbors(queryTFIDF_)
  return distances, indices


# In[14]:


distances, indices = getNearestN(test)
test = list(test) 
matches = []


# In[15]:


for i,j in enumerate(indices):
    dist=round(distances[i][0],2)
  
    temp = [dist]
    matches.append(temp)
    
matches = pd.DataFrame(matches, columns=['Match confidence'])


# In[16]:


df['match']=matches['Match confidence']
df1=df.sort_values('match')
df1[['Position', 'Company','Location']].head(10).reset_index()

