#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.head()


# In[4]:


black_friday.info()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[5]:


def q1():
    return black_friday.shape
    


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[6]:


def q2():
    black_friday_gender_F = black_friday['Gender']=='F'
    black_friday_age = black_friday['Age']=='26-35'
    return  int(black_friday[(black_friday_gender_F)&(black_friday_age)].shape[0])


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[7]:


def q3():
    return int(black_friday['User_ID'].nunique())


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[8]:


def q4():
    return int(black_friday.dtypes.nunique())        


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[9]:


# Há apenas duas colunas com nulos
black_friday.isna().sum()


# In[18]:


null_A  = black_friday['Product_Category_2'].isnull().sum()
# valor de n(B)
null_B = black_friday['Product_Category_3'].isnull().sum()
# valor de n(A∩B)
nulls_AandB = (null_A)&(null_B)

nulls_AandB


# In[19]:


def q5():
    # Usando teoria dos conjuntos : n(AUB) = n(A) + b(B) - n(A∩B)(com isso respondemos e 5° e 10° questão visto que n(A) = n(A∩B))
    # valor de n(A)
    null_A  = black_friday['Product_Category_2'].isnull()
    # valor de n(B)
    null_B = black_friday['Product_Category_3'].isnull()
    # valor de n(A∩B)
    nulls_AandB = ((null_A)&(null_B))
    # valor de n(AUB)
    nulos_total_AorB = null_A.sum() + null_B.sum() - nulls_AandB.sum()
    return   float(nulos_total_AorB/black_friday.shape[0])
q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[ ]:


def q6():
    return int(black_friday['Product_Category_3'].isna().sum())


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[ ]:


def q7():
    return black_friday['Product_Category_3'].value_counts().index[0]


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[24]:


def q8():
    purchase_var= black_friday['Purchase']
    normalize_purchase = (purchase_var -purchase_var.min())/(purchase_var.max() -purchase_var.min())
    return float(normalize_purchase.mean() )   


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[25]:


def q9():
    purchase_var= black_friday['Purchase']
    standard_purcharse = (purchase_var - purchase_var.mean())/purchase_var.std() 
    return int(((standard_purcharse>-1) & (standard_purcharse <1)).sum())


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[23]:


def q10():
    # número de ocorrencias null em Product_Category_2
    occurrence_null_product_category_2 = black_friday['Product_Category_2'].isnull()
    # número de ocorrencias null em Product_Category_3
    occurrence_null_product_category_3 = black_friday['Product_Category_3'].isnull()
    #Se número de ocorrências null do produto_categoria_2 for igual a inteserção dele mesmo com produto_categori_3 então realmente podemos afirma
    if occurrence_null_product_category_2.sum() == ((occurrence_null_product_category_2)&(occurrence_null_product_category_3)).sum():
        anwser = True
    else:
        anwser = False
    return anwser    

