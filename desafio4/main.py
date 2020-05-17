#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.stats as sct
import seaborn as sns


# In[2]:


'''%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()''';


# In[3]:


athletes = pd.read_csv("athletes.csv")


# In[4]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
athletes.head()


# In[6]:


athletes.describe()


# In[7]:


athletes.info()


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[8]:


height = get_sample(athletes,'height',n =3000)
height.head()


# In[9]:


def q1():
    alpha = 0.05
    return sct.shapiro(height)[1]> alpha
q1()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# In[90]:


sns.distplot(height,bins=25);
sm.qqplot(height,fit = True,line = '45');


# O gráfico do histograma visualmente parece seguir uma curva normal  e entra em desacordo com o resultado anterior.

# O segundo gráfico parece vim de um distribuição normal também,o que pode indicar que se trata do `erro tipo 1`:em que se rejeita a $H_{0}$ sendo a mesma verdadeira

# talvez 10%

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[11]:


def q2():
    # Retorne aqui o resultado da questão 2.
    alpha = 0.05
    return sct.jarque_bera(height)[1]> alpha
q2()


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# In[44]:


print('Skewness: ',height.skew(),'\n',
      'Kurtosis: ',height.kurt(),'\n',)


# O resultado  se aproxima de muito de uma normal.
# Provavelmente o resultado 'false' é falso posito:erro tipo 1

# ## Questão 3
# 
# Considerando agoraskewness_heighttra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[12]:


weight = get_sample(athletes,'weight',n = 3000)


# In[13]:


def q3():
    alpha = 0.05
    return sct.normaltest(weight)[1]> alpha
q3()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# In[27]:


fig, axs = plt.subplots(1, 2, figsize=(12, 8))
sns.distplot(weight,bins =25,ax = axs[0])
sns.boxplot(y= 'weight',data =athletes,ax = axs[1]);


# In[73]:


skewness_weight = weight.skew()
kurtosis_weight = weight.kurt()
print('Kurtosis weight: ',kurtosis_weight,'\n' 'Skewness_weight: ',skewness_weight)


# Temos uma assimetria possitiva vendo o histrograma,o que faz sentido não ser uma distruibuição normal.
# E há muitos outliears pelo gráfico boxplot.

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[45]:


weight_log = np.log(weight)


# In[46]:


def q4():
    alpha = 0.05
    return sct.normaltest(weight_log)[1]> alpha  
q4()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`)weight_logma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# In[88]:


fig, axs = plt.subplots(1, 3, figsize=(15, 8))
sns.distplot(weight_log,bins =25,ax = axs[0],axlabel ='log(weight)')
sns.distplot(weight,bins =25,ax = axs[1],axlabel ='weight')
sns.boxplot(y= weight_log,ax = axs[2])
plt.ylabel('log(weight)')
fig;


# In[86]:


sm.qqplot(weight,fit = True,line='45');
sm.qqplot(weight_log,fit = True,line='45');


# In[79]:


skewness_weight_log = weight_log.skew()
kurtosis_weight_log = weight_log.kurt()
print('Kurtosis weight_log: ',kurtosis_weight_log,'\n' 'Skewness weight_log: ',skewness_weight_log)


# A aplicação do logaritmo reduziu bastante a kurtosi e diminuiu a assimetria e tirou bastante dos outliers se comparado ao sem logarítmo,reduziu bastante o viés. O gráfico agora se assemelha bastante a de uma distribuição normal,provavelmente o aconteceu foi o `erro do tipo 1`

# O gráfico qqplot mostra que aplicação do logaritmo em `weight` parece seguir uma distruição normal

# Eu realmente esperava um resultado diferente,talvez um `alpha =0.1` funcionce 

# Alguns links que me ajudaram a entender melhor o que aconteceu:
# 
# [Por que usar transformação logarítmica em dados](http://rstudio-pubs-static.s3.amazonaws.com/289147_99e32d5403f942339c3fe05414ac62fd.html)
# 
# [Transformações de dados amostrais](http://www.forp.usp.br/restauradora/gmc/gmc_livro/gmc_livro_cap13.html)

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[16]:


athletes['nationality'].value_counts().head()


# In[17]:


bra = athletes[athletes['nationality'] =='BRA']['height'].dropna()
usa = athletes[athletes['nationality'] =='USA']['height'].dropna()
can = athletes[athletes['nationality'] =='CAN']['height'].dropna()


# In[18]:


alpha = 0.05


# In[19]:


def q5():
    return sct.ttest_ind(bra,usa,equal_var=False)[1]>alpha
q5()


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[20]:


def q6():
    return sct.ttest_ind(bra,can,equal_var=False)[1]>alpha
q6()


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[21]:


def q7():
    return float(round((sct.ttest_ind(usa,can,equal_var=False)[1]),8))
q7()


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?

# In[ ]:




