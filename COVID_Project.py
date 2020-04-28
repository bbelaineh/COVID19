#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from matplotlib.pyplot import *
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy.stats as stats
from fbprophet import Prophet
from scipy.integrate import odeint
pd.options.display.float_format = "{:.2f}".format


#I've set April 25th,2020 as the cutoff date for all COVID data


# In[3]:


US_deaths_df = pd.read_csv(r'C:\Users\Admin\Desktop\Project_Data\time_series_covid19_deaths_US.csv')
US_predictors = pd.read_csv(r'C:\Users\Admin\Desktop\Project_Data\COVID19_State.csv')


# In[4]:


#Creating predictor dataframes for New York, Viriginia, and California deaths


New_York_predictors = US_predictors[US_predictors['State']=='New York']
California_predictors = US_predictors[US_predictors['State']=='California']
Virginia_predictors = US_predictors[US_predictors['State']=='Virignia']


# In[5]:


US_predictors[["Tested","Infected","Deaths","Population","Pop Density","Gini","ICU Beds","Income","GDP","Physicians","Hospitals","Health Spending"]].describe()
US_predic_small = US_predictors[["Tested","Infected","Deaths","Population","ICU Beds","GDP","Physicians","Hospitals","Health Spending"]]


# In[6]:


US_predic_small.describe()


# In[7]:



cols = US_deaths_df.keys()

US_deaths = US_deaths_df.loc[:, cols[12]:cols[-1]]


# In[8]:


US_deaths
dates = US_deaths.keys()


# In[9]:


#dates
NY_Deaths = []
California_Deaths = []
Virginia_Deaths = []
US_Deaths_Tot = []


# In[10]:


#Creates the running sum for Death counts

for i in dates:
    NY_Deaths.append(US_deaths_df[US_deaths_df['Province_State']=='New York'][i].sum())
    California_Deaths.append(US_deaths_df[US_deaths_df['Province_State']=='California'][i].sum())
    Virginia_Deaths.append(US_deaths_df[US_deaths_df['Province_State']=='Virginia'][i].sum())
    US_Deaths_Tot.append(US_deaths[i].sum())


# In[11]:


#Converts the dates into "Days since January 20"
days_since = np.array([i for i in range(len(dates))]).reshape(-1,1)


# In[12]:



plt.plot(days_since,NY_Deaths)
plt.xlabel('Days Since 1/20/2020')
plt.ylabel('Number of Cases')
plt.title('Number of Deaths in New York')

plt.show()


# In[13]:



plt.plot(days_since,California_Deaths)
plt.xlabel('Days Since 1/20/2020')
plt.ylabel('Number of Cases')
plt.title('Number of Deaths in California')
plt.show()


# In[14]:



plt.plot(days_since,Virginia_Deaths)
plt.xlabel('Days Since 1/20/2020')
plt.ylabel('Number of Cases')
plt.title('Number of Deaths in Virginia')
plt.show()


# In[15]:


plt.figure(figsize=(12,12))
plt.plot(dates, US_Deaths_Tot)

plt.xlabel('Days Since 1/20/2020')
plt.ylabel('Number of Cases')
plt.title('Number of Deaths in the United States')
plt.show()


# In[16]:


new_dates = []
for date in dates:
    new_date = date.replace('/','-')
    new_dates.append(new_date)
    
  


# In[17]:


Virginia_Deaths_df = pd.DataFrame({"ds":new_dates,"y":Virginia_Deaths})
Cali_Deaths_df = pd.DataFrame({"ds":new_dates,"y":California_Deaths})
NY_Deaths_df = pd.DataFrame({"ds":new_dates,"y":NY_Deaths})
US_Deaths_df = pd.DataFrame({"ds":new_dates,"y":US_Deaths_Tot})


# In[18]:


Virginia_predic = Prophet()
Virginia_predic.fit(Virginia_Deaths_df)

NY_predic = Prophet()
NY_predic.fit(NY_Deaths_df)

Cali_predic = Prophet()
Cali_predic.fit(Cali_Deaths_df)

US_predic = Prophet()
US_predic.fit(US_Deaths_df)


# In[20]:


Virginia_future = Virginia_predic.make_future_dataframe(periods=60)

Virginia_forceast = Virginia_predic.predict(Virginia_future)

fig = Virginia_predic.plot(Virginia_forceast,xlabel="Date",ylabel='Deaths')
ax = fig.gca()
ax.set_title("Predicted Deaths in 60 Days For Virginia")


# In[ ]:


fig2 = Virginia_predic.plot_components(forecast)


# In[ ]:


NY_future = NY_predic.make_future_dataframe(periods=60)

NY_forceast = NY_predic.predict(NY_future)

fig = NY_predic.plot(NY_forceast,xlabel="Date",ylabel='Deaths')
ax = fig.gca()
ax.set_title("Predicted Deaths in 60 Days For New York")


# In[ ]:


Cali_future = Cali_predic.make_future_dataframe(periods=60)

Cali_forceast = Cali_predic.predict(Cali_future)

fig = Cali_predic.plot(Cali_forceast,xlabel="Date",ylabel='Deaths')
ax = fig.gca()
ax.set_title("Predicted Deaths in 60 Days For California")


# In[21]:


US_future = US_predic.make_future_dataframe(periods=60)

US_forceast = US_predic.predict(US_future)

fig = US_predic.plot(US_forceast,xlabel="Date",ylabel='Deaths')
ax = fig.gca()
ax.set_title("Predicted Deaths in 60 Days For the United States")


# In[99]:


corrmatrix = US_predic_small.corr()
sns.pairplot(US_predic_small)
sns.plt.show()


# In[65]:




def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# In[66]:




N = 19440469

beta = 1.0  # infected person infects 1 other person per day
D = 14.0 # infections lasts four days
gamma = 1.0 / D

S0, I0, R0 = (19440469-1), 1, 0  # initial conditions: one infected, rest susceptible


# In[67]:


t = np.linspace(0, 94, 94) # Grid of time points (in days)
y0 = S0, I0, R0 # Initial conditions vector

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T


# In[76]:




def plotsir(t, S, I, R):
  f, ax = plt.subplots(1,1,figsize=(10,4))
  ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
  ax.plot(t, I, 'y', alpha=0.7, linewidth=2, label='Infected')
  ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
  
  ax.set_xlabel('Time (days)')
  ax.set_title("SIR Model Simulation for New York over 80 Days")
  ax.yaxis.set_tick_params(length=0)
  ax.xaxis.set_tick_params(length=0)
  ax.get_yaxis().get_major_formatter().set_scientific(False)
  ax.grid(b=True, which='major', c='w', lw=2, ls='-')
  legend = ax.legend()
  legend.get_frame().set_alpha(0.5)
  for spine in ('top', 'right', 'bottom', 'left'):
      ax.spines[spine].set_visible(False)
  plt.show();


# In[77]:


plotsir(t, S, I, R)


# In[93]:


N = 328000000
beta = 1.0  # infected person infects 1 other person per day
D = 14.0 # infections lasts four days
gamma = 1.0 / D

S0, I0, R0 = (N-1), 1, 0  # initial conditions: one infected, rest susceptible


# In[94]:


t = np.linspace(0, 94, 94) # Grid of time points (in days)
y0 = S0, I0, R0 # Initial conditions vector

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T


# In[95]:



def plotsir(t, S, I, R):
  f, ax = plt.subplots(1,1,figsize=(10,4))
  ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
  ax.plot(t, I, 'y', alpha=0.7, linewidth=2, label='Infected')
  ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
  
  ax.set_xlabel('Time (days)')
  ax.set_title("SIR Model Simulation for United States Over 80 Days")
  ax.yaxis.set_tick_params(length=0)
  ax.xaxis.set_tick_params(length=0)
  ax.get_yaxis().get_major_formatter().set_scientific(False)
  ax.grid(b=True, which='major', c='w', lw=2, ls='-')
  legend = ax.legend()
  legend.get_frame().set_alpha(0.5)
  for spine in ('top', 'right', 'bottom', 'left'):
      ax.spines[spine].set_visible(False)
  plt.show();


# In[96]:


plotsir(t, S, I, R)


# In[ ]:



labels = [c for c in US_predic_small.columns]
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
ax.
ax.matshow(US_predic_small.corr(),cmap=plt.cm.RdYlGn)
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels)
ax.set_xticklabels(lables)

plt.show()


# In[ ]:


mask = np.triu(np.ones_like(US_predic_small.corr(),dtype=np.bool))
f,ax = plt.subplots(figsize=(12,6))
cmap = sns.diverging_palette(220,10,as_cmap=True)
sns.heatmap(US_predic_small.corr(), mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)


# In[97]:


plt.subplots(figsize=(10,6))
plt.title("Health Care Predictors Correlations")
sns.heatmap(US_predic_small.corr(),xticklabels=US_predic_small.corr().columns,yticklabels=US_predic_small.corr().columns,cmap='BuGn')


# In[100]:


print(corrmatrix)

corrmatrix.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[ ]:




