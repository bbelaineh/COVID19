#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


#Used to create the dataframes for the total deaths/confirmed cases and the predictors for the correlation visualizations

US_deaths_df = pd.read_csv(r'C:\Users\Admin\Desktop\Project_Data\time_series_covid19_deaths_US.csv')
US_predictors = pd.read_csv(r'C:\Users\Admin\Desktop\Project_Data\COVID19_State.csv')
US_confirmed_df = pd.read_csv(r'C:\Users\Admin\Desktop\Project_Data\time_series_covid19_confirmed_US.csv')


# In[3]:


#
US_deaths_df.groupby(['Province_State']).sum()


# In[4]:


#Creating predictor dataframes for New York, Viriginia, and California deaths
#Didn't end up using this was just testing something


New_York_predictors = US_predictors[US_predictors['State']=='New York']
California_predictors = US_predictors[US_predictors['State']=='California']
Virginia_predictors = US_predictors[US_predictors['State']=='Virignia']


# In[5]:


#Created a new dataframe that only had predictors I felt were related to health care/spending
US_predic_small = US_predictors[["Tested","Infected","Deaths","Population","ICU Beds","GDP","Physicians","Hospitals","Health Spending"]]


# In[6]:


#This is the aggregation of predictors for all 50 states
US_predic_small.describe()


# In[9]:


#Used to get the dates from the time series datasheet 
cols = US_deaths_df.keys()

US_deaths = US_deaths_df.loc[:, cols[12]:cols[-1]]
US_confirm = US_confirmed_df.loc[:,cols[12]:cols[-1]]


# In[24]:


#Creates a descending list of the total deaths in each state/province....its a bit ugly I know lol
All_US_Deaths_grouped = US_deaths_df[['Province_State',cols[-1]]]
x = (All_US_Deaths_grouped.groupby('Province_State')[cols[-1]].sum())
x.sort_values(ascending = False )


# In[19]:


#Total Sum of all deaths in the US up to April 25th 2020
All_Deaths = US_deaths_df[cols[-1]]
All_Deaths.sum()
All_Deaths


# In[ ]:


#Total sum of all confirmed cases up to April 25th 2020
All_Confirmed = US_confirmed_df[cols[-1]]
All_Confirmed.sum()


# In[11]:


#Stores the column headers to loop through for later usage
dates = US_deaths.keys()


# In[8]:


#Empty lists to store the amount of deaths for each state for each day
NY_Deaths = []
California_Deaths = []
Virginia_Deaths = []
US_Deaths_Tot = []
US_Confirmed_Tot = []


# In[12]:


#Creates the running sum for Death counts

for i in dates:
    NY_Deaths.append(US_deaths_df[US_deaths_df['Province_State']=='New York'][i].sum())
    California_Deaths.append(US_deaths_df[US_deaths_df['Province_State']=='California'][i].sum())
    Virginia_Deaths.append(US_deaths_df[US_deaths_df['Province_State']=='Virginia'][i].sum())
    US_Deaths_Tot.append(US_deaths[i].sum())
    US_Confirmed_Tot.append(US_confirm[i].sum())


# In[13]:


#Converts the dates into "Days since January 20"
days_since = np.array([i for i in range(len(dates))]).reshape(-1,1)


# In[11]:


#Visualiztion of the running sum of total deaths in New York (I say since 1/20/2020 since that was the initital starting point for the John Hopkins dataset)
plt.plot(days_since,NY_Deaths)
plt.xlabel('Days Since 1/20/2020')
plt.ylabel('Number of Cases')
plt.title('Number of Deaths in New York')

plt.show()


# In[12]:


#Visualiztion of the running sum of total deaths in California
plt.plot(days_since,California_Deaths)
plt.xlabel('Days Since 1/20/2020')
plt.ylabel('Number of Cases')
plt.title('Number of Deaths in California')
plt.show()


# In[17]:


#Visualiztion of the running sum of total confirmed in the United States
plt.figure(figsize=(6,6))
plt.plot(days_since,US_Confirmed_Tot)
plt.xlabel('Days Since 1/20/2020')
plt.ylabel('Number of Cases')
plt.title('Number of Confirmed Cases in the United States')
plt.show()


# In[18]:


#Visualiztion of the running sum of total deaths in California
plt.figure(figsize=(6,6))
plt.plot(days_since, US_Deaths_Tot)

plt.xlabel('Days Since 1/20/2020')
plt.ylabel('Number of Cases')
plt.title('Number of Deaths in the United States')
plt.show()


# In[15]:


#To create the visualizations for the forecasting visualizations had to do some formatting changes in the dates
new_dates = []
for date in dates:
    new_date = date.replace('/','-')
    new_dates.append(new_date)
    
  


# In[16]:


#Prophet dataframes have to have the date column named "ds" and the actual data (in this case the death counts) as "y"
Virginia_Deaths_df = pd.DataFrame({"ds":new_dates,"y":Virginia_Deaths})
Cali_Deaths_df = pd.DataFrame({"ds":new_dates,"y":California_Deaths})
NY_Deaths_df = pd.DataFrame({"ds":new_dates,"y":NY_Deaths})
US_Deaths_df = pd.DataFrame({"ds":new_dates,"y":US_Deaths_Tot})


# In[17]:


#Used to create the prophet visualizes I follwed the steps from https://facebook.github.io/prophet/docs/quick_start.html#python-api
Virginia_predic = Prophet()
Virginia_predic.fit(Virginia_Deaths_df)

NY_predic = Prophet()
NY_predic.fit(NY_Deaths_df)

Cali_predic = Prophet()
Cali_predic.fit(Cali_Deaths_df)

US_predic = Prophet()
US_predic.fit(US_Deaths_df)


# In[18]:


#Creates the "future" forecasting where periods represents 60 days from the end of the original dataset
Virginia_future = Virginia_predic.make_future_dataframe(periods=60)

Virginia_forceast = Virginia_predic.predict(Virginia_future)

fig = Virginia_predic.plot(Virginia_forceast,xlabel="Date",ylabel='Deaths')
ax = fig.gca()
ax.set_title("Predicted Deaths in 60 Days For Virginia")


# In[19]:


#Ignore this one 
fig2 = Virginia_predic.plot_components(Virginia_forceast)


# In[20]:


#Same thing but for New York deaths
NY_future = NY_predic.make_future_dataframe(periods=60)

NY_forceast = NY_predic.predict(NY_future)

fig = NY_predic.plot(NY_forceast,xlabel="Date",ylabel='Deaths')
ax = fig.gca()
ax.set_title("Predicted Deaths in 60 Days For New York")


# In[21]:


#Same thing but for California deaths
Cali_future = Cali_predic.make_future_dataframe(periods=60)

Cali_forceast = Cali_predic.predict(Cali_future)

fig = Cali_predic.plot(Cali_forceast,xlabel="Date",ylabel='Deaths')
ax = fig.gca()
ax.set_title("Predicted Deaths in 60 Days For California")


# In[22]:


#Same thing but for deaths in the whole United States
US_future = US_predic.make_future_dataframe(periods=60)

US_forceast = US_predic.predict(US_future)

fig = US_predic.plot(US_forceast,xlabel="Date",ylabel='Deaths')
ax = fig.gca()
ax.set_title("Predicted Deaths in 60 Days For the United States")


# In[27]:


#Creates the correlation matrix for the health care predictors
corrmatrix = US_predic_small.corr()


#Was testing a few different correlation visualizations didn't like this one but kept it in the code
sns.pairplot(US_predic_small)
sns.plt.show()


# In[35]:



#https://towardsdatascience.com/infectious-disease-modelling-part-i-understanding-sir-28d60e29fdfc this article goes into detail on how to use the model and it's really easy to follow.
#https://colab.research.google.com/github/hf2000510/infectious_disease_modelling/blob/master/part_one.ipynb took the code from here and changed the parameters for our needs
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# In[ ]:



#Represents the population of New York
N = 19440469

beta = 1.0  # infected person infects 1 other person per day
D = 14.0 # infections lasts for 14 days
gamma = 1.0 / D

S0, I0, R0 = (N-1), 1, 0  # initial conditions: SO is the suspecitble population, IO is those who are infected in this case we start with one person, and R0 is 0 because no one has recovered from the virus yet.


# In[ ]:


t = np.linspace(0, 94, 94) # Grid of time points (in days) this represents how the outbreak would look over 94 days
y0 = S0, I0, R0 # Initial conditions vector

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T


# In[31]:


#Used to plot the SIR model for New York 

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


# In[32]:


plotsir(t, S, I, R) #Calls the function to plot everything


# In[33]:


#Same code as above just changed to represent the entire U.S. population

#Represents the U.S. Population size
N = 328000000 
beta = 1.0  # infected person infects 1 other person per day
D = 14.0 # infections lasts four days
gamma = 1.0 / D

S0, I0, R0 = (N-1), 1, 0  # initial conditions: one infected, rest susceptible


# In[36]:


t = np.linspace(0, 94, 94) # Grid of time points (in days)
y0 = S0, I0, R0 # Initial conditions vector

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T


# In[37]:



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


# In[38]:


plotsir(t, S, I, R)


# In[ ]:


#Didn't use this correlation plot
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


#Didn't use this correlation plot either
mask = np.triu(np.ones_like(US_predic_small.corr(),dtype=np.bool))
f,ax = plt.subplots(figsize=(12,6))
cmap = sns.diverging_palette(220,10,as_cmap=True)
sns.heatmap(US_predic_small.corr(), mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)


# In[25]:


#Ended up liking this one you can look into the sns library and look at various cmap colors if you want to change how it looks
plt.subplots(figsize=(10,6))
plt.title("Health Care Predictors Correlations")
sns.heatmap(US_predic_small.corr(),xticklabels=US_predic_small.corr().columns,yticklabels=US_predic_small.corr().columns,cmap='BuGn')


# In[28]:


print(corrmatrix)

#Creates the same visualzation above in a table format
corrmatrix.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[ ]:





# In[ ]:




