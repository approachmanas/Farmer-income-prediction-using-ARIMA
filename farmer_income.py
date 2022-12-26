#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pylab as plt


# In[2]:


data=pd.read_excel(r'C:\Farmer_Income_Predict\Dataset\dataset.xlsx')
data.describe()


# In[3]:


data['DATE']=pd.to_datetime(data['year'],infer_datetime_format=True)
index=data.set_index(['DATE'])
from datetime import datetime
index.head()


# In[4]:


plt.xlabel("Year")
plt.ylabel("Farmer Income")
plt.plot(data['year'],data['value'])
plt.savefig('sequence.png', dpi=600)


# In[5]:


from statsmodels.tsa.stattools import adfuller
print("Observations of Dickey-fuller test")
dftest = adfuller(data['value'],autolag='AIC')
dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
for key,value in dftest[4].items():
    dfoutput['critical value (%s)'%key]= value
print(dfoutput)


# In[6]:


data['df_log']=np.sqrt(data['value'])
data['df_diff_1']=data['df_log'].diff().dropna()
data['df_diff_1']=data['df_diff_1'].dropna()


# In[7]:


from statsmodels.tsa.stattools import adfuller
print("Observations of Dickey-fuller test")
dftest = adfuller(data['df_diff_1'].dropna(),autolag='AIC')
dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#lags used','number of observations used'])
for key,value in dftest[4].items():
    dfoutput['critical value (%s)'%key]= value
print(dfoutput)


# In[8]:


from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import het_white
import statsmodels.api as sm
#define response variable
y = data['year']

#define predictor variables
x = data[['value']]

#add constant to predictor variables
x = sm.add_constant(x)

#fit regression model
model = sm.OLS(y, x).fit()
#perform White's test
white_test = het_white(model.resid,  model.model.exog)

#define labels to use for output of White's test
labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']

#print results of White's test
print(dict(zip(labels, white_test)))


# In[9]:


data


# In[10]:


data.head()


# In[11]:


from scipy.stats import kruskal

def seasonality_test(series):
        seasoanl = False
        idx = np.arange(len(series.index)) % 12
        H_statistic, p_value = kruskal(series, idx)
        if p_value <= 0.05:
            seasonal = True
        return p_value, H_statistic
seasonality_test(data['value'])


# In[12]:


from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot
plot_acf(data['value'])
pyplot.xlabel('Lags')
pyplot.ylabel('ACF value')
plt.savefig('ACF_Raw.png', dpi=600)
pyplot.show()


# In[13]:


from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot
plot_pacf(data['value'])
pyplot.xlabel('Lags')
pyplot.ylabel('PACF value')
plt.savefig('PACF_Raw.png', dpi=600)
pyplot.show()


# In[14]:


data.head()


# In[15]:


from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot
plot_acf(data['df_diff_1'].dropna())
pyplot.xlabel('Lags')
pyplot.ylabel('ACF value')
plt.savefig('ACF_FDiff.png', dpi=600)
pyplot.show()


# In[16]:


from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot
plot_pacf(data['df_diff_1'].dropna())
pyplot.xlabel('Lags')
pyplot.ylabel('PACF value')
plt.savefig('PACF_FDiff.png', dpi=600)
pyplot.show()


# In[17]:


from statsmodels.tsa.arima_model import ARIMA  
mymodel = ARIMA(data['value'], order = (1, 1, 1))  
modelfit = mymodel.fit(disp = 0)  
print(modelfit.summary())  


# In[18]:


from statsmodels.tsa.arima_model import ARIMA  
mymodel = ARIMA(data['value'], order = (1, 1, 1))  
modelfit = mymodel.fit(disp = 0)  
print(modelfit.summary())  


# In[ ]:




