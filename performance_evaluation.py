import pandas as pd
import numpy as np
import math
import sklearn
import matplotlib.pyplot as plt
import re

#Nuevo
import pandas as pd
from sklearn import preprocessing

#Case Base comes from an Excel provided by the supervisor
path = r"C:\Users\emman\Desktop\Performance_cleaned.xlsx"
df = pd.read_excel(path)
Performance=df.loc[:,['Performance indicator', 'Performance', 'Publication Year']]
Performance['Performance'] = Performance['Performance'].astype(str)
Performance['Publication Year'] = Performance['Publication Year'].astype(str)
Performance['Performance indicator'] = Performance['Performance indicator'].astype(str)

pattern = re.compile(r'\s+')
Performance.to_excel('Performance.xlsx', index=False)
for a in range(0,len(Performance)):
    if Performance['Performance indicator'][a] == "" or Performance['Performance indicator'][a] == " ":
        Performance['Performance indicator'][a] = "None" 
    Performance['Performance indicator'][a] = re.sub(pattern, '', Performance['Performance indicator'][a])

for i in range(0,len(Performance)):
    indicator=Performance['Performance indicator'][i]
    indicator = re.sub(pattern, '', indicator)
    value=Performance['Performance'][i]
    if "," in indicator:
        value=value.replace(" ", "")
        indicator_list=indicator.replace(" ", "")
        indicator_list=indicator.split(",")
        value_list=value.split("),")
        if len(indicator_list)!=len(value_list):
            value_list=value_list[0:len(indicator_list)]
            value_list=[value_list[0]]*len(indicator_list)
        for j in range(1,len(indicator_list)):
            new_row = pd.DataFrame({'Performance indicator':[indicator_list[j]], 'Performance':[value_list[j]], 'Publication Year':[Performance['Publication Year'][i]]})
            Performance=pd.concat([Performance, new_row], ignore_index=True)
        Performance['Performance indicator'][i]=indicator_list[0]
        Performance['Performance'][i]=value_list[0]


#Clean the values
def clean_values(value):
    value=value.replace("(", "")
    value=value.replace(")", "")
    if "," in value:
        value=value.split(",")
        for i in range(0,len(value)):
            value[i]=value[i].strip("[]")
            value[i]=value[i].replace("[", "")
            value[i]=value[i].replace("]", "") 
            value[i]=float(value[i])
        value=np.mean(value)
    elif "-" in value and "[" in value:
        value=value.split("-")
        for i in range(0,len(value)):
            value[i]=value[i].strip("[]")
            value[i]=value[i].replace("[", "")
            value[i]=value[i].replace("]", "") 
            value[i]=float(value[i])
        value=np.mean(value)
    elif "±" in value:
        value=value.split("±")
        value=value[0]
        value=float(value)
    else:
        value=float(value)
    return value

performance_dict=Performance.to_dict(orient='index')

for i in performance_dict:
    if performance_dict[i]['Performance_indicator']=="Meanabsoluteprecentageerror" or  performance_dict[i]['Performance_indicator']=="Meanabsoluteerrorpercentage":
         performance_dict[i]['Performance_indicator']="Meanabsolutepercentageerror"
    performance_dict[i]['Performance']=clean_values(performance_dict[i]['Performance'])
    print(performance_dict[i]['Performance'])

Performance=pd.DataFrame.from_dict(performance_dict, orient='index')
#Create a dictionary with the indicators and their values
dict_indicators={}
for i in range(0,len(Performance)):
    indicator=Performance['Performance indicator'][i]
    value=Performance['Performance'][i]
    if indicator not in dict_indicators:
        dict_indicators[indicator]=[]
    dict_indicators[indicator].append(value)
        
#Clean the performance dataframe without separation
print("Done")

performance_separated_df=pd.DataFrame(list(dict_indicators.items()))
performance_separated_df.to_excel('performance_separated.xlsx')
print("Done")

normalized_performance={}
#Normalized data
for i in dict_indicators:
    if len(dict_indicators[i])>1:
        normalized_performance[i]=preprocessing.normalize([dict_indicators[i]])
    else:  
        normalized_performance[i]=preprocessing.normalize([[dict_indicators[i]]])

normalized_performance_df=pd.DataFrame(list(normalized_performance.items()))
normalized_performance_df.to_excel('normalized_values.xlsx')