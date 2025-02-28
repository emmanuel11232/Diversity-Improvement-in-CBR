import pandas as pd
import numpy as np
import math
import sklearn
import matplotlib.pyplot as plt
import re

#Nuevo
import pandas as pd
from sklearn import preprocessing

#Remove paretentheses. If there are multiple values, the mean is calculated
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
    elif "[" in value :
        value=value.strip("[]")
        value=value.replace("[", "")
        value=value.replace("]", "")
        value=float(value)
    else:
        value=float(value)
    return value


path = r"C:\Users\emman\Documents\TEC\Diversity-Improvement-in-CBR\Datasets\Performance_cleaned.xlsx"
df = pd.read_excel(path)
Performance=df.loc[:,['Performance indicator', 'Performance', 'Publication Year']]
Performance['Performance'] = Performance['Performance'].astype(str)
Performance['Publication Year'] = Performance['Publication Year'].astype(str)
Performance['Performance indicator'] = Performance['Performance indicator'].astype(str)

#Eliminate the spaces in the performance indicator
pattern = re.compile(r'\s+')
Performance.to_excel('Performance.xlsx', index=False)
for a in range(0,len(Performance)):
    #Remove spaces
    Performance['Performance indicator'][a] = re.sub(pattern, '', Performance['Performance indicator'][a])
    Performance['Performance'][a] = re.sub(pattern, '', Performance['Performance'][a])
    #Separate the values
    Performance['Performance indicator'][a] = Performance['Performance indicator'][a].split(',')
    Performance['Performance'][a] = Performance['Performance'][a].split('),')
    if len(Performance['Performance indicator'][a])!=len(Performance['Performance'][a]):
        Performance['Performance'][a]=Performance['Performance'][a][0:len(Performance['Performance indicator'][a])]
        Performance['Performance'][a]=[Performance['Performance'][a][0]]*len(Performance['Performance indicator'][a])
    if len(Performance['Performance'][a])>1:
        for j in range(0,len(Performance['Performance'][a])):
            Performance['Performance'][a][j] = clean_values(Performance['Performance'][a][j])
                #Replace empty strings with "None"
            if Performance['Performance indicator'][a][j] == "" or Performance['Performance indicator'][a][j] == " " or Performance['Performance indicator'][a][j] == "nan":
                Performance['Performance indicator'][a][j] = "Not available"
            #Replace the performance indicator with the correct name
            if Performance['Performance indicator'][a][j]=="Meanabsoluteprecentageerror" or  Performance['Performance indicator'][a][j]=="Meanabsoluteerrorpercentage":
                Performance['Performance indicator'][a][j]="Meanabsolutepercentageerror"
    else:
        Performance['Performance'][a] = clean_values(Performance['Performance'][a][0])
                        #Replace empty strings with "None"
        if Performance['Performance indicator'][a][0] == "" or Performance['Performance indicator'][a][0] == " " or Performance['Performance indicator'][a][0] == "nan":
            Performance['Performance indicator'][a][0] = "Not available"
        #Replace the performance indicator with the correct name
        if Performance['Performance indicator'][a][0]=="Meanabsoluteprecentageerror" or  Performance['Performance indicator'][a][0]=="Meanabsoluteerrorpercentage":
            Performance['Performance indicator'][a][0]="Meanabsolutepercentageerror"
print(Performance)

#Now normalize and substitute
names_list=[]
for i in range(0,len(Performance)):
    if len(Performance['Performance indicator'][i])>1:
        for j in range(0,len(Performance['Performance'][i])):
            if Performance['Performance indicator'][i][j] not in names_list:
                names_list.append(Performance['Performance indicator'][i][j])
    else:
        if Performance['Performance indicator'][i][0] not in names_list:
            names_list.append(Performance['Performance indicator'][i][0])

for i in names_list:
    list_values=[]
    for j in range(0,len(Performance)):
        if len(Performance['Performance indicator'][j])>1:
            for k in range(0,len(Performance['Performance indicator'][j])):
                if Performance['Performance indicator'][j][k]==i:
                    list_values.append(Performance['Performance'][j][k])
        else:
            if Performance['Performance indicator'][j][0]==i:
                list_values.append(Performance['Performance'][j])        
    #Normalize
    list_values = np.array(list_values)
    list_values = list_values.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    list_values = min_max_scaler.fit_transform(list_values)
    list_values = list_values.tolist()
    #Replace the values
    count=0
    for j in range(0,len(Performance)):
        if len(Performance['Performance indicator'][j])>1:
            for k in range(0,len(Performance['Performance indicator'][j])):
                if Performance['Performance indicator'][j][k]==i:
                    Performance['Performance'][j][k]=list_values[count][0]
                    count=count+1
        else:
            if Performance['Performance indicator'][j][0]==i:
                Performance['Performance'][j]=list_values[count][0]
                count=count+1   

#Now normalize the publication Year
list_values=[]
for j in range(0,len(Performance)):
    list_values.append(Performance['Publication Year'][j])
list_values = np.array(list_values)
list_values = list_values.reshape(-1,1)
min_max_scaler = preprocessing.MinMaxScaler()
list_values = min_max_scaler.fit_transform(list_values)
list_values = list_values.tolist()
#Replace the values
count=0
for j in range(0,len(Performance)):
    Performance['Publication Year'][j]=list_values[count][0]
    count=count+1

print(Performance)
Performance.to_excel('performance_normalized.xlsx', index=False)

#Create average of the values
average_performance=[]
for i in range(0,len(Performance)):
    list_performance=[]
    if len(Performance['Performance indicator'][i])>1:
        for j in range(0,len(Performance['Performance'][i])):
            list_performance.append(Performance['Performance'][i][j])
        list_performance.append(Performance['Publication Year'][i])
        average_performance.append(np.mean(list_performance))
    else:
        list_performance.append(Performance['Publication Year'][i])
        list_performance.append(Performance['Performance'][i])
        average_performance.append(np.mean(list_performance))

Performance['Average_Performance']=average_performance
print(Performance)
Performance.to_csv('performance_normalized_averaged.csv', index=False)
