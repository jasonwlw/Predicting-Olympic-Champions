# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:13:58 2019

@author: Jing
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df1 = pd.read_csv('athlete_events.csv')
df2 = pd.read_csv('noc_regions_W&H.csv')

print(df1)

# In[31]:
# General Information
print(df1.info())
print(df1.isnull().sum())

print("All Countries:")
print(df1.Team.unique())

print("All Years:")
print(np.sort(df1.Year.unique()))

print("All Sports:")
print(df1.Sport.unique())

df1_gender=df1['Sex'].value_counts()
print(df1_gender.head())


#Separating dataset into 2 subsets by season: summner and winter
summer = df1[df1["Season"] == "Summer"]
winter = df1[df1["Season"] == "Winter"]

print ("Summer Olympic")
print("Total Sports    : ",  summer["Sport"].nunique())
print("Total Events    : ",  summer["Event"].nunique())
print("Total Countries : ",  summer["NOC"].nunique())
print("Total Sporters  : ",  summer["ID"].nunique())
print("Total Female Sporters : ",  summer[summer["Sex"] == "F"]["ID"].nunique())
print("Total Male Sporters   : ",  summer[summer["Sex"] == "M"]["ID"].nunique())

print ("Winter Olympic")
print("Total Sports    : ",  winter["Sport"].nunique())
print("Total Events    : ",  winter["Event"].nunique())
print("Total Countries : ",  winter["NOC"].nunique())
print("Total Sporters  : ",  winter["ID"].nunique())
print("Total Female Sporters : ",  winter[winter["Sex"] == "F"]["ID"].nunique())
print("Total Male Sporters   : ",  winter[winter["Sex"] == "M"]["ID"].nunique())


#Statistical Summary of 2 subsets: summer & winter
print(df1.describe())

print("Summer Description")
print(summer.describe())
print("Winter Description")
print(winter.describe())


# Participation by atheletes over 120 Years
import seaborn as sns

sum_c1 = summer.groupby(["Year"])["ID"].nunique().reset_index()
win_c1 = winter.groupby(["Year"])["ID"].nunique().reset_index()

fig = plt.figure(figsize=(16,20))
plt.subplot(211)
ax = sns.pointplot(x = sum_c1["Year"] , y = sum_c1["ID"], markers="h", color="red")
plt.xticks(rotation = 90)
ax.set_facecolor("w")
plt.grid(True,alpha=.2)
plt.ylabel("Number of Athletes",size=20)
plt.title("Athletes participated in Summer Olympics over 120 years",size=20,color="k")

plt.subplot(212)
ax1 = sns.pointplot(x = win_c1["Year"] , y = win_c1["ID"], markers="h",color = "blue")
plt.xticks(rotation = 90),
ax1.set_facecolor("w")
plt.grid(True,alpha=.2)
plt.ylabel("Number of Athletes",size=20)
plt.title("Athletes participated in Winter Olympics over 120 years",size=20, color="k")
plt.subplots_adjust(hspace = .3)
plt.show()


# Participation by gender over 120 Years
sum_gc = summer.groupby(["Year","Sex"])["ID"].nunique().reset_index()
win_gc = winter.groupby(["Year","Sex"])["ID"].nunique().reset_index()

fig = plt.figure(figsize=(16,20))
plt.subplot(211)
ax = sns.pointplot(x = sum_gc["Year"] , y = sum_gc["ID"], markers="h", palette = ["g","y"], hue = sum_gc["Sex"])
plt.grid(True)
plt.xticks(rotation = 90)
ax.set_facecolor("w")
plt.ylabel("Number of Athletes",size=22)
plt.title("Athletes by gender for Summer Olympics over 120 years",size=22,color="k")
plt.legend(loc = "best",prop={"size":16})

plt.subplot(212)
ax1 = sns.pointplot(x = win_gc["Year"] , y = win_gc["ID"], markers="h",palette = ["y","g"], hue = win_gc["Sex"])
plt.xticks(rotation = 90)
ax1.set_facecolor("w")
plt.grid(True)
plt.ylabel("Number of Athletes",size=22)
plt.title("Athletes by gender in Winter Olympics over 120 years",size=22,color="k")
plt.legend(loc = "best",prop={"size":16})
plt.subplots_adjust(hspace = .3)
plt.show()


# Participation by country over 120 years
plt.figure(figsize=(16,20))
plt.subplot(211)
ax = summer.groupby("Year")["NOC"].nunique().plot(kind = "bar",color="red",linewidth = 0.5,edgecolor="r" *summer["Year"].nunique())
plt.xticks(rotation = 90)
ax.set_facecolor("w")
plt.ylabel("Number of countries",size=20)
plt.title("Countries participated Summer Olympic over 120 years",size=20)
plt.grid(True,alpha=.3)

plt.subplot(212)
ax1 = winter.groupby("Year")["NOC"].nunique().plot(kind = "bar",color="blue",linewidth = 0.5,edgecolor="b" *summer["Year"].nunique())
plt.xticks(rotation = 90)
ax1.set_facecolor("w")
plt.ylabel("Number of countries",size=20)
plt.title("Countries participated Winter Olympic over 120 years",size=20)
plt.grid(True,alpha=.3)
plt.show()


# Age Distribution
fig, ax = plt.subplots(figsize=(25,8))
a = sns.boxplot(x="Year", y="Age", hue="Sex", palette={"M": "lightgreen", "F":"pink"}, data=df1[df1['Season']=='Summer'], ax=ax)     
ax.set_xlabel('Year', size=16, color="black")
ax.set_ylabel('Age', size=16, color="black")
ax.set_title('Age distribution in Summer Olympic', size=18, color="black")
plt.show()

fig, ax = plt.subplots(figsize=(25,8))
a = sns.boxplot(x="Year", y="Age", hue="Sex", palette={"M": "lightgreen", "F":"pink"}, data=df1[df1['Season']=='Winter'], ax=ax)     
ax.set_xlabel('Year', size=16, color="black")
ax.set_ylabel('Age', size=16, color="black")
ax.set_title('Age distribution in Winter Olympic', size=18, color="black")
plt.show()


# Height and Weight Distribiution of Athletes
metric_data1 = df1[(df1["Height"].notnull())]

plt.figure(figsize=(16,12))
sns.distplot(metric_data1["Height"],color = "g")
plt.axvline(metric_data1["Height"].mean(),linestyle = "dotted",linewidth = 3, color= "k",label = "Average Height")
plt.legend(loc = "best",prop = {"size" : 12})
plt.title("Distribution of Height",size=20)
plt.show()

metric_data2 = df1[(df1["Weight"].notnull())]

plt.figure(figsize=(16,12))
sns.distplot(metric_data2["Weight"],color = "y")
plt.axvline(metric_data2["Weight"].mean(),linestyle = "dotted",linewidth = 3, color= "k",label = "Average Weight")
plt.legend(loc = "best",prop = {"size" : 12})
plt.title("Distribution of Weight",size=20)
plt.show()


#Average of Attributes by Sports
import itertools
metric_data3 = df1[(df1["Height"].notnull()) & (df1["Weight"].notnull())]
cols = ['Age', 'Weight' ,'Height']
length = len(cols)
palette = ["r","g","b",]

sns.set_style("darkgrid")
plt.figure(figsize=(16,30))
for i,j,k in itertools.zip_longest(cols,range(length),palette) :
    plt.subplot(4,1,j+1)
    avg = metric_data3.groupby("Sport")[i].mean().reset_index()
    avg = avg.sort_values(by = i ,ascending =False)
    sns.barplot("Sport",i,data=avg[avg[i].notnull()],linewidth = .5,edgecolor = "w"*len(avg),color = k)
    plt.xticks(rotation = 90)
    plt.subplots_adjust(hspace = .6)
    plt.grid(True)
    plt.title("Average "+ i + "  by Sports",size=20)
    plt.xlabel("")
    

# Sports over 120 Years
sum_c2 = summer.groupby(["Year"])["Sport"].nunique().reset_index()
win_c2 = winter.groupby(["Year"])["Sport"].nunique().reset_index()

fig = plt.figure(figsize=(16,20))
plt.subplot(211)
ax = sns.pointplot(x = sum_c2["Year"] , y = sum_c2["Sport"],markers="h", color="red")
plt.xticks(rotation = 90)
ax.set_facecolor("w")
plt.grid(True,alpha=.2)
plt.ylabel("Sports")
plt.title("Sports in Summer Olympics over 120 years",size=20,color="k")

plt.subplot(212)
ax1 = sns.pointplot(x = win_c2["Year"] , y = win_c2["Sport"],markers="h",color = "blue")
plt.xticks(rotation = 90)
ax1.set_facecolor("w")
plt.grid(True,alpha=.2)
plt.ylabel("Sports")
plt.title("Sports in Winter Olympics over 120 years",size=20,color="k")
plt.subplots_adjust(hspace = .3)
plt.show()


#Events by gender over years
summer_ys = summer.groupby(["Year","Sex"])["Event"].nunique().reset_index()
plt.figure(figsize=(20,20))
plt.subplot(211)
sns.barplot("Year","Event", data=summer_ys,hue="Sex",linewidth = 1,palette = ["b","r"],edgecolor = "k"*summer_ys["Year"].nunique())
plt.xticks(rotation = 90)
plt.grid(True)
plt.legend(loc = "upper left",prop = {"size" : 20})
plt.title("Events by gender in summer olympics",size=20)

winter_ys = winter.groupby(["Year","Sex"])["Event"].nunique().reset_index()
plt.figure(figsize=(20,20))
plt.subplot(212)
sns.barplot("Year","Event",data=winter_ys,hue="Sex",linewidth = 1,palette = ["b","r"],edgecolor = "k"*winter_ys["Year"].nunique())
plt.xticks(rotation = 90)
plt.grid(True)
plt.legend(loc = "upper left",prop = {"size" : 20})
plt.title("Events by gender in winter olympics",size=20)
plt.show()


#Event by sport
summer_sp = pd.pivot_table(index="Sport",columns="Year",data=summer, values="Event",aggfunc="nunique")
summer_sp = summer_sp.fillna(0)
plt.figure(figsize=(20,20))
sns.heatmap(summer_sp,linewidth=0.5,annot=True,cmap="rainbow",linecolor="k")
plt.title("Summer olympics events by sports",color="b",size=20)
plt.show()

winter_sp = pd.pivot_table(index="Sport",columns="Year",data=winter,values="Event",aggfunc="nunique")
winter_sp = winter_sp.fillna(0)
plt.figure(figsize=(20,20))
sns.heatmap(winter_sp,linewidth=0.5,annot=True,cmap="rainbow",linecolor="k")
plt.title("Winter olympics events by sports",color="b",size=20)
plt.xticks(rotation = 90)
plt.show()


# Top 30 countries with the maximum number of medals 
t30_summer = df1[(df1['Season']=='Summer') & (df1['Medal']!='No Medal')].groupby('Team').count().reset_index()[['Team','Medal']].sort_values('Medal', ascending=False).head(30)
f, ax = plt.subplots(figsize=(16, 12))
sns.barplot(x="Medal", y="Team", data=t30_summer, label="Team", color="red")

for p in ax.patches:
    ax.text(p.get_width() + 125,p.get_y() + (p.get_height()/2) + .1,'{:1.0f}'.format(p.get_width()),ha="center")
ax.set_xlabel('Team', size=14, color="black")
ax.set_ylabel('Total Medals', size=14, color="black")
ax.set_title('Top 30 countries with total medals in Summer Olympic', size=18, color="black")
plt.show()

t30_winter = df1[(df1['Season']=='Winter') & (df1['Medal']!='No Medal')].groupby('Team').count().reset_index()[['Team','Medal']].sort_values('Medal', ascending=False).head(30)
f, ax = plt.subplots(figsize=(16, 12))
sns.barplot(x="Medal", y="Team", data=t30_winter, label="Team", color="blue")

for p in ax.patches:
    ax.text(p.get_width() + 20,p.get_y() + (p.get_height()/2) + .1,'{:1.0f}'.format(p.get_width()),ha="center")
ax.set_xlabel('Team', size=14, color="black")
ax.set_ylabel('Total Medals', size=14, color="black")
ax.set_title('Top 30 countries with total medals in Winter Olympic', size=18, color="black")
plt.show()


# Medal by Sport for USA
usa_summer = df1[(df1['Season']=='Summer') & (df1['Team'].isin(['United States'])) & (df1['Medal']!='NA')]
usa_summer = pd.pivot_table(usa_summer, index=['Sport'], columns=['Year'], values=['ID'],  aggfunc=len, fill_value=0)
usa_summer = usa_summer.reindex(usa_summer['ID'].sort_values(by=2016, ascending=False).index)

f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(usa_summer, annot=True, linewidths=0.06, ax=ax, cmap="GnYlRd")
ax.set_xlabel('Year', size=12, color="black")
ax.set_ylabel('Sport', size=12, color="black")
ax.set_title('USA Medals by Sports in Summer Olympic over Years', size=20, color="black")
plt.show()

usa_winter = df1[(df1['Season']=='Winter') & (df1['Team'].isin(['United States'])) & (df1['Medal']!='NA')]
usa_winter = pd.pivot_table(usa_winter, index=['Sport'], columns=['Year'], values=['ID'],  aggfunc=len, fill_value=0)
usa_winter = usa_winter.reindex(usa_winter['ID'].sort_values(by=2016, ascending=False).index)

f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(usa_winter, annot=True, linewidths=0.06, ax=ax, cmap="GnYlRd")
ax.set_xlabel('Year', size=12, color="black")
ax.set_ylabel('Sport', size=12, color="black")
ax.set_title('USA Medals by Sports in Winter Olympic over Years', size=18, color="black")
plt.show()