#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from collections import defaultdict

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class PreProcessing:

    def __init__(self):
        self.df1,self.df2 = self.Read_Data()



    def Read_Data(self):
        df1 = pd.read_csv('athlete_events.csv')
        df2 = pd.read_csv('noc_regions.csv')
        return (df1,df2)

    def Print_MVs(self):
        return self.df1.isna().sum()


    def Impute_Age_Mean(self):
        ### Impute Age with mean from country
        data_age = self.df1[['NOC', 'Age']].copy()
        data_age = data_age.groupby(['NOC']).mean().reset_index()
        data_age.set_index('NOC')
        d = defaultdict()
        for index, row in data_age.iterrows():
            d[row['NOC']] = row['Age']

        count = 0
        ctr = 0
        nrows = len(self.df1.index)
        for index, row in self.df1.iterrows():
            if (pd.isnull(self.df1['Age'].iloc[ctr])):
                NOC = row['NOC']
                self.df1.at[ctr, 'Age'] = d.get(NOC)
                count = count + 1
            ctr = ctr + 1
            #progress.setValue(float(ctr / nrows) * 100)
        print("Number of updates", count)
        #End Impute_Age_Mean


    def Impute_Height_Mean(self):
        # Create New Dataframe data_height to create a dictionary for mean height lookup by country and gender

        data_height = self.df1[['NOC','Height','Sex']].copy()

        data_height = data_height.groupby(['NOC','Sex']).mean().reset_index()
        #create a new column to be used a dictionary key
        data_height['combined'] = data_height['NOC'].astype(str)+'_'+data_height['Sex'].astype(str)


        #create a dictionary for fast lookup key: NOC_SEX Value: Mean height

        d_height = defaultdict()
        for index, row in data_height.iterrows():
            d_height[row['combined']]=row['Height']

        #update missing height using the d_height dictionary

        count = 0
        ctr = 0
        for index, row in self.df1.iterrows():
            if(pd.isnull(self.df1['Height'].iloc[ctr])):
                key= row['NOC']+"_"+row['Sex']

                self.df1.at[ctr,'Height'] = d_height.get(key)
                count = count+1
            ctr = ctr+1
            #progress.setValue(float(ctr / nrows) * 100)
        print("Number of height updates",count)

        #End Impute_Height_Mean()

    def Impute_Weight_Mean(self):
        # Create New Dataframe data_height to create a dictionary for mean height lookup by country and gender
        data_weight = self.df1[['NOC','Weight','Sex']].copy()
        data_weight = data_weight.groupby(['NOC','Sex']).mean().reset_index()
        #create a new column to be used a dictionary key
        data_weight['combined'] = data_weight['NOC'].astype(str)+'_'+data_weight['Sex'].astype(str)
        #create a dictionary for fast lookup key: NOC_SEX Value: Mean height
        d_weight = defaultdict()
        for index, row in data_weight.iterrows():
            d_weight[row['combined']]=row['Weight']
        #update missing height using the d_height dictionary

        count = 0
        ctr = 0
        for index, row in self.df1.iterrows():
            if(pd.isnull(self.df1['Weight'].iloc[ctr])):
                key= row['NOC']+"_"+row['Sex']

                self.df1.at[ctr,'Weight'] = d_weight.get(key)
                count = count+1
            ctr = ctr+1
            #progress.setValue(float(ctr / nrows) * 100)
        print("Number of Weight updates",count)

    ## End Impute_Weight_Mean

    ### ADD MORE FUNCTIONS TO IMPUTE LIKE BELOW FUNCTION FOR INDIVIDUAL VARIABLES


    def Impute_All_Country(self):
        ### Impute Age, Weight, Height with mean from respective countries
        self.Impute_Age_Mean()
        self.Impute_Height_Mean()
        self.Impute_Weight_Mean()

    def Impute_Age_Dataset(self):
        mean_age = self.df1.Age.mean()
        for index,row in self.df1.iterrows():
            if (pd.isnull(self.df1['Age'].iloc[ctr])):
                self.df1.at[ctr, 'Age'] = mean_age
                count = count + 1

    def Impute_Weight_Dataset(self):
        mean_weight = self.df1.Weight.mean()
        for index,row in self.df1.iterrows():
            if (pd.isnull(self.df1['Weight'].iloc[ctr])):
                self.df1.at[ctr, 'Weight'] = mean_weight
                count = count + 1

    def Impute_Height_Dataset(self):
        mean_height = self.df1.Height.mean()
        for index,row in self.df1.iterrows():
            if (pd.isnull(self.df1['Height'].iloc[ctr])):
                self.df1.at[ctr, 'Height'] = mean_height
                count = count + 1

    def Impute_All_Dataset(self):
        self.Impute_Weight_Dataset()
        self.Impute_Height_Dataset()
        self.Impute_Age_Dataset()



    def Plot_Histogram(self,var,bin_num = 50):
        if var == '':
            pass
        else:
            return self.df1[var].hist(bins = bin_num)

    def Plot_Corr(self):
        corr = self.df1.corr()
        return sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,vmin=-1,vmax=1,annot=True)

    def Standardize(self):
        sc = StandardScaler()
        sc.fit(self.X_train)
        self.X_train = sc.transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
        #X_combined_std = np.vstack((X_train_std, X_test_std))
        #y_combined = np.hstack((y_train, y_test))

    def Train_Test_Split(self):
        X = self.df1[['Sex', 'Age', 'Height', 'Weight', 'Team', 'NOC', 'Year', 'Season', 'City', 'Event']].copy()
        y = self.df1['Medal']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=0)
