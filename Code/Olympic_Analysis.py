#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from collections import defaultdict
import itertools

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

class PreProcessing:

    def __init__(self):
        self.df1,self.df2 = self.Read_Data()
        self.df1_nona = self.df1.dropna()
        self.train_cols = ['Sex', 'Age', 'Height', 'Weight', 'Team', 'NOC', 'Year', 'Season', 'City', 'Event']
        self.Encode_Medal()
        ### Encode other variables here
        self.le = LabelEncoder()
        self.Encode_Cats()
        self.sc = StandardScaler()
        self.Train_Test_Split()

        #self.sc.fit(self.X_train)


    def Read_Data(self):
        df1 = pd.read_csv('athlete_events.csv')
        df2 = pd.read_csv('noc_regions.csv')
        return (df1,df2)

    def Encode_Medal(self):
        mask = self.df1['Medal'].isna()
        self.df1.loc[mask,'Medal'] = 0
        mask = np.logical_not(mask)
        self.df1.loc[mask,'Medal'] = 1

    def Encode_Cats(self):
        self.df1[["Sex","Team","NOC","Season","City","Event"]] \
            = self.df1[["Sex","Team","NOC","Season","City","Event"]].apply(self.le.fit_transform)

    def Print_MVs(self):
        print(self.df1.isna().sum())


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
        self.Train_Test_Split()
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
                key= str(row['NOC'])+"_"+str(row['Sex'])

                self.df1.at[ctr,'Height'] = d_height.get(key)
                count = count+1
            ctr = ctr+1
            #progress.setValue(float(ctr / nrows) * 100)
        print("Number of height updates",count)
        self.Train_Test_Split()

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
                key= str(row['NOC'])+"_"+str(row['Sex'])

                self.df1.at[ctr,'Weight'] = d_weight.get(key)
                count = count+1
            ctr = ctr+1
            #progress.setValue(float(ctr / nrows) * 100)
        print("Number of Weight updates",count)
        self.Train_Test_Split()

    ## End Impute_Weight_Mean

    ### ADD MORE FUNCTIONS TO IMPUTE LIKE BELOW FUNCTION FOR INDIVIDUAL VARIABLES


    def Impute_All_Country(self):
        ### Impute Age, Weight, Height with mean from respective countries
        self.Impute_Age_Mean()
        self.Impute_Height_Mean()
        self.Impute_Weight_Mean()

    def Impute_Age_Dataset(self):
        mean_age = self.df1.Age.mean()
        count = 0
        for index,row in self.df1.iterrows():
            if (pd.isnull(self.df1['Age'].iloc[count])):
                self.df1.at[count, 'Age'] = mean_age
                count = count + 1
        self.Train_Test_Split()

    def Impute_Weight_Dataset(self):
        mean_weight = self.df1.Weight.mean()
        count = 0
        for index,row in self.df1.iterrows():
            if (pd.isnull(self.df1['Weight'].iloc[count])):
                self.df1.at[count, 'Weight'] = mean_weight
                count = count + 1
        self.Train_Test_Split()

    def Impute_Height_Dataset(self):
        mean_height = self.df1.Height.mean()
        count = 0
        for index,row in self.df1.iterrows():
            if (pd.isnull(self.df1['Height'].iloc[count])):
                self.df1.at[count, 'Height'] = mean_height
                count = count + 1
        self.Train_Test_Split()

    def Impute_All_Dataset(self):
        self.Impute_Weight_Dataset()
        self.Impute_Height_Dataset()
        self.Impute_Age_Dataset()



    def Plot_Histogram(self,var,bin_num = 50):
        if var == '':
            pass
        elif var == 'Medal':
            return self.df1[var].hist()
        else:
            return self.df1[var].hist(bins = bin_num)

    def Plot_Corr(self):
        corr = self.df1.corr()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        return sns.heatmap(corr,mask=mask, cmap=cmap, xticklabels=corr.columns,
                           yticklabels=corr.columns, center=0, square=True,linewidths=0.5,
                           vmin=-1,vmax=1,annot=False,cbar_kws={"shrink": 0.5})

    def PCA_Transform(self,comp = 10):
        comp = int(comp)
        pca = PCA(n_components=comp, whiten=True)
        #df_check = self.df1.dropna()
        #if df_check.shape != self.df1.shape:
            #self.df1.dropna(inplace=True)
            #self.Train_Test_Split()
        self.X_train = pca.fit_transform(self.X_train)
        #self.df1[self.train_cols] = pca.transform(self.df1[self.train_cols])
        self.X_test = pca.transform(self.X_test)
        self.sc.fit(self.X_train)
        explained_variance = pca.explained_variance_ratio_
        explained_variance_sum = pca.explained_variance_ratio_.cumsum()
        sns.set()
        plt.title("PCA")
        plt.xlabel("Number of Components")
        plt.ylabel("Variance Explained")
        sns.scatterplot(np.arange(1,comp+1),explained_variance_sum)
        plt.show()

    def Standardize(self):
        cols = self.df1.select_dtypes(include=['float64']).columns.values
        self.df1[cols] = self.sc.transform(self.df1[cols])
        self.X_train[cols] = self.sc.transform(self.X_train[cols])
        self.X_test[cols] = self.sc.transform(self.X_test[cols])
        #X_combined_std = np.vstack((X_train_std, X_test_std))
        #y_combined = np.hstack((y_train, y_test))

    def Train_Test_Split(self):
        self.df1_nona = self.df1.dropna()
        X = self.df1_nona[self.train_cols].copy()
        y = self.df1_nona['Medal']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        cols = self.df1.select_dtypes(include=['float64']).columns.values
        self.sc.fit(self.X_train[cols])

    def Drop_Years(self,split):
        self.df1.drop(self.df1[self.df1.Year < split].index,inplace = True)
        ### Redo Train Test Split to update all used dataframes
        self.Train_Test_Split()

    def Upsampling(self):
        #self.train = pd.concat([self.X_train,self.y_train],axis = 1,sort = False)
        #df_majority = self.train[self.train.Medal == 0]
        #df_minority = self.train[self.train.Medal == 1]
        #n_samples = self.train.Medal.value_counts()[0]
        df_majority = self.df1[self.df1.Medal == 0]
        df_minority = self.df1[self.df1.Medal == 1]
        n_samples = self.df1.Medal.value_counts()[0]
        df_minority_upsampled = resample(df_minority,
                                         replace=True,  # sample with replacement
                                         n_samples=n_samples,  # to match majority class
                                         random_state=123) # reproducible results
        #self.train = pd.concat([df_majority,df_minority_upsampled])
        self.df1 = pd.concat([df_majority,df_minority_upsampled])
        #self.train = self.train.reset_index()
        #self.test = pd.concat([self.X_test,self.y_test],axis = 1, sort = False)
        #self.X_train = self.train[self.train_cols]
        #self.y_train = self.train['Medal']
        #self.df1 = pd.concat([self.train,self.test])
        self.Train_Test_Split()

    def Downsampling(self):
        #self.train = pd.concat([self.X_train,self.y_train],axis = 1,sort = False)
        #df_majority = self.train[self.train.Medal == 0]
        #df_minority = self.train[self.train.Medal == 1]
        #n_samples = self.train.Medal.value_counts()[1]
        df_majority = self.df1[self.df1.Medal == 0]
        df_minority = self.df1[self.df1.Medal == 1]
        n_samples = self.df1.Medal.value_counts()[1]
        df_majority_downsampled = resample(df_majority,
                                         replace=True,  # sample with replacement
                                         n_samples=n_samples,  # to match majority class
                                         random_state=123) # reproducible results
        self.df1 = pd.concat([df_majority_downsampled,df_minority])
        #self.train = self.train.reset_index()
        #self.test = pd.concat([self.X_test,self.y_test],axis = 1, sort = False)
        #self.X_train = self.train[self.train_cols]
        #self.y_train = self.train['Medal']
        #self.df1 = pd.concat([self.train,self.test],axis=0,sort=False).reset_index()
        self.Train_Test_Split()

    def Decision_Trees(self,max_depth):
        self.clf = RandomForestClassifier(random_state=123,max_depth=max_depth)
        self.clf.fit(self.X_train, self.y_train)
        self.y_pred = self.clf.predict(self.X_test)

    def Log_Reg(self):
        self.clf = LogisticRegression(C=1000.0, random_state=0)
        ### Drop any null
        df_check = self.df1.dropna()
        #if df_check.shape != self.df1.shape:
            #self.df1.dropna(inplace=True)
            #self.Train_Test_Split()
        self.clf.fit(self.X_train, self.y_train)
        self.y_pred = self.clf.predict(self.X_test)

    def Plot_Confusion_Matrix(self, normalize=False):  # This function prints and plots the confusion matrix.
        cm = confusion_matrix(self.y_test, self.y_pred, labels=[0, 1])
        classes = ["Will Lose", "Will Win"]
        cmap = plt.cm.Blues
        title = "Confusion Matrix"
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.around(cm, decimals=3)
        fig = plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        thresh = cm.max() / 1.5
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color = "black")
            #color = "white" if cm[i, j] > thresh else "black"
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def Class_Report(self):
        print(classification_report(self.y_test,self.y_pred))