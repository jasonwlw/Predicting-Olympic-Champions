The code is assempled in two parts; a preprocessing module (Olympic_Analysis.py) with data preprocessing and modeling functions, and a gui script (Olympic_Event_GUI.py) that accesses the preprocessing module. 

To run, use a Python 3.7 Interpreter or Anaconda and make sure PyQt4 is installed. Run the GUI script as such:

python Olympic_Event_GUI.py

Home: This is the central tab for the GUI. It contains an option to reload the datafram and erase all changes made (Reset Data) and Variable Description, which pops up a window that describes the variables. 

The GUI has tabs for Data Imputation, Feature Selection, Data Transformation, Visualization, and Modeling.

Data Imputation: There are 6 buttons that correspond to different types and variables to impute. "By Country" means you will impute the values with the mean value from the respective country AND Sex. So the height of a female from the United States will be imputed with the average height of Olympian females from the United States. "By Dataset" means that you simply impute all missing values with the average from the entire dataset, irrespective of sex or country.

Feature Selection: There are two buttons, one that shows the correlation matrix of the data, and one that perform Principle Components Analysis (PCA). Note that after PCA the correlation plot will NOT change, it will still show the dataset correlations, as opposed to PCA. This is because Principle Components by definition are guaranteed to be orthogonal. 

Data Transformation: There are 4 Buttons here, one that will standardize all of the numeric variables, one that will drop data with a year less than a specified value, one that will upsample the medal class with replacement to balance classes, and one that will downsample the non-medal class with replacement to balance classes. Please note that if you upsample the medal class, you will not be able to subsequently downsample without resetting the data. Note that you cannot perform PCA and then subsequently standardize the variables. The code won't do it, which empasizes that you should standardize variables BEFORE performing PCA. 

Visualization: This tab visualizes the data. You can plot a histogram of any of the variables. Please note that after viewing data, you must change the selection and then change it back to the previous data in order to view it again. 

Modeling: This tab contains models and metrics. Decision Tree will fit the current data in memory after any adjustments with a random forest. You can set the max depth of the tree in the popup window. If you don't wish to enter a max depth, enter "0". Logistic Regression will perform a logistic regression on the data. Classification report will print to console the classification report from the model most recently ran, and Plot Confusion Matrix will pop up a window with the Confusion Matrix for the model most recently ran.  
