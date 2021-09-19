#import numpy as np
import pandas as pd

class Covid19_Dataset(object):
    def __init__(self):
        self.covid = pd.read_csv("Dataset/Covid Data - clean.csv", low_memory = False)

    def process_and_clean(self):
        # Cleaning the data to keep only the rows containing 1, 2. values as 97 and 99 are essentialling missing data
        self.covid = self.covid.loc[(self.covid.CLASIFFICATION_FINAL < 4)]
        self.covid = self.covid.loc[(self.covid.SEX == 1) | (self.covid.SEX == 2)]
        self.covid = self.covid.loc[(self.covid.USMER == 1) | (self.covid.USMER == 2)]
        self.covid = self.covid.loc[(self.covid.PATIENT_TYPE == 1) | (self.covid.PATIENT_TYPE == 2)]
        self.covid = self.covid.loc[(self.covid.PNEUMONIA == 1) | (self.covid.PNEUMONIA == 2)]
        self.covid = self.covid.loc[(self.covid.DIABETES == 1) | (self.covid.DIABETES == 2)]
        self.covid = self.covid.loc[(self.covid.COPD == 1) | (self.covid.COPD == 2)]
        self.covid = self.covid.loc[(self.covid.ASTHMA == 1) | (self.covid.ASTHMA == 2)]
        self.covid = self.covid.loc[(self.covid.INMSUPR == 1) | (self.covid.INMSUPR == 2)]
        self.covid = self.covid.loc[(self.covid.HIPERTENSION == 1) | (self.covid.HIPERTENSION == 2)]
        self.covid = self.covid.loc[(self.covid.OTHER_DISEASE == 1) | (self.covid.OTHER_DISEASE == 2)]
        self.covid = self.covid.loc[(self.covid.CARDIOVASCULAR == 1) | (self.covid.CARDIOVASCULAR == 2)]
        self.covid = self.covid.loc[(self.covid.OBESITY == 1) | (self.covid.OBESITY == 2)]
        self.covid = self.covid.loc[(self.covid.RENAL_CHRONIC == 1) | (self.covid.RENAL_CHRONIC == 2)]
        self.covid = self.covid.loc[(self.covid.TOBACCO == 1) | (self.covid.TOBACCO == 2)]

        # Modifying data to get it converted to One Hot Encoded data
        self.covid.SEX = self.covid.SEX.apply(lambda x: x if x == 1 else 0)  
        self.covid.USMER = self.covid.USMER.apply(lambda x: x if x == 1 else 0)                     # no = 0, yes = 1
        self.covid.PATIENT_TYPE = self.covid.PATIENT_TYPE.apply(lambda x: 0 if x == 1 else 1)     
        self.covid.PNEUMONIA = self.covid.PNEUMONIA.apply(lambda x: x if x == 1 else 0)           
        self.covid.DIABETES = self.covid.DIABETES.apply(lambda x: x if x == 1 else 0)             
        self.covid.COPD = self.covid.COPD.apply(lambda x: x if x == 1 else 0)                     
        self.covid.ASTHMA = self.covid.ASTHMA.apply(lambda x: x if x == 1 else 0)                 
        self.covid.INMSUPR = self.covid.INMSUPR.apply(lambda x: x if x == 1 else 0)               
        self.covid.HIPERTENSION = self.covid.HIPERTENSION.apply(lambda x: x if x == 1 else 0)     
        self.covid.OTHER_DISEASE = self.covid.OTHER_DISEASE.apply(lambda x: x if x == 1 else -0)  
        self.covid.CARDIOVASCULAR = self.covid.CARDIOVASCULAR.apply(lambda x: x if x == 1 else 0) 
        self.covid.OBESITY = self.covid.OBESITY.apply(lambda x: x if x == 1 else 0)               
        self.covid.RENAL_CHRONIC = self.covid.RENAL_CHRONIC.apply(lambda x: x if x == 1 else 0)   
        self.covid.TOBACCO = self.covid.TOBACCO.apply(lambda x: x if x == 1 else 0)               
        self.covid.DATE_DIED = self.covid.DATE_DIED.apply(lambda x: 0 if x == "9999-99-99" else 1)
        self.covid.PREGNANT = self.covid.PREGNANT.apply(lambda x: x if x == 1 else 0)           
        self.covid.INTUBED = self.covid.INTUBED.apply(lambda x: x if x == 1 else 0)                   
        self.covid.ICU = self.covid.ICU.apply(lambda x: x if x == 1 else 0)

        # Divide age column to 12 age groups
        # for i in range(1,13):
        #   covid['age_group_%d' %i] = covid['AGE']
        #   covid['age_group_%d' %i]  = covid['age_group_%d' %i].apply(lambda x: 1 if (x>=(i-1)*10 and x<i*10) else 0) 

        # Creating the label column from summing three columns of the data. This column represents whether the patient is at risk from covid.
        self.covid['AT_RISK'] = self.covid['DATE_DIED']+self.covid['INTUBED']+self.covid['ICU']
        self.covid.AT_RISK = self.covid.AT_RISK.apply(lambda x: 1 if x > 0 else 0) 

        # Drop a few columns which are intuitively not longer useful
        self.covid.drop(columns = ['CLASIFFICATION_FINAL', 'INTUBED', 'ICU', 'DATE_DIED'], inplace=True)

        # Normalize the cleaned data
        # covid = (covid-covid.mean())/covid.std()
        # covid.AT_RISK = covid.AT_RISK.apply(lambda x: 1 if x > 0 else 0) 
        
        return self.covid