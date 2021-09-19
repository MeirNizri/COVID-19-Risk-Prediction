# COVID-19 Risk Prediction | ML Course Project

## Introduction
Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus. Most people infected with COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment. Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness.
During the entire course of the pandemic, one of the main problems that healthcare providers have faced is the shortage of medical resources and a proper plan to efficiently distribute them. In these tough times, being able to predict what kind of resource an individual might require at the time of being tested positive or even before that will be of great help to the authorities as they would be able to procure and arrange for the resources necessary to save the life of that patient.
Using the model described in this report healthcare providers will be able to prioritize patients effectively and thus reduce mortality rates.

## Project goals
The main goal of this project is to build a machine learning model that, given a Covid-19 patient's current symptom, status, and medical history, will predict whether the patient is in high risk or not. To this end, we will use several classification techniques in machine learning. 
-	K-Nearest Neighbours
-	Support Vector Machine
-	Decision Trees
-	Multilayer Perceptron
In each technique we will test a variety of hyperparameters values to get the best model.

In addition, we would like to analyze how each feature, which will be detailed below, affects the chances of getting a severe illness from Covid-19, and consequently understand who the populations at increased risk are.

## Dataset
The dataset for this project obtained from Kaggle (link). It was provided by the Mexican government (link). This dataset contains a huge number of anonymized patient-related information including pre-conditions. The raw dataset consists of 40 different features and 1,048,576 unique patients. Since the description of the data and features names was in Spanish, i had to first translate all the features names into English. Thereafter, the following actions were taken to make the data usable. 
-	All patients who haven't tested positive for COVID-19 were deleted.
-	Features with unnecessary and irrelevant information have been deleted. 
-	For features which have many conclusive values all rows with inconclusive value were filtered. 
-	For features which have a very few conclusive values the entire feature deleted. 
-	All the data values modified to mainly ones and zeroes to get it converted into one hot vector.
After processing and cleaning, the dataset consists of 20 features (detailed below) and 388,878 unique patients. The entire data is divided into three groups: train (90%), validation (5%) and test (5%).
1.	sex: female or male.
2.	age: of the patient.
3.	patient type: hospitalized or not hospitalized.
4.	pneumonia: Indicates whether the patient already have air sacs inflammation or not.
5.	pregnancy: Indicates whether the patient is pregnant or not.
6.	diabetes: Indicates whether the patient has diabetes or not.
7.	copd: Indicates whether the patient has Chronic obstructive pulmonary disease or not.
8.	asthma: Indicates whether the patient has asthma or not.
9.	inmsupr: Indicates whether the patient is immunosuppressed or not.
10.	hypertension: Indicates whether the patient has hypertension or not.
11.	cardiovascular: Indicates whether the patient has heart or blood vessels related disease.
12.	renal chronic: Indicates whether the patient has chronic renal disease or not.
13.	other disease: Indicates whether the patient has other disease or not.
14.	obesity: Indicates whether the patient is obese or not.
15.	tobacco: Indicates whether the patient is a tobacco user.
16.	usmr: Indicates whether the patient treated medical units of the first, second or third level.
17.	medical unit: type of institution of the National Health System that provided the care.
18.	intubed: Indicates whether the patient was connected to the ventilator.
19.	icu: Indicates whether the patient had been admitted to an Intensive Care Unit.
20.	death: indicates whether the patient died or recovered. 
The last three features serve as the label. That is, if a patient is intubed or treated in an intensive care unit or dies, he will be classified as at high risk (label 1).

