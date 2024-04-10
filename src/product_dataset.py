import pandas as pd
import numpy as np
import random
from random import uniform


def product_dataset():
    # Create an empty DataFrame
    raw_resume_data = pd.DataFrame()

    #Create new first name form a text file
    extra_fname = pd.read_csv('/home/farid/Documents/TAAV_vscode_prj/english_resume/src/First names.txt', sep=' ')
    #Change label column
    extra_fname.columns = ['First Name']
    #Shuffel enrties (already sorted by count of characters)
    extra_fname = extra_fname.sample(frac = 1)
    #Reset indexes
    extra_fname = extra_fname.reset_index()
    #Drop id column
    extra_fname.drop('index', axis=1, inplace=True)
    extra_fname.head()


    #Create new last name form a text file
    extra_lname = pd.read_csv('/home/farid/Documents/TAAV_vscode_prj/english_resume/src/Last names.txt', sep=' ')
    #Change label column
    extra_lname.columns = ['Last Name']
    #Shuffel enrties (already sorted by count of characters)
    extra_lname = extra_lname.sample(frac = 1)
    #Reset indexes
    extra_lname = extra_lname.reset_index()
    #Drop id column
    extra_lname.drop('index', axis=1, inplace=True)
    extra_lname.head()


    #Joining first name and last name into a dataframe
    extra_fullnames = pd.DataFrame()
    extra_fullnames['First Name'] = extra_fname
    extra_fullnames['Last Name'] = extra_lname
    extra_fullnames.head()


    #Importing first name and last name into resume dataset
    raw_resume_data = pd.concat([raw_resume_data,extra_fullnames],axis=0)
    raw_resume_data.head()


    extra_fields = pd.read_csv('/home/farid/Documents/TAAV_vscode_prj/english_resume/src/Job Recommendation System.csv')
    #Slicing fields
    extra_fields = extra_fields.iloc[:,2:11]
    #Repeating 180times per entry
    extra_fields = pd.DataFrame(np.repeat(extra_fields.values, 180, axis=0), columns=extra_fields.columns)
    #Shuffel enrties after 180times repeating
    extra_fields = extra_fields.sample(frac = 1).reset_index(drop=True)
    #Importing fields into resume dataset --> Age - Gender - Academic bg - Field of study - Skills - ...
    raw_resume_data = pd.concat([raw_resume_data,extra_fields], axis=1)
    raw_resume_data.head()


    extra_job_exp = pd.read_csv('/home/farid/Documents/TAAV_vscode_prj/english_resume/src/jobss.csv')
    #Slicing fields
    extra_job_exp = extra_job_exp.iloc[:,2:3]
    #Repeating 350times per entry
    extra_job_exp = pd.DataFrame(np.repeat(extra_job_exp.values, 350, axis=0), columns=extra_job_exp.columns)
    #Shuffel enrties after 350times repeating
    extra_job_exp = extra_job_exp.sample(frac = 1).reset_index(drop=True)
    #Importing Job Experience into resume dataset
    raw_resume_data = pd.concat([raw_resume_data,extra_job_exp], axis=1)
    raw_resume_data.head()


    extra_university = pd.read_csv('/home/farid/Documents/TAAV_vscode_prj/english_resume/src/list_of_univs.csv')
    #Slicing fields
    extra_university = extra_university.iloc[:,5:6]
    #Repeating 350times per entry
    extra_university = pd.DataFrame(np.repeat(extra_university.values, 15, axis=0), columns=extra_university.columns)
    #Shuffel enrties after 350times repeating
    extra_university = extra_university.sample(frac = 1).reset_index(drop=True)
    #Importing Job Experience into resume dataset
    raw_resume_data = pd.concat([raw_resume_data,extra_university], axis=1)
    raw_resume_data.rename(columns={"name": "University"}, inplace=True)
    raw_resume_data.head()


    #Product a list of average's person with 180000 score and 5000 random outliers between -1000 to +1000
    extra_average = [round(uniform(10,20),2) for i in range(180000)]
    extra_average.extend(map(lambda _: round(uniform(-1000, 1000), 2), range(5000)))
    #Shuffle averages
    random.shuffle(extra_average)
    extra_average = pd.Series(extra_average)
    raw_resume_data['Average'] = extra_average
    raw_resume_data.head()

    # Info of dataset
    raw_resume_data.info()

    # Export a csv file from DataFrame
    raw_resume_data.to_csv('/home/farid/Documents/TAAV_vscode_prj/english_resume/src/RawEnglishResume.csv', index=False)

    print('Dataset producted and save into src repository.')