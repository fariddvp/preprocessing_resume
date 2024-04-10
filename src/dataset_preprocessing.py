import pandas as pd
import numpy as np
import string
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import time


def dataset_preprocessing():

    # Read cleaned datasetv from previous phase
    data = pd.read_csv('/home/farid/Documents/TAAV_vscode_prj/english_resume/src/data_cleaned_final.csv')
    print('Step 19 --> Read dataframe from: data_cleaned_final.csv')


    # Removal of punctuations from tokens form
    def remove_punctuation(text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    # Remove punctuations from all columns
    for column in data.columns:
        data[column] = data[column].apply(remove_punctuation)
    print('Step 20 --> Removed all of punctuations from dataset values.')



    # Convert nan string values to np.nan
    def replace_nan_strings_with_nan(token):
        return token.apply(lambda x: np.nan if x == "nan" else x)

    # Apply on entire column
    for column in data.columns:
        data[column] = replace_nan_strings_with_nan(data[column])

    print('Step 21 --> Converted nan string values to np.nan .')



    # convert numeric columns to numeric: 'Age', 'Salary Expectation (in USD)', 'Average'
    data_numeric_columns = list(['Age', 'Salary Expectation (in USD)', 'Average'])
    for column in data_numeric_columns:
        data[column] = data[column].apply(pd.to_numeric, errors='coerce')

    print('Step 22 --> Converted numeric columns to numeric with pandas that already were string.')



    # Convert average to float with 2 digits 
    data['Average'] = data['Average'] / 100
    print('Step 23 --> Converted average to float with 2 digits.')



    # Assign min values : '2 5 yrs' => 2
    for index, item in enumerate(data['Job Experience Required']):
        
        if isinstance(item, str):
            parts = item.split()
            if len(parts) == 3:
    #             data['Job Experience Required'][index] = (int(parts[0]) + int(parts[1])) / 2
                data['Job Experience Required'][index] = min(int(parts[0]), int(parts[1]))

    print('Step 24 --> Assigned min value into Job Experience Required feature and overrided.')


    ##### Cleaning Job Experience Required'
    # Calculate mean of 'Job Experience Required' column if to be int value
    count = 0
    total = 0
    for item in data['Job Experience Required']:
        if isinstance(item, int):
            count += 1
            total += item
    avg = total//count

    # fill str and NaN values in 'Job Experience Required' with mean
    mask = (data['Job Experience Required'] == 'video') | (data['Job Experience Required'] == 'mention') | (data['Job Experience Required'].isna())
    data.loc[mask, 'Job Experience Required'] = avg

    print('Step 25 --> Preprocessed Job Experience Required feature and overrided.')


    #####'Age' cleaning
    # Calculate mean of 'Age' column if between 18-70
    data_filtered_age = data[(data['Age'] >= 18) & (data['Age'] <= 70)]
    avg_age = round(data_filtered_age['Age'].mean())
    # Fill nan values with average (if exist)
    data['Age'].fillna(avg_age, inplace=True)
    #Replacing outlier 'Age' with mean
    mask = (data['Age'] < 18) | (data['Age'] > 70)
    data.loc[mask, 'Age'] = avg_age

    print('Step 26 --> Preprocessed Age feature and overrided.')


    #####'Salary Expectation (in USD)' cleaning
    #Calculate mean from Salary between 1-99000
    data_filtered_salary = data[(data['Salary Expectation (in USD)'] > 0) & (data['Salary Expectation (in USD)'] < 100000)]
    avg_salary = round(data_filtered_salary['Salary Expectation (in USD)'].mean())
    #Filling nan values (if exist) with mean
    data['Salary Expectation (in USD)'].fillna(avg_salary, inplace=True)
    #Replacing outlier 'Salary Expectation (in USD)' with mean
    mask = (data['Salary Expectation (in USD)'] <= 0) | (data['Salary Expectation (in USD)'] >= 100000)
    data.loc[mask, 'Salary Expectation (in USD)'] = avg_salary

    print('Step 27 --> Preprocessed Salary Expectation (in USD) feature and overrided.')



    #####'Average' cleaning
    data_filtered_average = data[(data['Average']>=10) & (data['Average']<=20)]
    avg_average = round(data_filtered_average['Average'].mean())
    #Filling with mean
    data['Average'].fillna(avg_average, inplace=True)
    #Replacing outlier 'Age' with mean
    mask = (data['Average'] < 10) | (data['Average'] > 20)
    data.loc[mask, 'Average'] = avg_average
    data['Average'].value_counts()

    print('Step 28 --> Preprocessed Average feature and overrided.')



    # Assign nan values in 'Gender' with 'Unknown'
    data['Gender'].fillna('Unknown', inplace=True)

    # Assign numerical codes to the Gender categories
    # 0 for Male, 1 for Female, and 2 for Unknown
    data['Gender'] = data['Gender'].replace({'male': 0, 'female': 1, 'Unknown': 2})
    

    # Search and assign 0, 1 or 2 for values like 'male254645654'
    for index, item in enumerate(data['Gender']):
        if isinstance(item, str):
            if 'male' in item:
                data['Gender'][index] = 0
            elif 'female' in item:
                data['Gender'][index] = 1
            else:
                data['Gender'][index] = 'Unknown'

    print('Step 29 --> Preprocessed Gender feature and overrided.')


    # One-Hot-Encoding 'Academic Background' column
    # Create an instance of OneHotEncoder
    encoder = OneHotEncoder()

    # Fit and transform the encoder to the data
    encoded_data = encoder.fit_transform(data[['Academic Background']])

    # Convert the sparse matrix to a DataFrame
    encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(['Academic Background']))

    # Concatenate the original DataFrame with the one-hot encoded DataFrame
    data = pd.concat([data.reset_index(drop=True), encoded_df], axis=1)

    # Drop 'Academic Background' column
    data = data.drop(columns=['Academic Background'])
    
    print('Step 30 --> Preprocessed Academic Background feature with OHE and add new columns to dataframe.')



    # Label Encoding for 'Field of Study' column
    # Create an instance of LabelEncoder
    encoder = LabelEncoder()

    # Fit and transform the encoder to the 'field_study' column
    data['field_study_numerical'] = encoder.fit_transform(data['Field of Study'])

    # Drop Field of Study' column
    data = data.drop(columns=['Field of Study'])
    
    print('Step 31 --> Preprocessed Field of Study feature with LabelEncoder and add new columns to dataframe.')



    # Split the 'list_skills' column into lists of skills
    data['Skills'] = data['Skills'].str.split()

    # Perform one-hot encoding
    one_hot_encoded_skills = data['Skills'].str.join('|').str.get_dummies()

    # Concatenate the original DataFrame with the one-hot encoded DataFrame
    data = pd.concat([data, one_hot_encoded_skills], axis=1)

    # Drop the original 'list_skills' column
    data.drop(columns=['Skills'], inplace=True)

    print('Step 32 --> Preprocessed Skills feature with OHE and add new columns to dataframe.')




    # Label Encoding for ''Industry Interest' column
    # Create an instance of LabelEncoder
    encoder = LabelEncoder()

    # Fit and transform the encoder to the ''Industry Interest' column
    data['Industry Interest_numerical'] = encoder.fit_transform(data['Industry Interest'])

    # Drop 'Industry Interest' column
    data.drop(columns=['Industry Interest'], inplace=True)

    print('Step 33 --> Preprocessed Industry Interest feature with LabelEncoder and add new columns to dataframe.')


    # Assign numerical codes to the 'Job Type Interest' categories
    # 0 for multi, 1 for contract, and 2 for ireland , ...
    data['Job Type Interest'] = data['Job Type Interest'].replace({'multi': 0, 'contract': 1, 'ireland': 2, 'parti': 3, 'internship': 4})

    print('Step 34 --> Preprocessed Job Type Interest feature with ReplaceValues and override.')


    # Label Encoder for 'Location Interest'
    # Assign nan values in 'Location Interest' with 'Unknown'
    data['Location Interest'].fillna('Unknown', inplace=True)


    encoder = LabelEncoder()

    # Fit and transform the encoder to the ''Industry Interest' column
    data['Location Interest'] = encoder.fit_transform(data['Location Interest'])

    print('Step 35 --> Preprocessed Location Interest feature with LabelEncoder.')



    # Label Encoder for 'University'
    # Assign nan values in 'University' with 'Unknown'
    data['University'].fillna('Unknown', inplace=True)


    encoder = LabelEncoder()

    # Fit and transform the encoder to the ''Industry Interest' column
    data['University'] = encoder.fit_transform(data['University'])

    print('Step 36 --> Preprocessed University feature with LabelEncoder.')



    data.drop(columns=['First Name', 'Last Name'], inplace=True)

    print('Step 37 --> Deleted First Name and Last Name from dataset.')



    # Min-Max scaling dataset
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Scale the DataFrame
    scaled_data = scaler.fit_transform(data)

    # Convert scaled data back to DataFrame
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

    # Display the result
    scaled_df.to_csv('/home/farid/Documents/TAAV_vscode_prj/english_resume/src/data_final_minmmax.csv', index=False)

    print('Step 38 --> Scaled dataset with MinMaxScaler and exported a file: data_final_minmmax.csv.')

    # Standardization (Z-score Scaling)
    # Initialize StandardScaler
    scaler_z = StandardScaler()

    # Scale the DataFrame
    scaled_data_z = scaler_z.fit_transform(data)

    # Convert scaled data back to DataFrame
    scaled_df_zscore = pd.DataFrame(scaled_data_z, columns=data.columns)

    # Display the result
    scaled_df_zscore
    scaled_df_zscore.to_csv('/home/farid/Documents/TAAV_vscode_prj/english_resume/src/data_final_zscore.csv', index=False)

    print('Step 37 --> Scaled dataset with Z-score Scaling and exported a file: data_final_zscore.csv.')



