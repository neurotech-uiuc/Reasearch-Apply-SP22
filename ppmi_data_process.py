import pandas as pd

def get_ppmi_data():
  # load in csvs into dataframes
  lower_extremity_df = pd.read_csv('ppmi_lower_extremity_function_mobility.csv')
  upper_extremity_df = pd.read_csv('ppmi_upper_extremity_function_mobility.csv')
  participant_status_df = pd.read_csv('ppmi_participant_status.csv')

  # merge all dataframes into one
  extremity_df = pd.merge(lower_extremity_df, upper_extremity_df, how='inner', on='PATNO')
  df = pd.merge(participant_status_df, extremity_df, how='inner', on='PATNO')

  # take only relevant columns
  features_columns = ['NQMOB37', 'NQMOB30', 'NQMOB26', 'NQMOB32', 'NQMOB25', 'NQMOB33', 'NQMOB31', 'NQMOB28', 'NQUEX29', 'NQUEX20', 'NQUEX44', 'NQUEX36', 'NQUEX30', 'NQUEX28', 'NQUEX33', 'NQUEX37']
  df = df[['PATNO', 'COHORT'] + features_columns]
  # drop NaNs
  df = df.dropna()
  # drop rows with duplicate participants (keep the latest row)
  df.drop_duplicates(subset=['PATNO'], keep='last', inplace=True)
  # drop rows with cohorts that are neither healthy nor parkinsons
  df.drop(df[(df['COHORT'] != 1) & (df['COHORT'] != 2)].index, inplace = True)

  # extract label array
  participant_status = (df['COHORT'].values == 1).astype(int)
  # extract feature matrix
  all_features = df[features_columns]

  return participant_status, all_features