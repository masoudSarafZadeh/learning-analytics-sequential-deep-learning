import joblib
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# ==========================================
# 1. Configuration & File Paths
# ==========================================
# NOTE: You must download the OULAD dataset from the official publisher website.
# Extract the files and update the paths below to point to your local directory.
BASE_PATH = "path/to/your/downloaded/oulad/anonymisedData"

# Base date for all time-series conversions
START_DATE = pd.to_datetime('2023-01-01')

# ==========================================
# 2. VLE Data Processing
# ==========================================
df_vle = pd.read_csv(f"{BASE_PATH}/vle.csv")
df_stuvle = pd.read_csv(f"{BASE_PATH}/studentVle.csv")

# Merge and clean
vle_stu = df_stuvle.merge(df_vle, how="outer", on="id_site")
vle_stu.drop(columns=['Unnamed: 0', 'code_module', 'code_presentation', 'id_site', 'week_from', 'week_to'], inplace=True, errors='ignore')
vle_stu.dropna(subset=['id_student'], inplace=True)

# Encode modules
label_cmcp = joblib.load('preprocess/label_encoder.joblib')
vle_stu['code_module - code_presentation'] = label_cmcp.transform(vle_stu['code_module - code_presentation'])
vle_stu['id_student'] = (vle_stu['id_student'] + (vle_stu['code_module - code_presentation'] / 100)) * 100
vle_stu.drop(columns=['code_module - code_presentation'], inplace=True)

# Format dates and set index for resampling
vle_stu['actual_date'] = START_DATE + pd.to_timedelta(vle_stu['date'] - 1, unit='D')
vle_stu.set_index('actual_date', inplace=True)
vle_stu.sort_values(by=['id_student', 'date'], inplace=True)
vle_stu.drop(columns=['date'], inplace=True)

# loop through activity types
activity_types = vle_stu['activity_type'].dropna().unique()
vle_dfs = []

for act in activity_types:
    # Filter for the specific activity
    df_act = vle_stu[vle_stu["activity_type"] == act].copy()
    
    # Resample weekly and sum
    df_act = df_act.groupby('id_student').resample('W').sum(numeric_only=True).replace(0., np.nan)
    df_act.reset_index(inplace=True)
    
    # Rename column
    df_act.rename(columns={'sum_click': f'sum_click_{act}'}, inplace=True)
    vle_dfs.append(df_act)

# Merge all activity dataframes dynamically
merged_df = reduce(lambda left, right: pd.merge(left, right, on=['id_student', 'actual_date'], how='outer'), vle_dfs)

# Calculate week number
merged_df['week_from_start'] = ((merged_df['actual_date'] - START_DATE).dt.days // 7) + 1

# ==========================================
# 3. Assessment Data Processing
# ==========================================
df_ass = pd.read_csv(f"{BASE_PATH}/assessments.csv")
df_stuass = pd.read_csv(f"{BASE_PATH}/studentAssessment.csv")
ass_stuass = df_stuass.merge(df_ass , on="id_assessment" , how="outer")
aca = ass_stuass.sort_values(by= ['id_student' , 'id_assessment'] , ascending=True)
aca.set_index('id_student' , inplace=True)
aca.dropna(subset=['date_submitted'] , inplace=True)

# Encode modules
aca['code_module - code_presentation'] = label_cmcp.transform(aca['code_module - code_presentation'])
aca['id_student'] = (aca['id_student'] + (aca['code_module - code_presentation'] / 100)) * 100
aca.drop(columns=['code_module - code_presentation'], inplace=True)

# Calculate weighted score
aca['score'] = aca['score'] * aca['weight'] / 100
aca.drop(columns=['Unnamed: 0', 'id_assessment', 'is_banked', 'date_submitted', 'weight'], inplace=True, errors='ignore')

# Format dates and set index for resampling
aca['actual_date'] = START_DATE + pd.to_timedelta(aca['date'] - 1, unit='D')
aca.set_index('actual_date', inplace=True)
aca.sort_values(by=['id_student', 'date'], inplace=True)

# Separate, resample, and merge TMA and CMA
aca_TMA = aca[aca["assessment_type"] == 'TMA'].groupby('id_student').resample('W').sum(numeric_only=True).replace(0., np.nan).reset_index()
aca_CMA = aca[aca["assessment_type"] == 'CMA'].groupby('id_student').resample('W').sum(numeric_only=True).replace(0., np.nan).reset_index()

aca_TMA.rename(columns={'score': 'score_TMA'}, inplace=True)
aca_CMA.rename(columns={'score': 'score_CMA'}, inplace=True)
aca_TMA.drop(columns=['date'], errors='ignore', inplace=True)
aca_CMA.drop(columns=['date'], errors='ignore', inplace=True)

merged_aca = pd.merge(aca_CMA, aca_TMA, on=['id_student', 'actual_date'], how='outer')
merged_aca.sort_values(by=['id_student', 'actual_date'], inplace=True)
merged_aca['week_from_start'] = ((merged_aca['actual_date'] - START_DATE).dt.days // 7) + 1

# Merge VLE and Assessments
merged_vle_aca = pd.merge(merged_aca, merged_df, on=['id_student', 'week_from_start'], how='outer')
merged_vle_aca.sort_values(by=['id_student', 'week_from_start'], inplace=True)

# ==========================================
# 4. Student Info & Demographics Processing
# ==========================================
df_info = pd.read_csv(f"{INFO_PATH}/studentInfo.csv")
df_reg = pd.read_csv(f"{INFO_PATH}/studentRegistration.csv")
df_all = pd.read_csv("preprocess/df_all.csv")

df_info['code_module - code_presentation'] = df_info[['code_module', 'code_presentation']].agg(' - '.join, axis=1)
df_info['code_module - code_presentation'] = label_cmcp.transform(df_info['code_module - code_presentation'])
df_info['id_student'] = (df_info['id_student'] + (df_info['code_module - code_presentation'] / 100)) * 100
df_info.drop(columns=['code_module - code_presentation', 'code_module', 'code_presentation', 'final_result'], inplace=True)

df_reg['code_module - code_presentation'] = df_reg[['code_module', 'code_presentation']].agg(' - '.join, axis=1)
df_reg['code_module - code_presentation'] = label_cmcp.transform(df_reg['code_module - code_presentation'])
df_reg['id_student'] = (df_reg['id_student'] + (df_reg['code_module - code_presentation']/100))*100
df_reg.drop(['code_module - code_presentation','code_module','code_presentation','date_registration'] , axis=1 , inplace=True)
df_reg['date_unregistration'] =df_reg['date_unregistration'].fillna(0)
df_reg['date_unregistration'] = df_reg['date_unregistration'].where(df_reg['date_unregistration'] == 0, 1)

df_info = df_info.merge(df_reg, how='left', on='id_student')
df_info = df_info.merge(df_all, how='right', on='id_student')

# Ordinal Encoders
cat_he = [['No Formal quals', 'Lower Than A Level', 'A Level or Equivalent', 'HE Qualification', 'Post Graduate Qualification']]
df_info['highest_education'] = OrdinalEncoder(categories=cat_he).fit_transform(df_info[['highest_education']])

cat_age = [['0-35', '35-55', '55<=']] # Fixed logical ordering
df_info['age_band'] = OrdinalEncoder(categories=cat_age).fit_transform(df_info[['age_band']])

cat_imd = [['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']]
df_info['imd_band'] = OrdinalEncoder(categories=cat_imd, handle_unknown='use_encoded_value', unknown_value=np.nan).fit_transform(df_info[['imd_band']])

df_info.fillna(-1, inplace=True)

# ==========================================
# 5. Final Dataset Merging & Encoding
# ==========================================
final_dataset = pd.merge(merged_vle_aca, df_info, on=['id_student'], how='left')
final_dataset.sort_values(by=['id_student', 'week_from_start'], inplace=True)
final_dataset.drop(columns=['actual_date_x', 'actual_date_y'], inplace=True, errors='ignore')

# Filter for active IDs
id_list = aca['id_student'].round(0).unique().tolist()
final_dataset['id_student'] = final_dataset['id_student'].round(0)
final_filtered = final_dataset[final_dataset['id_student'].isin(id_list)].copy()
final_filtered.fillna(0, inplace=True)

# Label Encoding for remaining categories
le = LabelEncoder()
for col in ['gender', 'region', 'disability']:
    if col in final_filtered.columns:
        final_filtered[col] = le.fit_transform(final_filtered[col].astype(str))

final_filtered['num_of_prev_attempts'] = final_filtered['num_of_prev_attempts'].astype(float)
final_filtered['studied_credits'] = final_filtered['studied_credits'].astype(float)

# Final Reference Info dataframe
df_info['id_student'] = df_info['id_student'].round(0)
df_info_filtered = df_info[df_info['id_student'].isin(id_list)].copy()
