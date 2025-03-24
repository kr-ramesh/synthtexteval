import sys
import pandas as pd

# Create mapping of all diagnoses to the full form of their ICD9 CODE
df1 = pd.read_csv(f"{sys.argv[1]}DIAGNOSES_ICD.csv")
df2 = pd.read_csv(f"{sys.argv[1]}D_ICD_DIAGNOSES.csv")
df1 = df1.drop(columns=['ROW_ID'])
df1 = pd.merge(df2, df1, on='ICD9_CODE', how = 'inner')
df1 = df1.drop(columns=['ROW_ID'])
print("Merging diagnoses table...")

# Patients are assigned a SUBJECT_ID. Mapping to their diagnoses is created.
df2 = pd.read_csv(f"{sys.argv[1]}PATIENTS.csv")
df1 = pd.merge(df2, df1, on='SUBJECT_ID', how = 'inner')
df1 = df1.drop(columns=['ROW_ID'])
print("Merging patients with their hospital admissions and ICD9-CODES....")

# Patients can have  multiple admissions. Every admission is linked to the patient associated with the HADM_ID.
df2 = pd.read_csv(f"{sys.argv[1]}ADMISSIONS.csv")
df1 = pd.merge(df2, df1, on= 'HADM_ID', how = 'inner')
df1 = df1.drop(columns=['ROW_ID'])
print("Merging admission details with patients....")

# Every admission can have multiple note events associated with it.
df2 = pd.read_csv(f"{sys.argv[1]}NOTEEVENTS.csv")
df1 = pd.merge(df2, df1, on = 'HADM_ID', how = 'inner')
df1.to_csv(f"{sys.argv[1]}MIMIC-NOTES.csv")
df1 = df1.drop(columns=['ROW_ID'])
print("All merges completed!")

