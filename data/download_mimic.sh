mimic_url="https://physionet.org/files/mimiciii/1.4/"
user_name="username"
wget -N -c -np --user ${user_name} --ask-password ${mimic_url}DIAGNOSES_ICD.csv.gz -P ./generate/data/mimic/
wget -N -c -np --user ${user_name} --ask-password ${mimic_url}D_ICD_DIAGNOSES.csv.gz -P ./generate/data/mimic/
wget -N -c -np --user ${user_name} --ask-password ${mimic_url}PATIENTS.csv.gz -P ./generate/data/mimic/
wget -N -c -np --user ${user_name} --ask-password ${mimic_url}ADMISSIONS.csv.gz -P ./generate/data/mimic/
wget -N -c -np --user ${user_name} --ask-password ${mimic_url}NOTEEVENTS.csv.gz -P ./generate/data/mimic/

gzip -d ./generate/data/mimic/DIAGNOSES_ICD.csv.gz
gzip -d ./generate/data/mimic/D_ICD_DIAGNOSES.csv.gz
gzip -d ./generate/data/mimic/PATIENTS.csv.gz
gzip -d ./generate/data/mimic/ADMISSIONS.csv.gz
gzip -d ./generate/data/mimic/NOTEEVENTS.csv.gz