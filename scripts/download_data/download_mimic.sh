PATH_TO_FOLDER=$1
mimic_url="https://physionet.org/files/mimiciii/1.4/"
user_name="username"
wget -N -c -np --user ${user_name} --ask-password ${mimic_url}DIAGNOSES_ICD.csv.gz -P ${PATH_TO_FOLDER}/generate/data/mimic/
wget -N -c -np --user ${user_name} --ask-password ${mimic_url}D_ICD_DIAGNOSES.csv.gz -P ${PATH_TO_FOLDER}/generate/data/mimic/
wget -N -c -np --user ${user_name} --ask-password ${mimic_url}PATIENTS.csv.gz -P ${PATH_TO_FOLDER}/generate/data/mimic/
wget -N -c -np --user ${user_name} --ask-password ${mimic_url}ADMISSIONS.csv.gz -P ${PATH_TO_FOLDER}/generate/data/mimic/
wget -N -c -np --user ${user_name} --ask-password ${mimic_url}NOTEEVENTS.csv.gz -P ${PATH_TO_FOLDER}/generate/data/mimic/

gzip -d ${PATH_TO_FOLDER}/generate/data/mimic/DIAGNOSES_ICD.csv.gz
gzip -d ${PATH_TO_FOLDER}/generate/data/mimic/D_ICD_DIAGNOSES.csv.gz
gzip -d ${PATH_TO_FOLDER}/generate/data/mimic/PATIENTS.csv.gz
gzip -d ${PATH_TO_FOLDER}/generate/data/mimic/ADMISSIONS.csv.gz
gzip -d ${PATH_TO_FOLDER}/generate/data/mimic/NOTEEVENTS.csv.gz


python format_mimic.py $PATH_TO_FOLDER