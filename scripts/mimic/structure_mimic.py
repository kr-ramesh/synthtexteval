import pandas as pd
import ast

def create_dict_format(text_type = "mimic"):
    if(text_type == "mimic"):
      print("Mimic III:")
      text = "Diagnoses, <i>, Long_Title, ICD9_Code\nDemographic, <d>, Gender, Ethnicity"
    else:
      text = "Diagnoses, <i>, Diagnosis, ICD9_Code\nDemographic, <d>, Gender, Ethnicity"
    #Medication, <m>, Drug, Drug_Type

    dict_format = {}
    for category in text.split('\n'):
        category = [i.strip() for i in category.split(',')]
        dict_format[category[0]] = {k:[] for k in category[2:]}
        dict_format[category[0]]['token'] = category[1]

    return dict_format

def create_structured_format(dict_format, csv_file_path):

    #read csv and store information in dictionary
    df = pd.read_csv(csv_file_path)
    df['ICD9_CODE'] = df['ICD9_CODE'].apply(ast.literal_eval)
    df['LONG_TITLE'] = df['LONG_TITLE'].apply(ast.literal_eval)
    #creating a structured format of strings
    f_string = ""
    for k in dict_format.keys():
      #f_string = f_string + dict_format[k]['token'] + " "
      for sub_k in dict_format[k].keys():
        if(sub_k != 'token'):
          try:
            dict_format[k][sub_k] = [', '.join(string_list) if type(string_list)==list else string_list for string_list in df[sub_k.upper()].tolist()]
          except:
            dict_format[k][sub_k] = df[sub_k.upper()].tolist()
          f_string = f_string + "<" + sub_k + "> "
      #f_string = f_string + " " + dict_format[k]['token'].replace('<', '</') + "\n"
      f_string = f_string + "\n"
    try:
      #TODO: Check for all variables
      gender_mapping = {'M': 'Male', 'F': 'Female'}
      dict_format["Demographic"]["Gender"] = [gender_mapping[i] if i in gender_mapping.keys() else "Other" for i in dict_format["Demographic"]["Gender"]]
    except:
      print("No gender attribute.")

    format_strings = [f_string] * len(df)
    for info_type in dict_format.keys():
      for lst in dict_format[info_type].keys():
        if(lst == 'token'):
          continue
        format_strings = [f.replace('<'+ lst + '>', lst + ": " + str(attribute)) for f, attribute in zip(format_strings, dict_format[info_type][lst])]
    format_strings = [f + 'Text: ' for f in format_strings]
    return format_strings, dict_format['Diagnoses']['ICD9_Code'], dict_format['Diagnoses']['Long_Title'], df['TEXT'].tolist()

def structured_format(output_file_path, input_file_path):
    dict_format = create_dict_format()
    codes, icd9_codes, diag_titles, outputs = create_structured_format(dict_format = dict_format, csv_file_path = input_file_path)
    df = pd.DataFrame({'ICD9_CODE': icd9_codes, 'Diagnosis': diag_titles, 'LONG_TITLE': codes, 'TEXT': outputs})
    df.to_csv(output_file_path)