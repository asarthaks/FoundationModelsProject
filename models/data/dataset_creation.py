"""
Script Name: dataset_creation.py
Author: Sarthak Singh, Yassir El Attar
Date: 2024-10-29
Description:
    This script reads a CSV file, and map the skin lesion abbreviations with their corresponding definition.
    It reads the csv file from the original dataset designed for PanDerm and outputs a .json file containing the mapping.
    Two of the skin lesions were dropped because of lack of related textual data.
    

Dependencies:
    - pandas
    - json

"""

import pandas as pd
import json

isic_training_gt = pd.read_csv("../../../ISIC2018_Task3_Training_GroundTruth.csv")

diagnosis_definitions = {
    "MEL": ["""Melanoma. Melanoma is a malignant neoplasm derived from melanocytes that may appear in different variants. If
excised in an early stage it can be cured by simple surgical excision. Melanomas can be invasive or noninvasive (in situ). Melanomas are usually, albeit not always, chaotic, and some melanoma specific criteria depend on
anatomic site.""", "Melanoma"],
    "NV": ["""Melanocytic nevus. Melanocytic nevi are benign neoplasms of melanocytes and appear in a myriad of variants. In
contrast to melanoma they are usually symmetric with regard to the distribution of color and structure""", "Melanocytic nevus"],
    "BCC": ["""Basal cell carcinoma. Basal cell carcinoma is a common variant of epithelial skin cancer that rarely metastasizes but grows
destructively if untreated. It appears in different morphologic variants (flat, nodular, pigmented, cystic).""", "Basal cell carcinoma"],
    "AK": ["""Actinic keratosis. Actinic Keratoses (Solar Keratoses) and Intraepithelial Carcinoma (Bowen's disease) are common noninvasive, variants of squamous cell carcinoma that can be treated locally without surgery. Some authors
regard them as precursors of squamous cell carcinomas and not as actual carcinomas. There is, however,
agreement that these lesions may progress to invasive squamous cell carcinoma - which is usually not
pigmented. Both neoplasms commonly show surface scaling and commonly are devoid of pigment. Actinic keratoses are more common on the face and Bowen's disease is more common on other body
sites. Because both types are induced by UV-light the surrounding skin is usually typified by severe sun
damaged except in cases of Bowen's disease that are caused by human papilloma virus infection and not
by UV. Pigmented variants exists for Bowen's disease and for actinic keratoses.""", "Actinic keratosis"],
    "BKL": ["""Benign keratosis. "Benign keratosis" is a generic class that includes seborrheic keratoses ("senile wart"), solar lentigo - which
can be regarded a flat variant of seborrheic keratosis - and lichen-planus like keratoses (LPLK), which
corresponds to a seborrheic keratosis or a solar lentigo with inflammation and regression.  The three
subgroups may look different dermatoscopically, but they are similar biologically and often reported under the same generic term histopathologically.
From a dermatoscopic
view, lichen planus-like keratoses are especially challenging because they can show morphologic features
mimicking melanoma. and are often biopsied or excised
for diagnostic reasons. The dermatoscopic appearance of seborrheic keratoses varies according to
anatomic site and type.""", "Benign keratosis"],
    "DF": ["""Dermatofibroma. Dermatofibroma is a benign skin lesion regarded as either a benign proliferation or an inflammatory
reaction to minimal trauma. The most common dermatoscopic presentation is reticular lines at the
periphery with a central white patch denoting fibrosis.""", "Dermatofibroma"],
    "VASC": ["""Vascular lesion. Vascular skin lesions in the dataset range from cherry angiomas to angiokeratomas and pyogenic granulomas. Hemorrhage is also included in this category. Angiomas are dermatoscopically characterized by red or purple color and solid, well circumscribed
structures known as red clods or lacunes.""", "Vascular lesion"],
    "SCC": ["""Squamous cell carcinoma. The second most common form of skin cancer, characterized by abnormal, accelerated growth of squamous cells. 
    Because it can mimic a benign condition, such as a wart, and it may initially be painless, verrucous squamous cell carcinoma can sometimes be mistaken for a less serious issue, leading to delayed diagnosis.
    a""", "Squamous cell carcinoma"],
    "UNK": ["Unknown", None]
}

# Because we are using the 2018 dataset, the two labels are not there SCC and UNK, so let's drop them
with open('questions_diag_mapping.json') as f:
    questions_diag_mapping = json.load(f)
    questions_diag_mapping.pop("SCC")
    questions_diag_mapping.pop("UNK")
    print(questions_diag_mapping.keys())
