import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
import uuid

df = pd.read_csv("question_answer_data.csv")
# image_dataset_dir = 'ISIC_2019_Training_Input'
# image_file_names = os.listdir(image_dataset_dir)
# image_file_names = [f for f in image_file_names if f.endswith('.jpg')]

diagnosis_definitions = {
    "MEL": """Melanoma. Melanoma is a malignant neoplasm derived from melanocytes that may appear in different variants. If
excised in an early stage it can be cured by simple surgical excision. Melanomas can be invasive or noninvasive (in situ). Melanomas are usually, albeit not always, chaotic, and some melanoma specific criteria depend on
anatomic site.""",
    "NV": """Melanocytic nevus. Melanocytic nevi are benign neoplasms of melanocytes and appear in a myriad of variants. In
contrast to melanoma they are usually symmetric with regard to the distribution of color and structure""",
    "BCC": """Basal cell carcinoma. Basal cell carcinoma is a common variant of epithelial skin cancer that rarely metastasizes but grows
destructively if untreated. It appears in different morphologic variants (flat, nodular, pigmented, cystic).""",
    "AK": """Actinic keratosis. Actinic Keratoses (Solar Keratoses) and Intraepithelial Carcinoma (Bowen’s disease) are common noninvasive, variants of squamous cell carcinoma that can be treated locally without surgery. Some authors
regard them as precursors of squamous cell carcinomas and not as actual carcinomas. There is, however,
agreement that these lesions may progress to invasive squamous cell carcinoma – which is usually not
pigmented. Both neoplasms commonly show surface scaling and commonly are devoid of pigment. Actinic keratoses are more common on the face and Bowen’s disease is more common on other body
sites. Because both types are induced by UV-light the surrounding skin is usually typified by severe sun
damaged except in cases of Bowen’s disease that are caused by human papilloma virus infection and not
by UV. Pigmented variants exists for Bowen’s disease and for actinic keratoses.""",
    "BKL": """Benign keratosis. "Benign keratosis" is a generic class that includes seborrheic keratoses ("senile wart"), solar lentigo - which
can be regarded a flat variant of seborrheic keratosis - and lichen-planus like keratoses (LPLK), which
corresponds to a seborrheic keratosis or a solar lentigo with inflammation and regression.  The three
subgroups may look different dermatoscopically, but they are similar biologically and often reported under the same generic term histopathologically.
From a dermatoscopic
view, lichen planus-like keratoses are especially challenging because they can show morphologic features
mimicking melanoma. and are often biopsied or excised
for diagnostic reasons. The dermatoscopic appearance of seborrheic keratoses varies according to
anatomic site and type.""",
    "DF": """Dermatofibroma. Dermatofibroma is a benign skin lesion regarded as either a benign proliferation or an inflammatory
reaction to minimal trauma. The most common dermatoscopic presentation is reticular lines at the
periphery with a central white patch denoting fibrosis.""",
    "VASC": """Vascular lesion. Vascular skin lesions in the dataset range from cherry angiomas to angiokeratomas and pyogenic granulomas. Hemorrhage is also included in this category. Angiomas are dermatoscopically characterized by red or purple color and solid, well circumscribed
structures known as red clods or lacunes.""",
    "SCC": """Squamous cell carcinoma. The second most common form of skin cancer, characterized by abnormal, accelerated growth of squamous cells. 
    Because it can mimic a benign condition, such as a wart, and it may initially be painless, verrucous squamous cell carcinoma can sometimes be mistaken for a less serious issue, leading to delayed diagnosis.
    a""",
    "UNK": "Unknown"
}
image_dataset_ground_truth = pd.read_csv('ISIC_2019_Training_GroundTruth.csv')
image_diagnosis_mapping = {}
diagnosis_image_mapping = {}

for i in range(len(image_dataset_ground_truth)):
    diagnosis = ""
    for j in range(1, len(image_dataset_ground_truth.columns)):
        if image_dataset_ground_truth.iloc[i][j] == 1:
            diagnosis = image_dataset_ground_truth.columns[j]
            break
    image_diagnosis_mapping[image_dataset_ground_truth.iloc[i]['image']] = diagnosis
    if diagnosis not in diagnosis_image_mapping:
        diagnosis_image_mapping[diagnosis] = []
    diagnosis_image_mapping[diagnosis].append(image_dataset_ground_truth.iloc[i]['image'])


print("First 5 rows of diagnosis_image_mapping:")
print(list(image_diagnosis_mapping.items())[:5])
print("\n\n\n\n")
print("question answer dataset:")
print(df.head())
try:

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    diagnosis_embeddings = {label: model.encode(description) for label, description in diagnosis_definitions.items()}

    similarity_threshold = 0.5

    print("Starting to write to file")

    output_file = open('output.txt', 'w')

    for i in range(len(df)):
        query_prompt = df.iloc[i]['prompt']
        query_response = df.iloc[i]['response']
        query = query_prompt + " " + query_response
        query_embedding = model.encode(query)
        max_similarity = -1
        predicted_diagnosis = "UNK"
        for diagnosis, definition_embedding in diagnosis_embeddings.items():
            similarity = util.cos_sim(query_embedding, definition_embedding)
            if similarity > similarity_threshold and similarity > max_similarity:
                max_similarity = similarity
                predicted_diagnosis = diagnosis
        vqa_query_id = str(uuid.uuid4())
        json_obj = {
            "id": vqa_query_id,
            "question": query_prompt,
            "answer": query_response,
            "diagnosis": predicted_diagnosis
        }
        if predicted_diagnosis != "UNK": 
            # for image in diagnosis_image_mapping[predicted_diagnosis]:
            #     json_obj_temp = json_obj.copy()
            #     json_obj_temp['image'] = image
            #     print("\n\n\n\n***********************************")
            #     print("Writing to file: ", json_obj_temp)
            #     print("***********************************\n\n\n\n")
            #     output_file.write(str(json_obj_temp) + "\n")
            json_obj['image'] = diagnosis_image_mapping[predicted_diagnosis]
            print("\n\n\n\n***********************************")
            print("Writing to file: ", json_obj)
            print("***********************************\n\n\n\n")
            output_file.write(str(json_obj) + "\n")

    output_file.close()

except Exception as e:
    output_file.close()
    print("Error occurred: ", e)
    raise e


