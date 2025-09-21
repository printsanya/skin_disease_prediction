# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# import pandas as pd
# import os

# # Reload same label mapping as training
# meta = pd.read_csv("dataset/HAM10000_metadata.csv")
# unique_labels = sorted(set(meta["dx"].values))
# label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
# index_to_label = {v: k for k, v in label_to_index.items()}

# # Add full names
# dx_fullnames = {
#     "akiec": "Actinic keratoses / intraepithelial carcinoma",
#     "bcc": "Basal cell carcinoma",
#     "bkl": "Benign keratosis-like lesions",
#     "df": "Dermatofibroma",
#     "mel": "Melanoma",
#     "nv": "Melanocytic nevi (common moles)",
#     "vasc": "Vascular lesions (angiomas, hemangiomas)"
# }

# # Load trained model
# model = load_model("efficientnetv2s_ham10000.keras")
# IMAGE_SIZE = (224, 224)

# def predict_image(img_path):
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, IMAGE_SIZE)
#     img = np.expand_dims(img, axis=0) / 255.0
#     pred = model.predict(img)
#     pred_class = np.argmax(pred, axis=1)[0]
#     short_code = index_to_label[pred_class]
#     return dx_fullnames[short_code]

# # Example prediction
# test_img = "dataset/HAM10000_images_part_1/ISIC_0024306.jpg"
# print("Prediction:", predict_image(test_img))
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import os

# -----------------------------
# Load metadata
# -----------------------------
meta = pd.read_csv("dataset/HAM10000_metadata.csv")
unique_labels = sorted(set(meta["dx"].values))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
index_to_label = {v: k for k, v in label_to_index.items()}

# Mapping full names
dx_fullnames = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi (common moles)",
    "vasc": "Vascular lesions (angiomas, hemangiomas)"
}

# -----------------------------
# Load trained model
# -----------------------------
model = load_model("efficientnetv2s_ham10000.keras")
IMAGE_SIZE = (224, 224)

# -----------------------------
# Prediction function with ground truth check
# -----------------------------
def predict_image(image_id):
    # Build correct path
    if os.path.exists(f"dataset/HAM10000_images_part_1/{image_id}.jpg"):
        img_path = f"dataset/HAM10000_images_part_1/{image_id}.jpg"
    else:
        img_path = f"dataset/HAM10000_images_part_2/{image_id}.jpg"

    # Load image
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMAGE_SIZE)
    img = np.expand_dims(img, axis=0) / 255.0

    # Predict
    pred = model.predict(img)
    pred_class = np.argmax(pred, axis=1)[0]
    short_code = index_to_label[pred_class]

    # Get true label from metadata
    true_code = meta.loc[meta["image_id"] == image_id, "dx"].values[0]

    return {
        "image_id": image_id,
        "predicted": dx_fullnames[short_code],
        "actual": dx_fullnames[true_code],
        "correct": (short_code == true_code)
    }

# -----------------------------
# Example
# -----------------------------
result = predict_image("ISIC_0024306")  # Just pass image_id (without .jpg)

print("Image ID:", result["image_id"])
print("Predicted:", result["predicted"])
print("Actual:", result["actual"])
print("Correct Prediction:", result["correct"])
