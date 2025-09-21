import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Load metadata
# -----------------------------
meta = pd.read_csv("dataset/HAM10000_metadata.csv")

# Encode labels from dx column
labels = meta["dx"].values
unique_labels = sorted(set(labels))   # ['akiec','bcc','bkl','df','mel','nv','vasc']
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
meta["label"] = meta["dx"].map(label_to_index)

# Function to build full image path
def get_image_path(image_id):
    if os.path.exists(f"dataset/HAM10000_images_part_1/{image_id}.jpg"):
        return f"dataset/HAM10000_images_part_1/{image_id}.jpg"
    else:
        return f"dataset/HAM10000_images_part_2/{image_id}.jpg"

meta["path"] = meta["image_id"].map(get_image_path)

print("Classes:", unique_labels)
print("Total images:", len(meta))

# -----------------------------
# Step 2: Train/Validation split
# -----------------------------
train_df = meta.sample(frac=0.8, random_state=42)
val_df = meta.drop(train_df.index)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# -----------------------------
# Step 3: ImageDataGenerator with flow_from_dataframe
# -----------------------------
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_ds = train_datagen.flow_from_dataframe(
    train_df,
    x_col="path",
    y_col="label",
    target_size=IMAGE_SIZE,
    class_mode="raw",   # because labels are integers already
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = val_datagen.flow_from_dataframe(
    val_df,
    x_col="path",
    y_col="label",
    target_size=IMAGE_SIZE,
    class_mode="raw",
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------------
# Step 4: Build EfficientNetV2-S model
# -----------------------------
base_model = EfficientNetV2S(
    input_shape=IMAGE_SIZE + (3,),
    weights="imagenet",
    include_top=False
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(len(unique_labels), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# -----------------------------
# Step 5: Train
# -----------------------------
EPOCHS = 4

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# -----------------------------
# Step 6: Save Model
# -----------------------------
model.save("efficientnetv2s_ham10000.keras")
print("✅ Model saved as efficientnetv2s_ham10000.keras")

# -----------------------------
# Step 7: Save Training Plots
# -----------------------------
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.legend()
plt.savefig("accuracy.png")
plt.close()

plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.legend()
plt.savefig("loss.png")
plt.close()

print("✅ Training plots saved as accuracy.png and loss.png")
