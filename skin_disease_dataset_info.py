import os
import shutil
import zipfile
import requests
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset sources
data_sources = {
    "HAM10000": "https://dataverse.harvard.edu/api/access/datafile/3173980",
    "Dermnet": "https://www.kaggle.com/datasets/shubhamgoel27/dermnet/download",
    "ISIC": "https://www.isic-archive.com/",
    "SD-198": "https://data.mendeley.com/datasets/zr7vgbcyr2/1/download",
    "Skin-Diseases": "https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset/download"
}

# Create dataset directory
os.makedirs("dataset", exist_ok=True)

def download_and_extract(url, extract_to):
    filename = url.split("/")[-1]
    filepath = os.path.join("dataset", filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            shutil.copyfileobj(response.raw, f)
        print(f"Downloaded {filename}")
    
    # Extract the dataset
    if zipfile.is_zipfile(filepath):
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted {filename}")
    
# Download and extract datasets
for name, url in data_sources.items():
    download_and_extract(url, "dataset")

# Organize dataset into train and validation sets
def organize_dataset():
    base_dir = "dataset/IMG_CLASSES"
    train_dir = "dataset/train"
    val_dir = "dataset/validation"
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if os.path.isdir(category_path):
            os.makedirs(os.path.join(train_dir, category), exist_ok=True)
            os.makedirs(os.path.join(val_dir, category), exist_ok=True)
            
            images = os.listdir(category_path)
            train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
            
            for img in train_images:
                shutil.copy(os.path.join(category_path, img), os.path.join(train_dir, category, img))
            for img in val_images:
                shutil.copy(os.path.join(category_path, img), os.path.join(val_dir, category, img))

organize_dataset()

# Prepare data
image_size = (224, 224)
batch_size = 32

data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
)

train_data = data_gen.flow_from_directory(
    "dataset/train",
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = data_gen.flow_from_directory(
    "dataset/validation",
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Define model
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(train_data.class_indices), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(train_data, validation_data=val_data, epochs=10)

# Save model
model.save("skin_disease_model.h5")

print("Model training complete and saved as skin_disease_model.h5")

# Load and test model
def test_model():
    model = keras.models.load_model("skin_disease_model.h5")
    loss, accuracy = model.evaluate(val_data)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

test_model()
