import os
import string

# Base directories for dataset
BASE_DIR = "dataSet"
TRAIN_DIR = os.path.join(BASE_DIR, "trainingData")
TEST_DIR = os.path.join(BASE_DIR, "testingData")

# Ensure base directories exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# List of folders to create (blank "0" and A-Z)
categories = ["0"] + list(string.ascii_uppercase)

# Create folders for training and testing datasets
for category in categories:
    train_category_path = os.path.join(TRAIN_DIR, category)
    test_category_path = os.path.join(TEST_DIR, category)

    os.makedirs(train_category_path, exist_ok=True)
    os.makedirs(test_category_path, exist_ok=True)

print("Folders created successfully!")
