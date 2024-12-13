import os
import shutil

# Paths to your original training and testing datasets
train_dataset_dir = 'dataSet/english/trainingData'  # E.g., dataSet/english/trainingData/
test_dataset_dir = 'dataSet/english/testingData'    # E.g., dataSet/english/testingData/

# Target folder to store the translated datasets
translated_dataset_dir = 'dataSet'

# Dictionary to map English labels (A to Z) to their corresponding translations
label_translation = {
    'A': {'Urdu': 'ا', 'German': 'A', 'Chinese': '阿'},
    'B': {'Urdu': 'ب', 'German': 'B', 'Chinese': '贝'},
    'C': {'Urdu': 'س', 'German': 'C', 'Chinese': '西'},
    'D': {'Urdu': 'د', 'German': 'D', 'Chinese': '迪'},
    'E': {'Urdu': 'ای', 'German': 'E', 'Chinese': '伊'},
    'F': {'Urdu': 'ف', 'German': 'F', 'Chinese': '福'},
    'G': {'Urdu': 'گ', 'German': 'G', 'Chinese': '吉'},
    'H': {'Urdu': 'ہ', 'German': 'H', 'Chinese': '哈'},
    'I': {'Urdu': 'آئی', 'German': 'I', 'Chinese': '艾'},
    'J': {'Urdu': 'جے', 'German': 'J', 'Chinese': '杰'},
    'K': {'Urdu': 'کے', 'German': 'K', 'Chinese': '凯'},
    'L': {'Urdu': 'ایل', 'German': 'L', 'Chinese': '艾尔'},
    'M': {'Urdu': 'ایم', 'German': 'M', 'Chinese': '艾姆'},
    'N': {'Urdu': 'این', 'German': 'N', 'Chinese': '恩'},
    'O': {'Urdu': 'او', 'German': 'O', 'Chinese': '欧'},
    'P': {'Urdu': 'پی', 'German': 'P', 'Chinese': '佩'},
    'Q': {'Urdu': 'کیو', 'German': 'Q', 'Chinese': '奇'},
    'R': {'Urdu': 'آر', 'German': 'R', 'Chinese': '艾尔'},
    'S': {'Urdu': 'ایس', 'German': 'S', 'Chinese': '艾斯'},
    'T': {'Urdu': 'ٹی', 'German': 'T', 'Chinese': '提'},
    'U': {'Urdu': 'یو', 'German': 'U', 'Chinese': '尤'},
    'V': {'Urdu': 'وی', 'German': 'V', 'Chinese': '维'},
    'W': {'Urdu': 'ڈبلیو', 'German': 'W', 'Chinese': '维'},
    'X': {'Urdu': 'ایکس', 'German': 'X', 'Chinese': '艾克斯'},
    'Y': {'Urdu': 'وائی', 'German': 'Y', 'Chinese': '艾'},
    'Z': {'Urdu': 'زی', 'German': 'Z', 'Chinese': '兹'},
    # Add more mappings for all necessary labels or signs
}

# Function to translate dataset for a specific language
def translate_dataset(source_dir, target_dir, lang):
    # Create main language folder
    lang_folder = os.path.join(target_dir, lang)
    os.makedirs(lang_folder, exist_ok=True)

    # Create subfolders for train and test datasets
    for subset in ['trainingData', 'testingData']:
        subset_folder = os.path.join(lang_folder, subset)
        os.makedirs(subset_folder, exist_ok=True)

        # Determine the source directory based on the subset
        subset_source_dir = train_dataset_dir if subset == 'trainingData' else test_dataset_dir
        
        # Iterate over each label in the original dataset (A to Z)
        for label in os.listdir(subset_source_dir):
            label_path = os.path.join(subset_source_dir, label)

            # Ensure it's a directory and that it's in the translation dictionary
            if os.path.isdir(label_path) and label in label_translation:
                # Get the translated label for the current language
                translated_label = label_translation[label][lang]

                # Create a new directory for the translated label
                translated_label_path = os.path.join(subset_folder, translated_label)
                os.makedirs(translated_label_path, exist_ok=True)

                # Copy each image from the original label directory to the new translated directory
                for image_file in os.listdir(label_path):
                    source_image_path = os.path.join(label_path, image_file)
                    target_image_path = os.path.join(translated_label_path, image_file)

                    # Copy the image
                    shutil.copy(source_image_path, target_image_path)

# Translate datasets into Urdu first
translate_dataset(train_dataset_dir, translated_dataset_dir, 'Urdu')
translate_dataset(test_dataset_dir, translated_dataset_dir, 'Urdu')

# Then translate into German
translate_dataset(train_dataset_dir, translated_dataset_dir, 'German')
translate_dataset(test_dataset_dir, translated_dataset_dir, 'German')

# Finally, translate into Chinese
translate_dataset(train_dataset_dir, translated_dataset_dir, 'Chinese')
translate_dataset(test_dataset_dir, translated_dataset_dir, 'Chinese')

print("Translation and reorganization completed for Urdu, German, and Chinese datasets!")
