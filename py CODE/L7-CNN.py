# %%
# %pip install tensorflow
# %pip install "numpy<2.0"
# %pip install --upgrade pandas tensorflow keras

# %%
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# %%

from sklearn.metrics import confusion_matrix, classification_report
import pydicom
from PIL import Image
from tqdm import tqdm
import numpy as np
import os

# %%
# (train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

# # Normalize pixel values to be between 0 and 1
# train_images, test_images = train_images / 255.0, test_images / 255.0

# %%
# 1. Setup Paths and Parameters
DATA = r'E:/edu/4-2/ETE4215/LAB/data/3Xray'
LIMIT = 54
target_files = []
class_counts = {}

print("Current working directory:", os.getcwd())
print("Does DATA folder exist?:", os.path.exists(DATA))
if os.path.exists(DATA):
    for class_folder in os.listdir(DATA):
        class_path = os.path.join(DATA, class_folder)
        
        if os.path.isdir(class_path):
            # Filtering for .png files as per your latest snippet
            all_files = [f for f in os.listdir(class_path) if f.lower().endswith(".png")]
            
            # Apply the limit (e.g., first 54)
            limited_files = all_files[:LIMIT]
            
            # Store full paths for processing later
            for filename in limited_files:
                target_files.append(os.path.join(class_path, filename))
            
            # Store count for the plot
            class_counts[class_folder] = len(limited_files)

print(f"📦 Found {len(target_files)} total files (Limited to {LIMIT} per class).")

# %%


# %%
import shutil

# 1. Define a new directory for the limited dataset
DATA_LIMITED = r'E:/edu/4-2/ETE4215/LAB/data/3Xray_Limited'
os.makedirs(DATA_LIMITED, exist_ok=True)

print(f"📂 Copying {len(target_files)} files to {DATA_LIMITED}...")
class_names={}
for filepath in target_files:
    # Get the class name (subfolder)
    class_name = os.path.basename(os.path.dirname(filepath))
    #add class names to the class_names dictionary
    if class_name not in class_names:
        class_names[class_name] = len(class_names)
    # Create the class subfolder in the new directory
    target_dir = os.path.join(DATA_LIMITED, class_name)
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy the file
    shutil.copy(filepath, os.path.join(target_dir, os.path.basename(filepath)))

print("✅ Done! You now have a clean folder with exactly 54 images per class.")



# %%
# for filepath in tqdm(dicom_files, desc="Converting limited DICOMs"):
#     try:
#         # 1. Read and Normalize
#         ds = pydicom.dcmread(filepath)
#         img = normalize_to_uint8(ds.pixel_array)
        
#         # 2. Maintain folder structure (Get 'ELBOW', 'FOOT', etc.)
#         class_folder = os.path.basename(os.path.dirname(filepath))
#         target_folder = os.path.join(DATA, class_folder)
#         os.makedirs(target_folder, exist_ok=True)
        
#         # 3. Save as PNG
#         filename = os.path.basename(filepath).replace(".dcm", ".png")
#         Image.fromarray(img).save(os.path.join(target_folder, filename))
        
#     except Exception as e:
#         print(f"Error with {filepath}: {e}")

# %%


# %%
# 3. Visualization: Plot the limited distribution
plt.figure(figsize=(8, 6))

# Create the bar chart
bars = plt.bar(class_counts.keys(), class_counts.values(), color='skyblue', edgecolor='navy')

# Add values on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title(f'Distribution of Classes (Limited to {LIMIT} per class)')
plt.xlabel('X-Ray Type')
plt.ylabel('Count')

# Adjust y-axis to make room for labels
if class_counts:
    plt.ylim(0, max(class_counts.values()) * 1.2)

plt.tight_layout()
plt.show()

# %%


# %%
BATCH_SIZE = 16 # Powers of 2 (16 or 32) are better for memory
IMG_SIZE = (254, 254)
# 1. Load Training Data (from the LIMITED folder)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_LIMITED,
    validation_split=0.2,
    subset="training",
    seed=16,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# 2. Load Validation Data (from the LIMITED folder)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_LIMITED,
    validation_split=0.2,
    subset="validation",
    seed=16,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# %%
# class_names = train_ds.class_names

plt.figure(figsize=(10, 12))
plt.title("Sample Images from Training Dataset")

# 'take(1)' retrieves the first batch of 32 images
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        
        # Images are already normalized/rescaled if you followed the previous step
        # If they look weird, you might need to use images[i].numpy().astype("uint8")
        plt.imshow(images[i].numpy())
        
        plt.title(class_name[labels[i]])
        plt.axis("off")

plt.tight_layout()
plt.savefig(f"{DATA_LIMITED}/sample_images.pdf")
plt.show()

# %%
# 3. Apply Normalization
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# 4. Print Verification
for images, labels in train_ds.take(1):
    print(f"Single image shape: {images[0].shape}")
    print(f"Batch size: {len(images)}")
    break

# %%
model = models.Sequential()
model.add(layers.Conv2D(256, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))


# %%
model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dropout(0.5))  # Add dropout for regularization
model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dropout(0.5))  # Add dropout for regularization
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))  # Add dropout for regularization
model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dropout(0.5))  # Add dropout for regularization
model.add(layers.Dense(3, activation='softmax'))

# %%
model.summary()

# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=10)

# %%
plt.title("Training and Validation Loss of CNN Model")
plt.plot(history.history['accuracy'][:8], label='accuracy')
plt.plot(history.history['val_accuracy'][:8], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.2, 1])
plt.legend(loc='lower right')
plt.savefig(f"{DATA_LIMITED}/cnn_accuracy.pdf")
plt.show()
test_loss, test_acc = model.evaluate(val_ds, verbose=2)

# %%
print(test_acc)

# %%
plt.title("Training and Validation Loss of CNN Model")
plt.plot(history.history['loss'][:8], label='loss')
plt.plot(history.history['val_loss'][:8], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.2, 1.5])
plt.legend(loc='lower right')
plt.savefig(f"{DATA_LIMITED}/cnn_loss.pdf")
plt.show()
# test_loss, test_acc = model.evaluate(val_ds, verbose=2)

# %%
class_name

# %%
y_true = tf.concat([labels for _, labels in val_ds], axis=0).numpy()
y_pred = tf.argmax(model.predict(val_ds), axis=1).numpy()

report = classification_report(y_true, y_pred, target_names=class_names)
cm = confusion_matrix(y_true, y_pred)
print(report)
cm = confusion_matrix(y_true, y_pred)

# %%
plt.title("ROC Curve of CNN Model")
plt.plot(history.history['accuracy'][:8], label='accuracy')
plt.plot(history.history['val_accuracy'][:8], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.2, 1])
plt.legend(loc='lower right')
plt.savefig(f"{DATA_LIMITED}/cnn_roc_curve.pdf")
plt.show()


# %%
from sklearn.metrics import ConfusionMatrixDisplay

# Create the display object
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

# Plot it
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix of CNN Model")
plt.savefig(f"{DATA_LIMITED}/cnn_confusion_matrix.pdf")
plt.show()


