import os
import random
import shutil
from collections import defaultdict

archive_images = "archive/images"
archive_labels = "archive/labels"
train_images = "train/images"
train_labels = "train/labels"
val_images = "val/images"
val_labels = "val/labels"

os.makedirs(train_images, exist_ok=True)
os.makedirs(train_labels, exist_ok=True)
os.makedirs(val_images, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

image_classes = {}
class_counts = defaultdict(int)

for img_file in os.listdir(archive_images):
    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(archive_labels, label_file)
    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as f:
        classes = [int(line.split()[0]) for line in f if line.strip()]
        image_classes[img_file] = set(classes)
        for c in set(classes):  # contamos cada clase una vez por imagen
            class_counts[c] += 1

train_class_balance = defaultdict(int)
val_class_balance = defaultdict(int)

train_ratio = 0.8
val_ratio = 1 - train_ratio

image_files = list(image_classes.keys())
random.shuffle(image_files)

train_files, val_files = [], []

for img in image_files:
    classes = image_classes[img]
    # Ver cuántas imágenes de esas clases ya están en train
    train_coverage = sum(train_class_balance[c] / class_counts[c] for c in classes) / len(classes)
    val_coverage = sum(val_class_balance[c] / class_counts[c] for c in classes) / len(classes)

    # Decisión basada en cobertura promedio por clase
    if train_coverage < train_ratio or len(val_files) >= val_ratio * len(image_files):
        train_files.append(img)
        for c in classes:
            train_class_balance[c] += 1
    else:
        val_files.append(img)
        for c in classes:
            val_class_balance[c] += 1

def copy_files(files, img_dest, lbl_dest):
    for img_file in files:
        src_img = os.path.join(archive_images, img_file)
        dst_img = os.path.join(img_dest, img_file)
        shutil.copy2(src_img, dst_img)

        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_lbl = os.path.join(archive_labels, label_file)
        dst_lbl = os.path.join(lbl_dest, label_file)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)

copy_files(train_files, train_images, train_labels)
copy_files(val_files, val_images, val_labels)

print(f"Total imágenes: {len(image_files)}")
print(f"Train: {len(train_files)}")
print(f"Val: {len(val_files)}")
print(f"Train ratio real: {len(train_files) / len(image_files):.2f}")
print(f"Val ratio real: {len(val_files) / len(image_files):.2f}")
print("Split estratificado multilabel corregido completado.")
