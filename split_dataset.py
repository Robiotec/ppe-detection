import os
import random
import shutil

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

image_files = [
    f
    for f in os.listdir(archive_images)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

random.shuffle(image_files)
split_idx = int(0.8 * len(image_files))
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]


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
        else:
            print(f"Warning: Label not found for {img_file}")


copy_files(train_files, train_images, train_labels)
copy_files(val_files, val_images, val_labels)

print(f"Train images: {len(train_files)}")
print(f"Val images: {len(val_files)}")
print("Split complete.")
