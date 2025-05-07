import os
import shutil
import xml.etree.ElementTree as ET

ANNOTATION_DIR = "/mnt/lustre/users/inf/kajm20/ILSVRC/Annotations/CLS-LOC/val"
VAL_IMAGE_DIR = "/mnt/lustre/users/inf/kajm20/ILSVRC/Data/CLS-LOC/val"

# Step 1: Create a temporary backup of original flat val dir
BACKUP_DIR = VAL_IMAGE_DIR + "_flat_backup"
os.makedirs(BACKUP_DIR, exist_ok=True)

# Move all JPEGs to backup dir to prevent overwriting during restructure
for fname in os.listdir(VAL_IMAGE_DIR):
    if fname.endswith(".JPEG"):
        shutil.move(os.path.join(VAL_IMAGE_DIR, fname), os.path.join(BACKUP_DIR, fname))

# Step 2: Parse annotations and move files into correct class subdirs
for fname in os.listdir(ANNOTATION_DIR):
    if not fname.endswith(".xml"):
        continue

    xml_path = os.path.join(ANNOTATION_DIR, fname)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_id = root.find("filename").text + ".JPEG"
    wnid = root.find("object").find("name").text

    src_img_path = os.path.join(BACKUP_DIR, image_id)
    dst_class_dir = os.path.join(VAL_IMAGE_DIR, wnid)
    os.makedirs(dst_class_dir, exist_ok=True)

    dst_img_path = os.path.join(dst_class_dir, image_id)
    if os.path.exists(src_img_path):
        shutil.move(src_img_path, dst_img_path)
