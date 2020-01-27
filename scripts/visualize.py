import os
import cv2
from imutils import paths
from pathlib import Path
import pandas
import xml.etree.ElementTree as ET
import shutil

AUG_NB_AUGMENTATION_PER_IMAGE = int(
    os.getenv('AUG_NB_AUGMENTATION_PER_IMAGE', 10))

DATASET_AUGM_DIR = "/usr/dataset-augmented"
DATASET_AUGM_IMAGES_DIR = "{}/images".format(DATASET_AUGM_DIR)
DATASET_AUGM_ANNOTS_DIR = "{}/annotations".format(DATASET_AUGM_DIR)
DATASET_VISU_DIR = "/usr/dataset-visualization"
DATASET_VISU_IMAGES_DIR = "{}/images".format(DATASET_VISU_DIR)
DATASET_VISU_ANNOTS_DIR = "{}/annotations".format(DATASET_VISU_DIR)


def create_directory(path, remove_dir=False):
    if (remove_dir):
        if (shutil.rmtree(path) == False):
            print("Failed to remove directory {}".format(path))
            return False
    path = str(path)
    if os.path.exists(path) is False:
        try:
            os.makedirs(path)
        except OSError:
            print("Failed to create directory {}".format(path))
            return False
        else:
            print("Successfully created the directory {}".format(path))
    return True


def get_image_annotation_path(stem):
    annotation_path = "{}/{}.xml".format(DATASET_AUGM_ANNOTS_DIR, stem)
    if (os.path.isfile(annotation_path) == False):
        print("Annotation not found for " + stem)
        return False
    return annotation_path


def decode_image_annotation(path_annotation):
    xml_list = []
    tree = ET.parse(path_annotation)
    root = tree.getroot()
    for member in root.findall('object'):
        value = (
            root.find('filename').text,
            int(root.find('size')[0].text),
            int(root.find('size')[1].text),
            member[0].text,
            int(member[4][0].text),
            int(member[4][1].text),
            int(member[4][2].text),
            int(member[4][3].text)
        )
        xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pandas.DataFrame(xml_list, columns=column_name)
    return xml_df

if (create_directory(DATASET_VISU_IMAGES_DIR) == False):
    exit(1)
if (create_directory(DATASET_VISU_ANNOTS_DIR) == False):
    exit(1)

images_path = list(paths.list_images(DATASET_AUGM_IMAGES_DIR))
for image_path in images_path:
    image_path_data = Path(image_path)
    stem = image_path_data.stem
    basename = image_path_data.name
    image_annotation_path = get_image_annotation_path(stem)
    image_annotation_path_data = Path(image_annotation_path)

    print("Rendering {}...".format(stem))
    if (image_annotation_path == False):
        continue
    labels_df = decode_image_annotation(image_annotation_path)

    image = cv2.imread(image_path)
    for object_index in labels_df.index:
        object_df = labels_df.iloc[object_index]
        image = cv2.rectangle(
            image,
            (object_df["xmin"], object_df["ymin"]),
            (object_df["xmax"], object_df["ymax"]),
            (0, 0, 255),
            2
        )
    image_dest = "{}/{}".format(DATASET_VISU_IMAGES_DIR, basename)
    cv2.imwrite(image_dest, image)

    print("OK.")
