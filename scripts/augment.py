import os
from imutils import paths
from pathlib import Path
import imgaug as ia
from math import floor
from math import isnan
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters
import imageio
import pandas
import numpy
import re
import xml.etree.ElementTree as ET
import shutil

AUG_NB_AUGMENTATION_PER_IMAGE = int(
    os.getenv('AUG_NB_AUGMENTATION_PER_IMAGE', 10))

DATASET_DIR = "/usr/dataset"
DATASET_IMAGES_DIR = "{}/images".format(DATASET_DIR)
DATASET_ANNOTS_DIR = "{}/annotations".format(DATASET_DIR)
DATASET_AUGM_DIR = "/usr/dataset-augmented"
DATASET_AUGM_IMAGES_DIR = "{}/images".format(DATASET_AUGM_DIR)
DATASET_AUGM_ANNOTS_DIR = "{}/annotations".format(DATASET_AUGM_DIR)


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
    annotation_path = "{}/{}.xml".format(DATASET_ANNOTS_DIR, stem)
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


def bbs_obj_to_df(bbs_object):
    bbs_array = bbs_object.to_xyxy_array()
    df_bbs = pandas.DataFrame(
        bbs_array, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    return df_bbs


def augment_image(df, images_path, aug_dest_dir, image_suffix, augmentor):
    aug_bbs_xy = pandas.DataFrame(
        columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
    grouped = df.groupby('filename')

    for filename in df['filename'].unique():
        group_df = grouped.get_group(filename)
        group_df = group_df.reset_index()
        group_df = group_df.drop(['index'], axis=1)
        image = imageio.imread("{}/{}".format(images_path, filename))
        bb_array = group_df.drop(
            ['filename', 'width', 'height', 'class'], axis=1).values
        bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
        image_aug, bbs_aug = augmentor(image=image, bounding_boxes=bbs)
        bbs_aug = bbs_aug.remove_out_of_image()
        bbs_aug = bbs_aug.clip_out_of_image()

        if re.findall('Image...', str(bbs_aug)) == ['Image([]']:
            pass
        else:
            filename_data = Path(filename)
            filename_final = "{}{}{}".format(
                filename_data.stem,
                image_suffix,
                filename_data.suffix
            )
            imageio.imwrite(
                "{}/{}".format(aug_dest_dir, filename_final),
                image_aug
            )
            info_df = group_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)
            for index, _ in info_df.iterrows():
                info_df.at[index, 'width'] = image_aug.shape[1]
                info_df.at[index, 'height'] = image_aug.shape[0]
            info_df['filename'] = filename_final
            bbs_df = bbs_obj_to_df(bbs_aug)
            aug_df = pandas.concat([info_df, bbs_df], axis=1)
            aug_bbs_xy = pandas.concat([aug_bbs_xy, aug_df])

    aug_bbs_xy = aug_bbs_xy.reset_index()
    aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)
    return aug_bbs_xy


def get_xml_from_dataframe(origin_annotation_path, labels_df):
    tree = ET.parse(origin_annotation_path)
    annotation_et = tree.getroot()
    objects_et = annotation_et.findall("object")
    for object_et in objects_et:
        annotation_et.remove(object_et)
    filename_et = annotation_et.find("filename")
    filename_et.text = labels_df.iloc[0]["filename"]
    width_et = annotation_et.find("size").find("width")
    width_et.text = str(labels_df.iloc[0]["width"])
    height_et = annotation_et.find("size").find("height")
    height_et.text = str(labels_df.iloc[0]["height"])
    for label_df_index in labels_df.index:
        label_df = labels_df.iloc[label_df_index]
        if (isnan(label_df["xmin"]) == False):
            object_et = ET.SubElement(annotation_et, 'object')
            ET.SubElement(object_et, 'name').text = label_df["class"]
            ET.SubElement(object_et, 'pose').text = "Unspecified"
            ET.SubElement(object_et, 'truncated').text = "0"
            ET.SubElement(object_et, 'difficult').text = "0"
            bndbox_et = ET.SubElement(object_et, 'bndbox')
            ET.SubElement(bndbox_et, 'xmin').text = str(
                floor(label_df["xmin"]))
            ET.SubElement(bndbox_et, 'ymin').text = str(
                floor(label_df["ymin"]))
            ET.SubElement(bndbox_et, 'xmax').text = str(
                floor(label_df["xmax"]))
            ET.SubElement(bndbox_et, 'ymax').text = str(
                floor(label_df["ymax"]))
    return tree


numpy.random.bit_generator = numpy.random._bit_generator
ia.seed(1)

if (create_directory(DATASET_AUGM_IMAGES_DIR) == False):
    exit(1)
if (create_directory(DATASET_AUGM_ANNOTS_DIR) == False):
    exit(1)

images_path = list(paths.list_images(DATASET_IMAGES_DIR))
for image_path in images_path:
    image_path_data = Path(image_path)
    stem = image_path_data.stem
    basename = image_path_data.name
    image_annotation_path = get_image_annotation_path(stem)
    image_annotation_path_data = Path(image_annotation_path)

    print("Processing {}...".format(stem))
    if (image_annotation_path == False):
        continue
    labels_df = decode_image_annotation(image_annotation_path)

    shutil.copy(image_path, "{}/{}".format(DATASET_AUGM_IMAGES_DIR, basename))
    shutil.copy(image_annotation_path,
                "{}/{}".format(DATASET_AUGM_ANNOTS_DIR, image_annotation_path_data.name))
    for i in range(0, AUG_NB_AUGMENTATION_PER_IMAGE):
        # This setup of augmentation parameters will pick 1 to 4
        # of the given augmenters and apply them in random order.
        sometimes = lambda aug: augmenters.Sometimes(0.5, aug)
        aug_config = augmenters.SomeOf(
            (1, 4),
            [
                augmenters.Affine(scale=(0.5, 1.5)),
                augmenters.Affine(rotate=(-60, 60)),
                augmenters.Affine(translate_percent={
                                  "x": (-0.3, 0.3), "y": (-0.3, 0.3)}),
                augmenters.Fliplr(1),
                sometimes(
                    augmenters.Superpixels(
                        p_replace=(0, 1.0),
                        n_segments=(20, 200)
                    )
                ),
                augmenters.OneOf([
                    augmenters.GaussianBlur((0, 3.0)),
                    augmenters.AverageBlur(k=(2, 7)),
                    augmenters.MedianBlur(k=(3, 11)),
                ]),
                augmenters.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                augmenters.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                sometimes(augmenters.OneOf([
                    augmenters.EdgeDetect(alpha=(0, 0.7)),
                    augmenters.DirectedEdgeDetect(
                        alpha=(0, 0.7), direction=(0.0, 1.0)
                    ),
                ])),
                augmenters.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                ),
                augmenters.OneOf([
                    augmenters.Dropout((0.01, 0.1), per_channel=0.5),
                    augmenters.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ]),
                augmenters.Invert(0.05, per_channel=True),
                augmenters.Add((-10, 10), per_channel=0.5),
                augmenters.Multiply((0.5, 1.5), per_channel=0.5),
                augmenters.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                augmenters.Grayscale(alpha=(0.0, 1.0)),
                sometimes(
                    augmenters.ElasticTransformation(
                        alpha=(0.5, 3.5), sigma=0.25)
                ),
                sometimes(augmenters.PiecewiseAffine(scale=(0.01, 0.05)))
            ],
            random_order=True
        )
        aug_file_suffix = '_aug_{}'.format(i)
        augmented_image_df = augment_image(
            labels_df,
            DATASET_IMAGES_DIR,
            DATASET_AUGM_IMAGES_DIR,
            aug_file_suffix,
            aug_config
        )
        image_annotation_dest_path = "{}/{}{}{}".format(
            DATASET_AUGM_ANNOTS_DIR,
            stem, aug_file_suffix,
            image_annotation_path_data.suffix
        )
        if (len(augmented_image_df.index)):
            xml_annotations_et = get_xml_from_dataframe(
                image_annotation_path,
                augmented_image_df
            )
            xml_annotations_et.write(image_annotation_dest_path)

    print("OK.")
