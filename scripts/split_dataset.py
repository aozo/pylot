"""Script for splitting datasets into traffic sign classes and traffic light classes.

Setup:

Install fiftyone: https://docs.voxel51.com/getting_started/install.html

Example usage:

    python split_dataset.py -s -d <image_dir>/ -l <labels>.json
"""

import argparse
import fiftyone as fo
import json
import os

SIGN_CLASSES = [
        "traffic_sign_30",
        "traffic_sign_60",
        "traffic_sign_90"]

LIGHT_CLASSES = [
        "traffic_light_green",
        "traffic_light_orange",
        "traffic_light_red"]

OBSTACLE_CLASSES = [
        "bike",
        "motobike",
        "person",
        "traffic_sign_30",
        "traffic_sign_60",
        "traffic_sign_90",
        "vehicle"]

SUPER_CATEGORY = {
        "RSVD": "RSVD",
        "bike": "vehicle",
        "motobike": "vehicle",
        "person": "pedestrian",
        "traffic_sign_30": "sign",
        "traffic_sign_60": "sign",
        "traffic_sign_90": "sign",
        "traffic_light_green": "light",
        "traffic_light_orange": "light",
        "traffic_light_red": "light",
        "vehicle": "vehicle"
        }

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Splits data set into multiple data sets according to categories.")
    parser.add_argument("--split", "-s", action="store_true")
    parser.add_argument("--view", "-v", action="store_true")
    parser.add_argument("--data", "-d")
    parser.add_argument("--labels", "-l")
    args = parser.parse_args()

    data_path = args.data
    labels_path = args.labels
    (label_file_name, _) = os.path.splitext(args.labels)
    dataset_type = fo.types.COCODetectionDataset

    if args.split:
        signs_dataset = fo.Dataset.from_dir( \
                dataset_type=dataset_type, \
                data_path=data_path, \
                labels_path=labels_path, \
                classes=SIGN_CLASSES, \
                name="signs_dataset")

        lights_dataset = fo.Dataset.from_dir( \
                dataset_type=dataset_type, \
                data_path=data_path, \
                labels_path=labels_path, \
                classes=LIGHT_CLASSES, \
                name="lights_dataset")

        obstacles_dataset = fo.Dataset.from_dir( \
                dataset_type=dataset_type, \
                data_path=data_path, \
                labels_path=labels_path, \
                classes=OBSTACLE_CLASSES, \
                name="obstacles_dataset")

        if signs_dataset.stats()["samples_count"] > 0:
            signs_dataset.export(
                    labels_path=label_file_name + "_signs.json", \
                    dataset_type=dataset_type, \
                    classes=["RSVD"] + SIGN_CLASSES) # Class ID 0 is reserved for COCO datasets

            # Reformat JSON with line breaks and indentation and separate into super categories
            with open(label_file_name + "_signs.json", "r+") as f:
                data = json.load(f)
                for category in data["categories"]:
                    category["supercategory"] = SUPER_CATEGORY[category["name"]]
                j = json.dumps(data, indent=4)
                f.seek(0)
                f.write(j)
                f.truncate()

            print("Num sign annotations: " + str(len(data["annotations"])))

        if lights_dataset.stats()["samples_count"] > 0:
            lights_dataset.export(
                    labels_path=label_file_name + "_lights.json", \
                    dataset_type=dataset_type, \
                    classes=["RSVD"] + LIGHT_CLASSES) # Class ID 0 is reserved for COCO datasets

            # Reformat JSON with line breaks and indentation and separate into super categories
            with open(label_file_name + "_lights.json", "r+") as f:
                data = json.load(f)
                for category in data["categories"]:
                    category["supercategory"] = SUPER_CATEGORY[category["name"]]
                j = json.dumps(data, indent=4)
                f.seek(0)
                f.write(j)
                f.truncate()

            print("Num light annotations: " + str(len(data["annotations"])))

        if obstacles_dataset.stats()["samples_count"] > 0:
            obstacles_dataset.export(
                    labels_path=label_file_name + "_obstacles.json", \
                    dataset_type=dataset_type, \
                    classes=["RSVD"] + OBSTACLE_CLASSES) # Class ID 0 is reserved for COCO datasets

            # Reformat JSON with line breaks and indentation and separate into super categories
            with open(label_file_name + "_obstacles.json", "r+") as f:
                data = json.load(f)
                for category in data["categories"]:
                    category["supercategory"] = SUPER_CATEGORY[category["name"]]
                j = json.dumps(data, indent=4)
                f.seek(0)
                f.write(j)
                f.truncate()

            print("Num obstacle annotations: " + str(len(data["annotations"])))

    if args.view:
        dataset = fo.Dataset.from_dir( \
                dataset_type=dataset_type, \
                data_path=data_path, \
                labels_path=labels_path, \
                name="dataset")

        session = fo.launch_app(dataset)
        session.wait()
