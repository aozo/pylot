import argparse
import json
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Splits data set into multiple data sets according to categories.")
    parser.add_argument("--file", "-f")
    args = parser.parse_args()
    (file_name, _) = os.path.splitext(args.file)

    f = open(file_name + ".json")
    data = json.load(f)

    sign_ids = []
    sign_img_ids = []
    sign_cats = []
    sign_imgs = []
    sign_anns = []

    light_ids = []
    light_img_ids = []
    light_cats = []
    light_imgs = []
    light_anns = []

    other_ids = []
    other_img_ids = []
    other_cats = []
    other_imgs = []
    other_anns = []

    for c in data["categories"]:
        if c["supercategory"] == "sign":
            sign_cats.append(c)
            sign_ids.append(c["id"])
        elif c["supercategory"] == "light":
            light_cats.append(c)
            light_ids.append(c["id"])
        else:
            other_cats.append(c)
            other_ids.append(c["id"])

    for a in data["annotations"]:
        if a["category_id"] in sign_ids:
            sign_anns.append(a)
            sign_img_ids.append(a["image_id"])
        elif a["category_id"] in light_ids:
            light_anns.append(a)
            light_img_ids.append(a["image_id"])
        else:
            other_anns.append(a)
            other_img_ids.append(a["image_id"])

    for i in data["images"]:
        if i["id"] in sign_img_ids:
            sign_imgs.append(i)
        elif i["id"] in light_img_ids:
            light_imgs.append(i)
        else:
            other_imgs.append(i)

    sign_json = {}
    sign_json["info"] = data["info"]
    sign_json["licenses"] = data["licenses"]
    sign_json["categories"] = sign_cats
    sign_json["images"] = sign_imgs
    sign_json["annotations"] = sign_anns
    j = json.dumps(sign_json, indent=4)
    f = open(file_name + "_signs.json", "w+")
    f.write(j)

    light_json = {}
    light_json["info"] = data["info"]
    light_json["licenses"] = data["licenses"]
    light_json["categories"] = light_cats
    light_json["images"] = light_imgs
    light_json["annotations"] = light_anns
    j = json.dumps(light_json, indent=4)
    f = open(file_name + "_lights.json", "w+")
    f.write(j)

    other_json = {}
    other_json["info"] = data["info"]
    other_json["licenses"] = data["licenses"]
    other_json["categories"] = other_cats
    other_json["images"] = other_imgs
    other_json["annotations"] = other_anns
    j = json.dumps(other_json, indent=4)
    f = open(file_name + "_other.json", "w+")
    f.write(j)
