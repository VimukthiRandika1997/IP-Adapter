import os
import shutil
import json
import argparse
from PIL import Image
import huggingface_hub
from datasets import load_dataset
from tqdm import tqdm


# - Helper functions
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a dataset-creation.")
    parser.add_argument(
        "hf_token",
        type=str,
        default=None,
        help="hf token of huggingface account",
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        default=None,
        help="name of the dataset",
    )

    return parser.parse_args()


def authenticate_hf(hf_token):
    huggingface_hub.login(hf_token)


def download_dataset(dataset_name):
    ds = load_dataset(dataset_name)

    return ds


def write_to_json_file(json_data, save_path):
    with open(save_path, "w") as f:
        json.dump(json_data, f, indent=4)

    
def create_images_and_json_files(hf_ds, split="train", image_column_name="image", prompt_column_name="text"):
    json_arr = []

    save_dir_path = "data/images"
    if os.path.exists(save_dir_path):
        shutil.rmtree(save_dir_path)
    os.makedirs(save_dir_path)

    # loop through dataset
    for i, image_data in tqdm(enumerate(hf_ds[split])):
        image = image_data[image_column_name] # this is pil image
        prompt = image_data[prompt_column_name]


        image_name = f"{str(i)}.png"
        json_dict = {"image_file": image_name, "text": prompt}

        # add to the array
        json_arr.append(json_dict)

        # save image
        save_image_path = os.path.join(save_dir_path, image_name)
        image.save(save_image_path)
    
    # save the json file
    save_json_filepath = "data.json"
    write_to_json_file(json_data=json_arr, save_path=save_json_filepath)


def run(args):
    # 01. Authenticate hf    
    authenticate_hf(hf_token=args.hf_token)

    # 02. Download the dataset
    hf_ds = download_dataset(dataset_name=args.dataset_name)

    # 03. Create image directory and data.json file
    create_images_and_json_files(hf_ds=hf_ds)


# - Main execution
if __name__ == "__main__":
    args = parse_args()
    run(args)