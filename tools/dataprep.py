import sys 
sys.path.append("../")

import os 

import argparse  
from tqdm import tqdm


# tool output directory 
from constants import tool_out_dir


def main(args):
    img_files = os.listdir(args.images)

    eval_labels = {}
    # read val annotations
    print("READING EVAL ANNOTATIONS")
    with open(args.eval_annotations, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            img_name, label = line.split()
            eval_labels[img_name] = "train" if label == "0" else "val" if label == "1" else "test"
            
    # read id annotations
    print("READING ID ANNOTATIONS")
    with open(args.id_annotations, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            img_name, id_label = line.split()

            # define save path and create directory
            save_path = os.path.join(tool_out_dir, "data", eval_labels[img_name], id_label)
            os.makedirs(save_path, exist_ok=True)

            if img_name in img_files:
                os.system(f"cp {os.path.join(args.images, img_name)} {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-images',
                        '--images',
                        type=str,
                        required=True,
                        help="path to a directory containing images")
    
    parser.add_argument('-id_annotations',
                        '--id_annotations',
                        type=str,
                        required=True,
                        help="path to annotation .txt file containing image names and their corresponding id labels")
    
    parser.add_argument('-eval_annotations',
                        '--eval_annotations',
                        type=str,
                        required=True,
                        help="path to eval .txt file containing image names and their train/val/test labels")

    
    args = parser.parse_args()
    main(args)

    