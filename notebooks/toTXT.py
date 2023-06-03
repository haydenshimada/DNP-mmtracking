# This script converts predicted data after forward to model into MOT labels.

# FROM
# - The value of key 'det_bboxes' is list with length
#   num_classes, and each element in list is ndarray with
#   shape(n, 5) in [tl_x, tl_y, br_x, br_y, score] format.
# - The value of key 'track_bboxes' is list with length
#   num_classes, and each element in list is ndarray with
#   shape(n, 6) in [id, tl_x, tl_y, br_x, br_y, score] format.

#TO
# <frame_id>, <instance_id>, <x1>, <y1>, <w>, <h>, <conf>,
#       <x>, <y>, <z> # for 3D objects
# Classes in MOT:
#   1: 'pedestrian'
#   2: 'person on vehicle'
#   3: 'car'
#   4: 'bicycle'
#   5: 'motorbike'
#   6: 'non motorized vehicle'
#   7: 'static person'
#   8: 'distractor'
#   9: 'occluder'
#   10: 'occluder on the ground',
#   11: 'occluder full'
#   12: 'reflection'

#INPUT
#input: path to folder contains pred output of inference
#output: path to folder contains the results 

import argparse
import os
import os.path as osp
from collections import defaultdict
import re

import mmcv
import numpy as np
from tqdm import tqdm
import json

def parse_args():
    parser = argparse.ArgumentParser(
        description='Converts predicted data after forward to model into MOT labels.')
    parser.add_argument('-i', '--input', help='path of pred data', default='output/DNP/anotations')
    parser.add_argument('-o', '--output', help='path to saved converted MOT data', default='output/DNP/MOTformat')
    return parser.parse_args()

def parse_pred(pred_data):
    converted_tracks = []
    for frame in pred_data.keys():
        # print(pred_data[frame][0]['track_bboxes'][0])
        pred_tracks = pred_data[frame][0]['track_bboxes'][0]
        for pred_track in pred_tracks:
            instance_id, tl_x, tl_y, br_x, br_y, score = pred_track
            x1 = tl_x
            y1 = tl_y
            w = br_x - tl_x
            h = br_y - tl_y
            converted_tracks.append([int(frame) + 1, int(instance_id) + 1, x1, y1, w, h, 1, -1, -1, -1])
    print(len(converted_tracks))
    return converted_tracks
        
        

def main():
    args = parse_args()
    PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(PARENT_DIR)
    in_path = os.path.join(ROOT_DIR, args.input)
    out_path = os.path.join(ROOT_DIR, args.output)
    args = parse_args()
    # if not osp.isdir(out_path):
    #     os.makedirs(out_path)
    in_folder = osp.join(in_path)
    pred_files = os.listdir(in_folder)
    for pred_file in tqdm(pred_files):
        #take number of data file ( log001.txt -> 001)
        num = re.findall(r'\d+', pred_file)
        if (len(num) < 0):
            raise ValueError('Pred folder not follow the correct format')
        
        converted_file = osp.join(out_path, f'output{num[0]}.txt')
        # pred_data = mmcv.dict_from_file(osp.join(in_folder, pred_file))
        with open(osp.join(in_folder, pred_file)) as file:
            pred_data_str = file.read()
        pred_data = json.loads(pred_data_str)
        pred2mot = parse_pred(pred_data)
        with open(converted_file, 'w') as file:
            for _ in pred2mot:
                line = ','.join(str(item) for item in _) + '\n'
                file.write(line)

if __name__ == '__main__':
    main()