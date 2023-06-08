import os
import os.path as osp

import time
import argparse
import subprocess

import mmcv
import tempfile
from collections import defaultdict
from mmtrack.apis import inference_mot, init_model

def parse_args():
    parser = argparse.ArgumentParser(description='Run mmtracking')
    parser.add_argument('-config', help='path to config file', default=None)
    parser.add_argument('-checkpoint', help='checkpoint download link', default=None)
    return parser.parse_args()

def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

def main():
    args = parse_args()

    assert args.config, "No config file"

    input_folder = "data/DNP/video/"
    output_ano = "output/DNP/anotations/"
    output_vid = "output/DNP/videos/"

#     mot_config = 'configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py'
#     mot_checkpoint = 'checkpoints/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth'
    mot_config = args.config
    mot_checkpoint = osp.join("checkpoints", args.checkpoint.split('/')[-1])

    # if checkpoint is not downloaded, download it
    if not osp.exists(mot_checkpoint):
        print("Downdloading checkpoint ...")
        runcmd(f"wget -c {args.checkpoint} -P ./checkpoints")
        print("Finished")

    mot_model = init_model(mot_config, mot_checkpoint, device='cuda:0')

    file = open('output/DNP/time.txt', mode='w')

    for input_file in sorted(os.listdir(input_folder)):
        print(f"==========={input_file}==========\n")
        file.write(f"==========={input_file}==========\n")

        input_video = osp.join(input_folder, input_file)
        input_file = input_file.split(".")[0]
        
        imgs = mmcv.VideoReader(input_video)
        file.write(f"Video's FPS: {imgs.fps}\n")

        # build the model from a config file
        
        prog_bar = mmcv.ProgressBar(len(imgs))
        out_dir = tempfile.TemporaryDirectory()
        out_path = out_dir.name

        pred_file = osp.join(output_ano, input_file + ".json")
        output = osp.join(output_vid, input_file + ".mp4")

        out_data = defaultdict(list)

        start_time = time.time()
        # test and show/save the images
        for i, img in enumerate(imgs):
                result = inference_mot(mot_model, img, frame_id=i)
                out_data[i].append(result)
                mot_model.show_result(
                                img,
                                result,
                                show=False,
                                wait_time=int(1000. / imgs.fps),
                                out_file=f'{out_path}/{i:06d}.jpg')
                prog_bar.update()
        end_time = time.time()
        file.write("Tracking time: %s seconds\n" % (end_time - start_time))

        # print out pred in json format
        start_time = time.time()
        mmcv.dump(out_data, pred_file)
        end_time = time.time()
        file.write("Create json anotations: %s seconds\n" % (end_time - start_time))
        
        print(f'\nMaking the output video at {output} with a FPS of {imgs.fps}\n')
        start_time = time.time()
        mmcv.frames2video(out_path, output, fps=imgs.fps, fourcc='mp4v')
        end_time = time.time()
        file.write("Create output video: %s seconds\n" % (end_time - start_time))

        out_dir.cleanup()

        file.write('\n')
        print()
        
    file.close()

if __name__ == "__main__":
     main()