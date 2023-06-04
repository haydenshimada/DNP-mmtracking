import os
import os.path as osp

import mmcv
import tempfile
from collections import defaultdict
from mmtrack.apis import inference_mot, init_model

def main():
    input_folder = "data/DNP/video/"
    output_ano = "output/DNP/anotations/"
    output_vid = "output/DNP/videos/"

    mot_config = 'configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py'
    mot_checkpoint = 'checkpoints/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth'
    mot_model = init_model(mot_config, mot_checkpoint, device='cuda:0')

    for input_file in os.listdir(input_folder):
        print(f"==========={input_file}==========")

        input_video = osp.join(input_folder, input_file)
        input_file = input_file.split(".")[0]
        
        imgs = mmcv.VideoReader(input_video)

        # build the model from a config file
        
        prog_bar = mmcv.ProgressBar(len(imgs))
        out_dir = tempfile.TemporaryDirectory()
        out_path = out_dir.name

        pred_file = osp.join(output_ano, input_file + ".json")
        output = osp.join(output_vid, input_file + ".mp4")

        out_data = defaultdict(list)

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

        # print out pred in json format
        mmcv.dump(out_data, pred_file)

        
        print(f'\n making the output video at {output} with a FPS of {imgs.fps}')
        mmcv.frames2video(out_path, output, fps=imgs.fps, fourcc='mp4v')
        out_dir.cleanup()
        
        print()

if __name__ == "__main__":
     main()