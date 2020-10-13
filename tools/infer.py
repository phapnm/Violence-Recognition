from mmaction.apis import init_recognizer, inference_recognizer
from torchvision import models
from torchsummary import summary
import torch
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Inference a recognizer')
    parser.add_argument('input', help='input video file path')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    config_file_sf = '/home/run/phap/mmlab/mmaction2/configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_kinetics400_rgb.py'

    # download the checkpoint from model zoo and put it in `checkpoints/`
    checkpoint_file_sf = '/home/run/phap/mmlab/mmaction2/work_dirs/dataset5/latest.pth'

    start = time.time()
    model_sf = init_recognizer(config_file_sf, checkpoint_file_sf, device='cpu')

    # test a single video and show the result:
    label = '/home/run/phap/mmlab/mmaction2/demo/label.txt'

    #results_tsn = inference_recognizer(model_tsn, video, label)
    results_sf = inference_recognizer(model_sf, args.input, label)
    print(results_sf)
    print('time:', time.time() - start)

if __name__ == '__main__':
    main()