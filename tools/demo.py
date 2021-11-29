from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import time
import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
torch.backends.cudnn.benchmark = True

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
parser.add_argument('--save_path', default='', type=str,
                    help='path of inference video')
parser.add_argument('--sfps', default=25.0, type=float,
                    help='fps of saved video')
args = parser.parse_args()

def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
            video_name.endswith('mp4'):

        cap = cv2.VideoCapture(args.video_name)

        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
                                     map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]

    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    writer = None
    init_rect = None
    for frame in get_frames(args.video_name):
        key = 255
        key = cv2.waitKey(40) & 0xff
        if key == ord("s"):
            if first_frame:
                try:
                    init_rect = cv2.selectROI(video_name, frame, False, False)
                    if args.save_path != "":

                        fourcc = cv2.VideoWriter_fourcc(*'MP42')
                        size = (frame.shape[1], frame.shape[0])
                        writer = cv2.VideoWriter(args.save_path, fourcc, args.sfps, size)
                except:
                    exit()
                tracker.init(frame, init_rect)
                first_frame = False
        elif key == ord("q"):
            break
        if init_rect is not None:
            time_s = time.time()
            writerputs = tracker.track(frame)
            time_cost = time.time() - time_s
            FPS = round(1/time_cost, 2)
            if 'polygon' in writerputs:
                polygon = np.array(writerputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((writerputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)


            else:
                bbox = list(map(int, writerputs['bbox']))
                cv2.putText(frame, f"FPS:{FPS}", org=(0, 80), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(0, 0, 255), thickness=3)
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                              (0, 255, 0), 3)
        cv2.imshow(video_name, frame)
        if args.save_path != "" and writer != None:

            writer.write(frame)

        cv2.waitKey(40)
        # else:
        #     cv2.imshow(video_name, frame)
            # cv2.waitKey(30)
    if writer != None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

