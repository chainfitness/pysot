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

class Tracker:
    def __init__(self, video_name, bdbox = None):
        self.video_name = video_name
        self.bdbox = bdbox
        self.writer = None
        self.bdbox_flag = True
        self.first_frame = True
        self.pred_bdbox_list = []

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
        self.tracker = build_tracker(model)
        if self.video_name:
            self.window = self.video_name.split('/')[-1].split('.')[0]

        else:
            self.window = 'webcam'

    def pred_n_visualization(self):
        cv2.namedWindow(self.window, cv2.WND_PROP_FULLSCREEN)
        init_rect = None
        for frame in self.get_frames(self.video_name):

            key = 255
            key = cv2.waitKey(40) & 0xff

            if self.bdbox != None and self.bdbox_flag:
                bbox_list = self.bdbox.split(",")
                bbox_list = [float(x) for x in bbox_list]
                init_rect = tuple(bbox_list)
                self.tracker.init(frame, init_rect)
                fourcc = cv2.VideoWriter_fourcc(*'MP42')
                size = (frame.shape[1], frame.shape[0])
                self.writer = cv2.VideoWriter(args.save_path, fourcc, args.sfps, size)
                self.bdbox_flag = False

            if key == ord("s"):
                if self.first_frame:
                    try:
                        init_rect = cv2.selectROI(self.window, frame, False, False)
                        if args.save_path != "":
                            fourcc = cv2.VideoWriter_fourcc(*'MP42')
                            size = (frame.shape[1], frame.shape[0])
                            self.writer = cv2.VideoWriter(args.save_path, fourcc, args.sfps, size)
                    except:
                        exit()
                    self.tracker.init(frame, init_rect)
                    self.first_frame = False
            elif key == ord("q"):
                break
            if init_rect is not None:
                time_s = time.time()
                writerputs = self.tracker.track(frame)
                time_cost = time.time() - time_s
                FPS = round(1 / time_cost, 2)
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
                    self.pred_bdbox_list.append(bbox)
                    cv2.putText(frame, f"FPS:{FPS}", org=(0, 80), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2,
                                color=(0, 0, 255), thickness=3)
                    cv2.rectangle(frame, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                  (0, 255, 0), 3)
            cv2.imshow(self.window, frame)
            if args.save_path != "" and self.writer != None:
                self.writer.write(frame)

            cv2.waitKey(40)
            # else:
            #     cv2.imshow(video_name, frame)
            # cv2.waitKey(30)
        if self.writer != None:
            self.writer.release()
        cv2.destroyAllWindows()
        return self.pred_bdbox_list
    def just_get_pred_bdbox(self):
        init_rect = None
        for frame in self.get_frames(self.video_name):
            if self.bdbox != None and self.bdbox_flag:
                bbox_list = self.bdbox.split(",")
                bbox_list = [float(x) for x in bbox_list]
                init_rect = tuple(bbox_list)
                self.tracker.init(frame, init_rect)
                fourcc = cv2.VideoWriter_fourcc(*'MP42')
                size = (frame.shape[1], frame.shape[0])
                self.writer = cv2.VideoWriter(args.save_path, fourcc, args.sfps, size)
                self.bdbox_flag = False
            if init_rect is not None:
                writerputs = self.tracker.track(frame)
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
                    self.pred_bdbox_list.append(bbox)
                    print(bbox)

    def get_pred_bdbox(self):
        return self.pred_bdbox_list

    def get_frames(self, video_name):
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

            cap = cv2.VideoCapture(self.video_name)

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

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    torch.set_num_threads(1)

    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str, help='config file')
    parser.add_argument('--snapshot', type=str, help='model name')
    parser.add_argument('--video_name', default='', type=str,
                        help='videos or image files')
    parser.add_argument('--save_path', default='', type=str,
                        help='path of inference video')
    parser.add_argument('--sfps', default=15.0, type=float,
                        help='fps of saved video')
    args = parser.parse_args()
    # 想要初始化第一帧的bdox坐标（X1,Y1,X2,Y2）
    bdbox = '316,792,404,123'
    # 传入bdbox坐标实例化Tracker类并调用预测并可视化接口，若不传入bdbox进行实例化，则需手动按s选择后按空格开始跟踪。
    pred_bdbox_list = Tracker(args.video_name).pred_n_visualization()
    print(pred_bdbox_list)
    # Tracker(args.video_name, bdbox).just_get_pred_bdbox()
