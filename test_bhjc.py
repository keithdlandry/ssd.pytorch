from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from data import BaseTransform
from data.bhjc20180123_bball.bhjc import BhjcBballDataset, AnnotationTransformBhjc
import torch.utils.data as data
from ssd import build_ssd
from master_config import configs
import json


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
# parser.add_argument('--trained_model', default='weights/ssd1166_bhjctrained_iter104000_ballonlysquare.pth',
#                     type=str, help='Trained state_dict file path to open')
parser.add_argument('--trained_model', default='weights/ssd1166_300_iter76000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
# parser.add_argument('--id_file', default='/Users/keith.landry/code/ssd.pytorch/data/bhjc20180123_bball/bhjc_testonly.txt')
parser.add_argument('--id_file', default='/home/ec2-user/computer_vision/bball_detection/ssd.pytorch/data/bhjc20180123_bball/bhjc_testonly.txt')
parser.add_argument('--ball_only', default=True, type=str2bool)
parser.add_argument('--square_boxes', default=True, type=str2bool)
parser.add_argument('--anno_dir', default='/home/ec2-user/computer_vision/bball_detection/ssd.pytorch/data/bhjc20180123_bball/annotations/')
parser.add_argument('--img_dir', default='/home/ec2-user/computer_vision/bball_detection/ssd.pytorch/data/bhjc20180123_bball/images/')
parser.add_argument('--outname', default='bbox_predictions_ssd300_testonly_thresh.0.json')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, net_name):

    if args.ball_only:
        labelmap = ['basketball']
    else:
        labelmap = ('person', 'basketball')

    predictions = []

    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder + args.outname
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        # img = testset.pull_image(i)
        # img_id, img, annotation = testset.pull_image_anno(i)
        img_id, img = testset.pull_image(i)
        x = transform(img)[0]

        # this .copy line makes a huge difference (at least on lookahead architecture)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])

        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= args.visual_threshold:
                # if pred_num == 0:
                #     with open(filename, mode='a') as f:
                #         f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                # convert to width and height as is the COCO standard
                pt[2] = pt[2] - pt[0]
                pt[3] = pt[3] - pt[1]
                # use coco label ids:
                if label_name == 'person':
                    label_id = 1
                if label_name == 'basketball':
                    label_id = 37  # label id for "sports ball"
                else:
                    raise ValueError('label name must be either person or basketball')

                predictions.append({'bbox': pt.tolist(),
                                    'category_id': label_id,
                                    'image_id': int(img_id),
                                    'score': score
                                    })

                j += 1
                if j == detections.shape[2]:
                    break

    with open(filename, 'w') as outfile:
        json.dump(predictions, outfile)


if __name__ == '__main__':
    # load net
    if args.ball_only:
        class_dict = configs['classes']['ball_only']
    else:
        class_dict = configs['classes']['all_class']

    num_classes = len(class_dict) + 1
    network_name = '300'

    net = build_ssd('test', configs, network_name, num_classes, square_boxes=args.square_boxes)
    # net = build_ssd('test', 300, num_classes) # initialize SSD
    # net.load_state_dict(torch.load(args.trained_model))
    net.load_weights(args.trained_model)

    net.eval()  # required for dropout or batchnorm layers (not using any)
    print('Finished loading model!')

    # load data
    with open(args.id_file) as f:
        test_image_ids = f.readlines()
        test_image_ids = [im_id.rstrip() for im_id in test_image_ids]

    # use unannotated images instead
    # test_image_ids = [str(i).zfill(5) for i in range(800, 1805)]

    # test_image_ids = ['00700', '00701']

    test_set = BhjcBballDataset(
        args.anno_dir, args.img_dir, test_image_ids, None,
        AnnotationTransformBhjc(ball_only=args.ball_only, class_to_ind=class_dict))

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    # evaluation
    test_net(args.save_folder, net, args.cuda, test_set,
             BaseTransform(net.size, (104, 117, 123)), net_name=network_name)
