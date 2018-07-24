import os
import sys
import numpy as np
import cv2
import boto3
import tempfile
# from data import VOC_CLASSES as labels
from matplotlib import pyplot as plt
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


'''
Takes a file of COCO like box predictions and generates a sequence of images with 
bounding boxes. The boxes in the prediction file must be ordered by image id. 
'''

if __name__ == "__main__":

    # colors = ['green', 'purple']  # plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    labels = {37: 'basketball', 1: 'person'}

    # prediction_file = '/Users/keith.landry/code/ssd.pytorch/eval/bbox_predictions_300.json'
    # prediction_file = '/Users/keith.landry/code/ssd.pytorch/eval/bbox_predictions_ssd300_76K_unanno_thresh.0.json'
    prediction_file = '/Users/keith.landry/code/ssd.pytorch/eval/diagonal-view-lookahead300-99k-thresh.0.json'
    # prediction_file = '/Users/keith.landry/code/ssd.pytorch/eval/bbox_predictions_lookahead300_unannotated_thresh.15.json'
    img_path = '/Users/keith.landry/data/internal-experiments/basketball/bhjc/20180123/images/left_cam/'
    img_path = '/Users/keith.landry/data/internal-experiments/basketball/bhjc/20180412/images/diagonal-view/'

    # img_prefix = 'left_scene2_rot180_'
    img_prefix = 'digonal_frame_'
    id_zeropadding = 4

    # img_ids = [i for i in range(700, 710)]
    with open(
            '/Users/keith.landry/code/ssd.pytorch/data/bhjc20180123_bball/bhjc_testonly.txt') as f:
        id_list = f.readlines()
    img_ids = [int(im_id.strip()) for im_id in id_list]

    img_ids = [i for i in range(1, 1201)]

    thresh = 0.5

    with open(prediction_file, encoding='utf-8') as data_file:
        data = json.loads(data_file.read())

    scores = []
    last_id = 'start'

    for detection in data:
        if detection['image_id'] in img_ids:
            if detection['image_id'] % 10 == 0:
                print(detection['image_id'])

            # new_image = True if detection['image_id'] != last_id else False
            if detection['image_id'] != last_id:
                new_image = True

                # all finished... write to file
                if last_id != 'start':
                    # outfile = '/Users/keith.landry/code/ssd.pytorch/data/output_imgs/unannot/ssd1166_300_76K_vanilla_permute/leftcam_detect_unannot_{}.png'.format(
                    #     str(last_id).zfill(id_zeropadding))
                    outfile = '/Users/keith.landry/code/ssd.pytorch/data/output_imgs/diag-view/lookahead_99K_diag_view_thres.5_{}.png'.format(
                        str(last_id).zfill(id_zeropadding))

                    cv2.imwrite(outfile, image)
                    # cv2.imshow('image', image)
                    # if cv2.waitKey():
                    #     cv2.destroyAllWindows()

                # fig = plt.gcf()
                # fig = plt.gca()
                # canvas = FigureCanvas(f)
                # canvas.draw()

                # img = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

                # plt.savefig(outfile)
                # plt.close()
            else:
                new_image = False

            if new_image:
                full_image_id = str(detection['image_id']).zfill(id_zeropadding)
                img_file = img_path + img_prefix + full_image_id + '.png'
                print(img_file)

                image = cv2.imread(img_file, cv2.IMREAD_COLOR)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                last_id = detection['image_id']

                # f = plt.figure(figsize=(10, 10))
                # plt.axis('off')
                # plt.imshow(rgb_image)  # plot the image for matplotlib
                # currentAxis = plt.gca()

            score = detection['score']
            if score >= thresh:
                pt = detection['bbox']
                # coords = (pt[0], pt[1]), pt[2] + 1, pt[3] + 1  # why the plus 1 (maybe so it's never zero?

                label_name = labels[detection['category_id']]
                display_txt = '%s: %.2f' % (label_name, score)

                # color = 'green'
                # currentAxis.add_patch(plt.Rectangle(*coords, fill=False,
                #                                     edgecolor=color, linewidth=1, alpha=.8))
                # currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.2})
                #
                # currentAxis.add_patch(plt.Rectangle(*coords, fill=False,
                #                                     edgecolor=color, linewidth=1, alpha=.8))
                # currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.2})

                x = np.round(pt[0]).astype(int)
                y = np.round(pt[1]).astype(int)
                w = np.round(pt[2]).astype(int)
                h = np.round(pt[3]).astype(int)

                rgb = [51, 51, 185]
                rgb2 = [221, 82, 123]

                cv2.rectangle(image, (x, y), (x + w, y + h), rgb, 2)

                label_overlay = image.copy()
                label_alpha = .75

                cv2.putText(label_overlay, display_txt, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, .5,
                            rgb, thickness=1, lineType=1)

                cv2.addWeighted(label_overlay, label_alpha, image, 1-label_alpha, 0, image)







