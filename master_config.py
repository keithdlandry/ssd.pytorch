# config.py
import os.path

# gets home dir cross platform
home = os.path.expanduser("~")
ddir = os.path.join(home,"code/ssd.pytorch/data/VOCdevkit/")

# note: if you used our download scripts, this should be right
VOCroot = ddir # path to VOCdevkit root dir

# default batch size
BATCHES = 32
# data reshuffled at every epoch
SHUFFLE = True
# number of subprocesses to use for data loading
WORKERS = 4

configs = {

    'classes': {
        'ball_only': {'basketball': 0},
        'all_class': {'basketball': 0, 'person': 1}
    },

    'trunc': {
        'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512],
        'layers5to7': False,
        'extras': [],
        'mbox': [4],
        'vgg_source': [-2],
        'final_base_layer_dim': 512,
        'box_configs': {
            'feature_maps': [146],
            'min_dim': 1166,
            'steps': [8],  # default for 1166
            'min_sizes': [30],  # default for 300
            'max_sizes': [60],  # default for 300
            'aspect_ratios': [[2]],  # default for 300
            'square_only': False,  # This should be set
            'variance': [0.1, 0.2],
            'clip': True,
            'name': 'v2',
            'center_step_size': 1
        }
    },

    '300': {
        'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
        'layers5to7': True,
        'extras': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
        'mbox': [4, 6, 6, 6, 4, 4],
        'vgg_source': [21, -2],
        'final_base_layer_dim': 1024,
        'box_configs': {
            'feature_maps': [146, 73, 37, 19, 17, 15],
            'min_dim': 1166,
            'steps': [8, 16, 32, 64, 69, 78],  # default for 1166
            'min_sizes': [30, 60, 111, 162, 213, 264],  # default for 300
            'max_sizes': [60, 111, 162, 213, 264, 315],  # default for 300
            'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],  # default for 300
            'square_only': False,
            'variance': [0.1, 0.2],
            'clip': True,
            'name': 'v2',
            'center_step_size': 1
        }
    },

    'small583': {
        'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
        'layers5to7': True,
        'extras': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
        'mbox': [4, 6, 6, 6, 4, 4],
        'vgg_source': [21, -2],
        'final_base_layer_dim': 1024,
        'box_configs': {
            'feature_maps': [73, 36, 18, 9, 7, 5],
            'min_dim': 1166,
            'steps': [8, 16, 32, 64, 69, 78],  # default for 1166
            'min_sizes': [30, 60, 111, 162, 213, 264],  # default for 300
            'max_sizes': [60, 111, 162, 213, 264, 315],  # default for 300
            'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],  # default for 300
            'square_only': False,
            'variance': [0.1, 0.2],
            'clip': True,
            'name': 'v2',
            'center_step_size': 1
        }
    }
}
