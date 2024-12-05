import argparse
from pathlib import Path


def parse_opts(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',
                        default='/media/ttzhang/T7/CTS_final_dataset',
                        type=str,
                        help='Root directory path')
    parser.add_argument('--result_path2',
                        default='./mambaresnet_output',
                        type=Path,
                        help='Result directory path')
    parser.add_argument('--result_path',
                        default='./mambaresnet_output',
                        type=Path,
                        help='weight Result directory path')
    parser.add_argument(
        '--n_classes',
        default=2,
        type=int,
        help=
        'Number of classes (activitynet: 200, kinetics: 400 or 600, ucf101: 101, hmdb51: 51)'
    )
    parser.add_argument('--image_path',
                        required=True,
                        type=str,
                        help='Path to input image (.nii.gz file or DICOM folder)')
    parser.add_argument('--pretrain_path',
                        default=None,
                        type=Path,
                        help='Pretrained model path (.pth).')
    parser.add_argument(
        '--ft_begin_module',
        default='',
        type=str,
        help=('Module name of beginning of fine-tuning'
              '(conv1, layer1, fc, denseblock1, classifier, ...).'
              'The default means all layers are fine-tuned.'))
    parser.add_argument('--sample_size',
                        default=224,
                        type=int,
                        help='Height and width of inputs')
    parser.add_argument('--sample_duration',
                        default=30,
                        type=int,
                        help='Temporal duration of inputs')
    parser.add_argument('--resume_path',
                        default=None,
                        type=Path,
                        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--center_crop',
        default=False,
        type=bool,
        help='data augmentation'
    )
    parser.add_argument(
        '--flip',
        default=False,
        type=bool,
        help='data augmentation'
    )
    parser.add_argument(
        '--rot',
        default=False,
        type=bool,
        help='data augmentation'
    )
    parser.add_argument(
        '--resize_select',
        default=False,
        type=bool,
        help='data augmentation'
    )
    parser.add_argument('--no_cuda',
                        action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--n_threads',
                        default=0,
                        type=int,
                        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--model',
        default='mambaresnet',
        type=str,
        help=
        '(resnet | resnet2p1d | preresnet | wideresnet | resnext | densenet | mambaresnet')
    parser.add_argument('--n_input_channels',
                        default=1,
                        type=int,
                        help='channels')
    parser.add_argument('--model_depth',
                        default=34,
                        type=int,
                        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--conv1_t_size',
                        default=7,
                        type=int,
                        help='Kernel size in t dim of conv1.')
    parser.add_argument('--conv1_t_stride',
                        default=2,
                        type=int,
                        help='Stride in t dim of conv1.')
    parser.add_argument('--dilation_flag',
                        default=True,
                        type=bool,
                        help='define the dilation in ResNet')
    parser.add_argument('--no_max_pool',
                        action='store_true',
                        help='If true, the max pooling after conv1 is removed.')
    parser.add_argument('--resnet_shortcut',
                        default='B',
                        type=str,
                        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--resnet_widen_factor',
        default=1.0,
        type=float,
        help='The number of feature maps of resnet is multiplied by this value')
    parser.add_argument('--wide_resnet_k',
                        default=2,
                        type=int,
                        help='Wide resnet k')
    parser.add_argument('--resnext_cardinality',
                        default=32,
                        type=int,
                        help='ResNeXt cardinality')
    parser.add_argument('--input_type',
                        default='rgb',
                        type=str,
                        help='(rgb | flow)')
    parser.add_argument('--manual_seed',
                        default=1,
                        type=int,
                        help='Manually set random seed')
    parser.add_argument('--accimage',
                        action='store_true',
                        help='If true, accimage is used to load images.')
    parser.add_argument('--output_topk',
                        default=5,
                        type=int,
                        help='Top-k scores are saved in json file.')
    parser.add_argument('--file_type',
                        default='jpg',
                        type=str,
                        help='(jpg | hdf5)')
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Use multi-processing distributed training to launch '
        'N processes per node, which has N GPUs.')
    parser.add_argument('--dist_url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--world_size',
                        default=-1,
                        type=int,
                        help='number of nodes for distributed training')

    args = parser.parse_args(args)

    return args
