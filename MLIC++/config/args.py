import argparse


def train_options():
    parser = argparse.ArgumentParser(description="Training script.")
    parser.add_argument(
        "-exp",
        "--experiment",
        default="mlicplus0483mse",
        type=str,
        required=False,
        help="Experiment name"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="/home/npr/dataset/",
        type=str,
        required=False,
        help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=500,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.045,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="mse",
        help="Optimized for (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID"
    )
    parser.add_argument(
        "--cuda",
        default=True,
        help="Use cuda"
    )
    parser.add_argument(
        "--save",
        default=True,
        help="Save model to disk"
    )
    parser.add_argument(
        "--seed",
        type=float,
        default=192.1,
        help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        type=str,
        help="pretrained model path"
    )
    parser.add_argument(
        '--world_size',
        default=1,
        type=int,
        help='number of distributed processes'
    )
    parser.add_argument(
        '--dist_url',
        default='env://',
        help='url used to set up distributed training'
    )
    args = parser.parse_args()
    return args


def test_options():
    parser = argparse.ArgumentParser(description="Testing script.")
    parser.add_argument(
        "-exp",
        "--experiment",
        default="swlic_mse_0932_testv3",
        type=str,
        required=False,
        help="Experiment name"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="/home/npr/dataset/",
        type=str,
        required=False,
        help="Training dataset"
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=1,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="mse",
        help="Optimized for (default: %(default)s)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=3,
        help="GPU ID"
    )
    parser.add_argument(
        "--cuda",
        default=True,
        help="Use cuda"
    )
    parser.add_argument(
        "--save",
        default=True,
        help="Save model to disk"
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        type=str,
        help="pretrained model path"
    )
    args = parser.parse_args()
    return args
