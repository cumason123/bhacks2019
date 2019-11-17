import argparse

from classifier.train import train
from classifier.eval import evaluate


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', default='balanced_data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--freeze-weights', action='store_true')
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--learning-rate', type=int, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-dir', type=str, default='trained_models')
    parser.add_argument('--model-file', type=str, default='model.pt')
    parser.add_argument('--pretrained-model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet101', 'resnet152', 'wide_resnet50_2',
                                 'wide_resnet101_2', 'resnet50', 'resnext50_32x4d', 'resnext101_32x8d'])
    parser.add_argument('--eval', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    if args.eval:
        evaluate(args)
    else:
        train(parse())
