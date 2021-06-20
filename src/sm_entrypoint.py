import json
import logging
import os
from io import BytesIO

import boto3
import configargparse
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import transforms

CONFIG_FILENAME = 'configs/eccv_final_model'
CHECKPOINT_PATH_FOLDER = 'pretrained_weights'
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'
PNG_CONTENT_TYPE = 'image/png'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# same loader used during training
inference_loader = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CHOICES_DATA_USED = [
    "pos_only",
    "pos_and_neg"
]
CHOICES_ARCHITECTURES = [
    "resnet18",
    "resnet50"
]
CHOICES_ACTIVATION = [
    "softmax",
    "sigmoid"
]
CHOICES_LOSSES = [
    "softmax_cross_entropy",
    "sigmoid_bce_no_masking",
    "sigmoid_bce_with_masking"
]
INCIDENTS = [
    "damaged",
    "flooded",
    "dirty contamined",
    "blocked",
    "collapsed",
    "snow covered",
    "under construction",
    "burned",
    "on fire",
    "with smoke",
    "ice storm",
    "drought",
    "dust sand storm",
    "thunderstorm",
    "wildfire",
    "tropical cyclone",
    "heavy rainfall",
    "tornado",
    "derecho",
    "earthquake",
    "landslide",
    "mudslide mudflow",
    "rockslide rockfall",
    "snowslide avalanche",
    "volcanic eruption",
    "sinkhole",
    "storm surge",
    "fog",
    "hailstorm",
    "dust devil",
    "fire whirl",
    "traffic jam",
    "ship boat accident",
    "airplane accident",
    "car accident",
    "train accident",
    "bus accident",
    "bicycle accident",
    "motorcycle accident",
    "van accident",
    "truck accident",
    "oil spill",
    "nuclear explosion",
]


def get_parser():
    parser = configargparse.ArgumentParser(description="Incident Model Parser.")
    parser.add_argument('-c',
                        '--config',
                        required=True,
                        is_config_file=True,
                        help='Config file path.')
    parser.add_argument("--mode",
                        default="train",
                        required=True,
                        type=str,
                        choices=["train", "test"],
                        help="How to use the model, such as 'train' or 'test'.")
    parser.add_argument("--checkpoint_path",
                        default="pretrained_weights/",
                        type=str,
                        help="Path to checkpoints for training.")

    # TODO: make sure to use this
    parser.add_argument("--images_path",
                        default="data/images/",
                        help="Path to the downloaded images.")

    parser.add_argument("--dataset_train",
                        default="data/eccv_train.json")
    parser.add_argument("--dataset_val",
                        default="data/eccv_val.json")
    parser.add_argument("--dataset_test",
                        default="data/eccv_test.json")

    parser.add_argument('--num_gpus',
                        default=4,
                        type=int,
                        help='Number of gpus to use.')
    parser.add_argument('-b',
                        '--batch_size',
                        default=256,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('--loss',
                        type=str,
                        choices=CHOICES_LOSSES)
    parser.add_argument('--activation',
                        type=str,
                        choices=CHOICES_ACTIVATION,
                        required=True)
    parser.add_argument('--dataset',  # TODO: could use a better name than "dataset" here
                        default='pos_only',
                        help='Which dataset to train with.',
                        choices=CHOICES_DATA_USED)
    parser.add_argument('--arch',
                        '-a',
                        metavar='ARCH',
                        default='resnet18',
                        choices=CHOICES_ARCHITECTURES,
                        help='Which model architecture to use.')
    parser.add_argument('--ignore_places_during_training',
                        default="False",
                        type=str)
    parser.add_argument('--percent_of_training_set',
                        default=100,
                        type=int)
    parser.add_argument('--pretrained_with_places',
                        default="True",
                        type=str)
    parser.add_argument('-j',
                        '--workers',
                        default=16,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs',
                        default=40,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.0001,
                        type=float,
                        metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--weight-decay',
                        '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq',
                        '-p',
                        default=10,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--pretrained',
                        dest='pretrained',
                        action='store_false',
                        help='use pre-trained model')
    parser.add_argument('--num-places',
                        default=49,
                        type=int,
                        help='num of class in the model')
    parser.add_argument('--num-incidents',
                        default=43, type=int)
    parser.add_argument('--fc-dim',
                        default=1024,
                        type=int,
                        help='output dimension of network')
    return parser


def get_incident_to_index_mapping():
    incident_to_index_mapping = {}
    for idx, incident in enumerate(INCIDENTS):
        incident_to_index_mapping[incident] = idx
    return incident_to_index_mapping


def get_index_to_incident_mapping():
    x = get_incident_to_index_mapping()
    # https://dev.to/renegadecoder94/how-to-invert-a-dictionary-in-python-2150
    x = dict(map(reversed, x.items()))
    return x


def get_trunk_model(args, model_dir):
    if args.pretrained_with_places:
        print("loading places weights for pretraining")
        model = models.__dict__[args.arch](num_classes=365)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.arch == "resnet18":
            model_file = os.path.join(model_dir, "pretrained_weights/resnet18_places365.pth.tar")
            checkpoint = torch.load(model_file, map_location=device)
            state_dict = {str.replace(k, 'module.', ''): v for k,
                                                               v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)
            model.fc = nn.Linear(512, 1024)
            model = nn.Sequential(model, nn.ReLU())
        elif args.arch == "resnet50":
            model_file = os.path.join(model_dir, "pretrained_weights/resnet50_places365.pth.tar")
            checkpoint = torch.load(model_file, map_location=device)
            state_dict = {str.replace(k, 'module.', ''): v for k,
                                                               v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)
            model.fc = nn.Linear(2048, 1024)
            model = nn.Sequential(model, nn.ReLU())
        return model
    else:
        print("loading imagenet weights for pretraining")
        # otherwise load with imagenet weights
        if args.arch == "resnet18":
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(512, 1024)
            model = nn.Sequential(model, nn.ReLU())
        elif args.arch == "resnet50":
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(2048, 1024)
            model = nn.Sequential(model, nn.ReLU())
        return model


def get_incident_layer(args):
    if args.activation == "softmax":
        return nn.Linear(args.fc_dim, args.num_incidents + 1)
    elif args.activation == "sigmoid":
        return nn.Linear(args.fc_dim, args.num_incidents)


def get_place_layer(args):
    if args.activation == "softmax":
        return nn.Linear(args.fc_dim, args.num_places + 1)
    elif args.activation == "sigmoid":
        return nn.Linear(args.fc_dim, args.num_places)


def get_incidents_model(args, model_dir):
    """
    Returns [trunk_model, incident_layer, place_layer]
    """
    # the shared feature trunk model
    trunk_model = get_trunk_model(args, model_dir)
    # the incident model
    incident_layer = get_incident_layer(args)
    # the place model
    place_layer = get_place_layer(args)

    print("Let's use", args.num_gpus, "GPUs!")
    trunk_model = torch.nn.DataParallel(trunk_model, device_ids=range(args.num_gpus))
    incident_layer = torch.nn.DataParallel(incident_layer, device_ids=range(args.num_gpus))
    place_layer = torch.nn.DataParallel(place_layer, device_ids=range(args.num_gpus))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trunk_model.to(device)
    incident_layer.to(device)
    place_layer.to(device)
    return [trunk_model, incident_layer, place_layer]


def update_incidents_model_with_checkpoint(incidents_model, args):
    """
    Update incidents model with checkpoints (in args.checkpoint_path)
    """

    trunk_model, incident_layer, place_layer = incidents_model

    # optionally resume from a checkpoint
    # TODO: bring in the original pretrained weights maybe?
    # TODO: remove the args.trunk_resume, etc.
    # TODO: remove path prefix

    config_name = os.path.basename(args.config)
    print(config_name)

    trunk_resume = os.path.join(
        args.checkpoint_path, "{}_trunk.pth.tar".format(config_name))
    place_resume = os.path.join(
        args.checkpoint_path, "{}_place.pth.tar".format(config_name))
    incident_resume = os.path.join(
        args.checkpoint_path, "{}_incident.pth.tar".format(config_name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for (path, net) in [(trunk_resume, trunk_model), (place_resume, place_layer), (incident_resume, incident_layer)]:
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location=device)
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            print("Loaded checkpoint '{}' (epoch {}).".format(path, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'.".format(path))


def update_incidents_model_to_eval_mode(incidents_model):
    print("Switching to eval mode.")
    for m in incidents_model:
        # switch to evaluation mode
        m.eval()


def model_fn(model_dir):
    """Load models"""
    logger.info('START model_fn')

    # Assemble arguments
    parser = get_parser()
    args = parser.parse_args(
        args=f"--config={model_dir}/{CONFIG_FILENAME} " +
             f"--checkpoint_path={model_dir}/{CHECKPOINT_PATH_FOLDER} " +
             "--mode=test " +
             "--num_gpus=0")

    # Create models
    incident_model = get_incidents_model(args, model_dir)
    # Load pretrained weights
    update_incidents_model_with_checkpoint(incident_model, args)
    # Change mode into eval
    update_incidents_model_to_eval_mode(incident_model)

    logger.info('END model_fn')
    return incident_model


def input_fn(request_body, content_type=JSON_CONTENT_TYPE):
    """Convert request body into input of models"""
    logger.info("START input_fn")
    if content_type == JPEG_CONTENT_TYPE or content_type == PNG_CONTENT_TYPE or content_type == JSON_CONTENT_TYPE:
        if content_type == JPEG_CONTENT_TYPE or content_type == PNG_CONTENT_TYPE:
            f = BytesIO(request_body)
        else:
            # Get image from S3
            s3 = boto3.resource("s3")
            bucket = s3.Bucket(request_body["s3_bucket_name"])
            obj = bucket.Object(request_body["s3_object_name"])
            response = obj.get()
            f = BytesIO(response["Body"].read())
        try:
            input_data = Image.open(f).convert('RGB')
            input_data = inference_loader(input_data)
        except:
            input_data = Image.new('RGB', (300, 300), 'white')
            input_data = inference_loader(input_data)

    else:
        logger.error(f'Content-Type invalid: {content_type}')
        input_data = {'errors': [f'Content-Type invalid: {content_type}']}

    logger.info("END input_fn")
    return input_data


def predict_fn(input_data, model):
    """Predict from input"""
    logger.info("START predict_fn")
    if isinstance(input_data, dict) and "errors" in input_data:
        logger.info("SKIP predict_fn")
        logger.info("END predict_fn")
        return input_data

    trunk_model, incident_layer, _ = model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_on_device = input_data.to(device)
    output = trunk_model(input_on_device)
    incident_output = incident_layer(output)
    incident_output = F.softmax(incident_output, dim=1)

    logger.info("END predict_fn")

    return incident_output


def output_fn(prediction, accept=JSON_CONTENT_TYPE):
    """Convert output of models into response body"""
    logger.info("START output_fn")
    logger.info(f"Accept: {accept}")

    if isinstance(prediction, dict) and "errors" in prediction:
        logger.info("SKIP output_fn")
        response = json.dumps(prediction)
        content_type = JSON_CONTENT_TYPE
    else:
        incident_map = get_index_to_incident_mapping()
        topk = 3
        incident_threshold = 0.5

        incident_probs, incident_idx = prediction.sort(1, True)
        incidents = []
        probs = incident_probs[0].cpu().detach().numpy()[:topk].tolist()
        if probs[0] < incident_threshold:
            incidents.append("no incident")
        else:
            for idx in incident_idx[0].cpu().numpy()[:topk]:
                if idx < len(incident_map):
                    incidents.append(incident_map[idx])
                else:
                    incidents.append("no incident")

        response = json.dumps({"results": {"incidents": incidents, "probs": probs}})
        content_type = JSON_CONTENT_TYPE

    logger.info("END output_fn")

    return response, content_type
