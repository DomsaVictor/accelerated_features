# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use


from PIL import Image
import torch
from os import path
import numpy as np

from torchvision.transforms import ToTensor, Resize, Compose

from modules.xfeat import XFeat

# Kapture is a pivot file format, based on text and binary files, used to describe SfM (Structure From Motion)
# and more generally sensor-acquired data
# it can be installed with
# pip install kapture
# for more information check out https://github.com/naver/kapture

import kapture
from kapture.io.records import get_image_fullpath
from kapture.io.csv import kapture_from_dir
from kapture.io.csv import get_feature_csv_fullpath, keypoints_to_file, descriptors_to_file
from kapture.io.features import get_keypoints_fullpath, keypoints_check_dir, image_keypoints_to_file
from kapture.io.features import get_descriptors_fullpath, descriptors_check_dir, image_descriptors_to_file
from kapture.io.csv import get_all_tar_handlers

from tqdm import tqdm

def extract_kapture_keypoints(args):
    """
    Extract xFeat keypoints and descritors to the kapture format directly
    """
    print('extract_kapture_keypoints...')
    with get_all_tar_handlers(args.kapture_root,
                              mode={kapture.Keypoints: 'a',
                                    kapture.Descriptors: 'a',
                                    kapture.GlobalFeatures: 'r',
                                    kapture.Matches: 'r'}) as tar_handlers:
        kdata = kapture_from_dir(args.kapture_root, None,
                                 skip_list=[kapture.GlobalFeatures,
                                            kapture.Matches,
                                            kapture.Points3d,
                                            kapture.Observations],
                                 tar_handlers=tar_handlers)

        assert kdata.records_camera is not None
        image_list = [filename for _, _, filename in kapture.flatten(kdata.records_camera)]
        if args.keypoints_type is None:
            args.keypoints_type = path.splitext(path.basename(args.model))[0]
            print(f'keypoints_type set to {args.keypoints_type}')
        if args.descriptors_type is None:
            args.descriptors_type = path.splitext(path.basename(args.model))[0]
            print(f'descriptors_type set to {args.descriptors_type}')

        if kdata.keypoints is not None and args.keypoints_type in kdata.keypoints \
                and kdata.descriptors is not None and args.descriptors_type in kdata.descriptors:
            print('detected already computed features of same keypoints_type/descriptors_type, resuming extraction...')
            image_list = [name
                          for name in image_list
                          if name not in kdata.keypoints[args.keypoints_type] or
                          name not in kdata.descriptors[args.descriptors_type]]

        if len(image_list) == 0:
            print('All features were already extracted')
            return
        else:
            print(f'Extracting xFeat features for {len(image_list)} images')

        iscuda = 'cuda' if torch.cuda.is_available() else 'cpu'


        xfeat = XFeat(top_k=args.top_k, weights=args.model)

        transforms = Compose([
            ToTensor(),
            Resize((480, 640)) # VGA resolution for XFeat
        ])

        if kdata.keypoints is None:
            kdata.keypoints = {}
        if kdata.descriptors is None:
            kdata.descriptors = {}

        if args.keypoints_type not in kdata.keypoints:
            keypoints_dtype = None
            keypoints_dsize = None
        else:
            keypoints_dtype = kdata.keypoints[args.keypoints_type].dtype
            keypoints_dsize = kdata.keypoints[args.keypoints_type].dsize
        if args.descriptors_type not in kdata.descriptors:
            descriptors_dtype = None
            descriptors_dsize = None
        else:
            descriptors_dtype = kdata.descriptors[args.descriptors_type].dtype
            descriptors_dsize = kdata.descriptors[args.descriptors_type].dsize


        if args.dense == True:
            extract_function = xfeat.detectAndComputeDense
            third_arg = "scales"
        else:
            extract_function = xfeat.detectAndCompute
            third_arg = "scores"

        print(f"Extracting {args.top_k} features per image... This may take a while...")

        for i in tqdm(range(len(image_list))):
            image_name = image_list[i]
            img_path = get_image_fullpath(args.kapture_root, image_name)
            # print(f"\nExtracting features for {img_path}")
            img = Image.open(img_path)
            width_o, height_o = img.size

            img = transforms(img)

            # extract keypoints/descriptors for a single image
            output = extract_function(img.unsqueeze(0), top_k=args.top_k)[0]
            xys = output['keypoints'].cpu().numpy()    # tensor(top_k, 2) - (x, y)

            # convert to original image size
            xys[:, 0] = xys[:, 0] * width_o / 640
            xys[:, 1] = xys[:, 1] * height_o / 480

            desc = output['descriptors'].cpu().numpy() # tensor(top_k, 64)
            # TODO: see if there is a significant difference between the sparse/dense kp and if we need the third output
            third_output = output[third_arg] # can be either scales or scores depending on the function (dense/sparse)


            if keypoints_dtype is None or descriptors_dtype is None:
                keypoints_dtype = xys.dtype
                descriptors_dtype = desc.dtype

                keypoints_dsize = xys.shape[1]
                descriptors_dsize = desc.shape[1]

                kdata.keypoints[args.keypoints_type] = kapture.Keypoints('xfeat', keypoints_dtype, keypoints_dsize)
                kdata.descriptors[args.descriptors_type] = kapture.Descriptors('xfeat', descriptors_dtype,
                                                                               descriptors_dsize,
                                                                               args.keypoints_type, 'MNN')
                keypoints_config_absolute_path = get_feature_csv_fullpath(kapture.Keypoints,
                                                                          args.keypoints_type,
                                                                          args.kapture_root)
                descriptors_config_absolute_path = get_feature_csv_fullpath(kapture.Descriptors,
                                                                            args.descriptors_type,
                                                                            args.kapture_root)
                keypoints_to_file(keypoints_config_absolute_path, kdata.keypoints[args.keypoints_type])
                descriptors_to_file(descriptors_config_absolute_path, kdata.descriptors[args.descriptors_type])
            else:
                assert kdata.keypoints[args.keypoints_type].dtype == xys.dtype
                assert kdata.descriptors[args.descriptors_type].dtype == desc.dtype
                assert kdata.keypoints[args.keypoints_type].dsize == xys.shape[1]

                assert kdata.descriptors[args.descriptors_type].dsize == desc.shape[1]
                assert kdata.descriptors[args.descriptors_type].keypoints_type == args.keypoints_type
                assert kdata.descriptors[args.descriptors_type].metric_type == 'MNN'

            keypoints_fullpath = get_keypoints_fullpath(args.keypoints_type, args.kapture_root,
                                                        image_name, tar_handlers)
            # print(f"Saving {xys.shape[0]} keypoints to {keypoints_fullpath}")
            image_keypoints_to_file(keypoints_fullpath, xys)
            kdata.keypoints[args.keypoints_type].add(image_name)

            descriptors_fullpath = get_descriptors_fullpath(args.descriptors_type, args.kapture_root,
                                                            image_name, tar_handlers)
            # print(f"Saving {desc.shape[0]} descriptors to {descriptors_fullpath}")
            image_descriptors_to_file(descriptors_fullpath, desc)
            kdata.descriptors[args.descriptors_type].add(image_name)

        if not keypoints_check_dir(kdata.keypoints[args.keypoints_type], args.keypoints_type,
                                   args.kapture_root, tar_handlers) or \
                not descriptors_check_dir(kdata.descriptors[args.descriptors_type], args.descriptors_type,
                                          args.kapture_root, tar_handlers):
            print('local feature extraction ended successfully but not all files were saved')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        "Extract xFeat local features for all images in a dataset stored in the kapture format")
    parser.add_argument("--model", type=str, required=True, help='model path')
    parser.add_argument('--keypoints-type', default=None,  help='keypoint type_name, default is filename of model')
    parser.add_argument('--descriptors-type', default=None,  help='descriptors type_name, default is filename of model')

    parser.add_argument("--kapture-root", type=str, required=True, help='path to kapture root directory')


    parser.add_argument("--dense", action='store_true', help='use dense feature extraction')
    parser.add_argument("--top-k", type=int, default=20000, help='number of keypoints to extract')

    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help='use -1 for CPU')
    args = parser.parse_args()

    extract_kapture_keypoints(args)