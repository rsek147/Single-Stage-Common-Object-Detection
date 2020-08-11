import argparse
import os
import glob
from mmdet.apis import init_detector
from mmdet.apis.inference_common_obj_detector import inference_common_obj_detector, show_codet_results
from mmdet.datasets import CocoDataset, VOCDataset

def parse_args():
    parser = argparse.ArgumentParser(description='test detector')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--img_file',nargs="+", default=[], type=str, help='Image path to infering')
    parser.add_argument('--img_folder', help='folder of test images')
    parser.add_argument('--classes_name', default='coco', help='class name to show for debug')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.img_folder is None:
        print("\nImage folder does not exist...")
        return 0
    imgs = sorted(glob.glob(f'{args.img_folder}/*.jpg') \
                + glob.glob(f'{args.img_folder}/*.jpeg') 
                + glob.glob(f'{args.img_folder}/*.png') )
    model = init_detector(args.config, args.checkpoint, device='cuda')
    imgs = imgs[:50]
    print("Start infer model !!!\n", 'num_imgs: ',len(imgs))
    results = inference_common_obj_detector(model,imgs)

    # Plot the codetion results:
    if args.classes_name.lower()=='coco':
        classes_name= CocoDataset.CLASSES
    elif args.classes_name.lower()=='voc':
        classes_name= VOCDataset.CLASSES
    else:
        classes_name= args.classes_name
    for result in results:
        # if len(result['pairs'])>0 and len(result['boxes'][0]) >0 and len(result['boxes'][1]) >0:
        show_codet_results(result,
                            obj_score_thr=0.5,
                            matching_score_thr=0.5, 
                            classes_name= classes_name,
                            wait_time=0,
                            show=False, 
                            out_dir="./cache")
        
if __name__ == '__main__':
    main()