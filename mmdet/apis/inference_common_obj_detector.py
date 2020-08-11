import torch
import numpy as np
import mmcv 
import cv2
import os
from mmcv.image import imwrite
from mmcv.visualization import imshow, color_val
from mmcv.parallel import collate, scatter
from mmdet.apis import init_detector, inference_detector
from mmdet.apis.inference import LoadImage, show_result
from mmdet.datasets.pipelines import Compose

def inference_common_obj_detector(model, imgs):
    """Inference image(s) with the detector.
    We do the same as inference_detecttor, except we can accepts a list of images 

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    img_infos=[test_pipeline(dict(img=img)) for img in imgs] 
    data = dict(img=[img_info['img'] for img_info in img_infos],
                img_meta=[img_info['img_meta'] for img_info in img_infos])

    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)
    return results

def show_codet_results(codet_result,
                        obj_score_thr=0.1,
                        matching_score_thr=0.8, 
                        wait_time=0,
                        classes_name=None,
                        show=True, 
                        out_dir=None,
                        center_color='blue',
                        text_color='red',
                        thickness=1,
                        font_scale=0.5,):
    """Visualize the common object detection results on the image.

    Args:
        codet_result: dict(img_metas=(img_meta_img0,img_meta_img1),
                            boxes=(box_img0,box_img1), where box_img0=[id:bbox_detect]
                            pairs=[(id0,id1,matching_score)]
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    # Draw Boxes for each image 
    img_pair=[]
    img_shape=[]
    filename=[]
    center_color = color_val(center_color)
    text_color = color_val(text_color)
    for i in range(2):
        img_meta= codet_result['img_metas'][i]
        boxes = codet_result['boxes'][i]
        det_id = [f'{k}' for k in boxes.keys()]
        det_bboxes = [v for k,v in boxes.items()]
        img =show_result(img_meta['filename'],det_bboxes, class_names=det_id,score_thr=obj_score_thr,show=False)
        # Show class name at center:
        labels =codet_result['labels'][i]
        if classes_name:            
            for k,bbox in boxes.items():
                label_text = classes_name[labels[k]]
                cv2.putText(img, label_text, (int(bbox[0][0]), int(bbox[0][3]) - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        img_pair.append(img)
        img_shape.append(img_meta['ori_shape'][0:2]) 
        filename.append(os.path.basename(img_meta['filename']).split('.')[0])
    
    # Horizontaly Stack two images
    h0,w0=img_shape[0]
    h1,w1=img_shape[1]
    ratio=h0/h1
    
    if ratio>1:
        #Scale to Imag1
        w1 = int(ratio*w1)
        img_pair[1] = mmcv.imresize(img_pair[1],(w1,h0))
    elif ratio<1:
        #Scale to Img2
        w0 = int(w0/ratio)
        img_pair[0] = mmcv.imresize(img_pair[0],(w0,h1))
    img_pair = np.concatenate(img_pair, axis=1)

    # Draw centerline 
    for pair in codet_result['pairs']:
        id0,id1,matching_score = pair
        if matching_score > matching_score_thr:
            #Get box center
            box0=codet_result['boxes'][0][id0][0]
            box1=codet_result['boxes'][1][id1][0]
            if ratio>=1:
                c0=(int(0.5*(box0[0] + box0[2])), int(0.5*(box0[1] + box0[3]))) 
                c1=(int(0.5*ratio*(box1[0] + box1[2]))+w0, int(0.5*ratio*(box1[1] + box1[3]))) 
            else:
                c0=(int(0.5/ratio*(box0[0] + box0[2])), int(0.5/ratio*(box0[1] + box0[3]))) 
                c1=(int(0.5*(box1[0] + box1[2]))+w0, int(0.5*(box1[1] + box1[3]))) 
            # Draw lines
            label_text = '{:.02f}'.format(matching_score)
            cv2.line(img_pair, c0, c1, center_color, thickness)
            cv2.putText(img_pair,label_text, (int((c0[0]+c1[0])*0.5),int((c0[1]+c1[1])*0.5)),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color) 

    win_name=f'{filename[0]}_{filename[1]}'
    if show:
        imshow(img_pair, win_name, wait_time)
    if out_dir is not None:
        imwrite(img_pair, out_dir+f'/{win_name}.jpg')
    if not (show or out_dir):
        return img_pair
