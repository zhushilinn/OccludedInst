import json
import numpy as np
import copy
import cv2
def draw_region(img,rect_1,rect_2=None,fill_value=255):
    rect_1 = rect_1.tolist()
    for point in rect_1:
        point[0] = int(point[0] + 0.5)
        point[1] = int(point[1] + 0.5)
    cv2.fillPoly(img, np.array([rect_1]), fill_value)
    if rect_2 is not None:
        rect_2 = rect_2.tolist()
        for point in rect_2:
            point[0] = int(point[0] + 0.5)
            point[1] = int(point[1] + 0.5)
        cv2.fillPoly(img, np.array([rect_2]), fill_value)
    return img

def Iou(single_box_original, single_remain_box_original):
    target_box = [[single_box_original[0], single_box_original[1]], [single_box_original[2], single_box_original[3]], [single_box_original[4], single_box_original[5]], [single_box_original[6], single_box_original[7]]]
    remain_box = [[single_remain_box_original[0], single_remain_box_original[1]], [single_remain_box_original[2], single_remain_box_original[3]], [single_remain_box_original[4], single_remain_box_original[5]], [single_remain_box_original[6], single_remain_box_original[7]]]
    target_box = np.array(target_box)
    remain_box = np.array(remain_box)
    target_polygon = copy.deepcopy(target_box) 
    other_polygon = copy.deepcopy(remain_box)
    # 获取外接矩形的左上角点和右下角点。
    target_minx = target_polygon[:,0].min()
    target_miny = target_polygon[:,1].min()
    target_maxx = target_polygon[:,0].max()
    target_maxy = target_polygon[:,1].max()

    other_minx = other_polygon[:,0].min()
    other_miny = other_polygon[:,1].min()
    other_maxx = other_polygon[:,0].max()
    other_maxy = other_polygon[:,1].max()

    lefttop_point = [min(target_minx, other_minx), min(target_miny, other_miny)] 
    rightbottom_point = [max(target_maxx, other_maxx), max(target_maxy, other_maxy)]

    width = int(rightbottom_point[0]- lefttop_point[0] + 0.5)
    height = int(rightbottom_point[1] - lefttop_point[1] + 0.5)

    # 转换成相对坐标系
    deltax = lefttop_point[0]
    deltay = lefttop_point[1]
    for target_point in target_polygon:
        target_point[0] = target_point[0] - deltax
        target_point[1] = target_point[1] - deltay
    for other_point in other_polygon:
        other_point[0] = other_point[0] - deltax
        other_point[1] = other_point[1] - deltay

    # 画到帆布中，利用像素填充求和。
    img1 = np.zeros((height, width), np.uint8)
    img2 = np.zeros((height, width), np.uint8)
    img3 = np.zeros((height, width), np.uint8)
    out_img1 = draw_region(img1, target_polygon, fill_value=255)
    out_img2 = draw_region(img2, other_polygon, fill_value=255)
    out_img3 = draw_region(img3, target_polygon, other_polygon, fill_value=255)
    area_1 = np.sum(out_img1 == 255)
    area_2 = np.sum(out_img2 == 255)
    area_com = np.sum(out_img3 == 255)
    iou = (area_1 + area_2 - area_com) * 1.0 / area_com
    return iou


    

def interArea(single_box, single_remain_box):
    target_box = [[single_box[0], single_box[1]], [single_box[2], single_box[3]], [single_box[4], single_box[5]], [single_box[6], single_box[7]]]
    remain_box = [[single_remain_box[0], single_remain_box[1]], [single_remain_box[2], single_remain_box[3]], [single_remain_box[4], single_remain_box[5]], [single_remain_box[6], single_remain_box[7]]]
    ltpoint = target_box[0]
    rtpoint = target_box[1]
    rbpoint = target_box[2]
    lbpoint = target_box[3]
    for point in remain_box:
        a = (rtpoint[0]-ltpoint[0])*(point[1]-ltpoint[1])-(rtpoint[1]-ltpoint[1])*(point[0]-ltpoint[0])
        b = (rbpoint[0]-rtpoint[0])*(point[1]-rtpoint[1])-(rbpoint[1]-rtpoint[1])*(point[0]-rtpoint[0])
        c = (lbpoint[0]-rbpoint[0])*(point[1]-rbpoint[1])-(lbpoint[1]-rbpoint[1])*(point[0]-rbpoint[0])
        d = (ltpoint[0]-lbpoint[0])*(point[1]-lbpoint[1])-(ltpoint[1]-lbpoint[1])*(point[0]-lbpoint[0])
        #print(a,b,c,d)
        if (a>0 and b>0 and c>0 and d>0) or (a<0 and b<0 and c<0 and d<0):
            return True
        else:
            continue
    return False
def filterate(fig_box, images):
    images_need = []
    for i, single_fig in enumerate(fig_box):
        if single_fig == []:
            continue
        for single_box in single_fig:
            single_fig_remain = copy.deepcopy(single_fig)
            single_fig_remain.remove(single_box)
            for single_remain_box in single_fig_remain:
                res1 = interArea(single_box, single_remain_box)
                res2 = interArea(single_remain_box, single_box)
                if ((res1 == True) or (res2 == True)):
                    iou = Iou(single_box, single_remain_box)
                    if iou>0.10:
                        print(i)
                        images_need.append(images[i])
    return images_need
                        
                        
def deleteAnnotations(images_need, annotations):
    imageId = []
    annotation_res = []
    
    for single in images_need:
        imageId.append(single['id'])
    
    for single in annotations:
        singleId = single['image_id']
        if singleId in imageId:
            annotation_res.append(single)
    return annotation_res
            
        
    




def gather_fig(annotations_copy, images_number):
    box = [[]for i in range (images_number)]
    for single in annotations_copy:
        image_id  = single['image_id']
        bbox = single['bbox']
        # 对box解码 x,y,width,height
        x = bbox[0]
        y = bbox[1]
        width = bbox[2]
        height = bbox[3]
        bbox_need = [x, y, x+width, y, x+width, y+height, x, y+height]
        #print(bbox_need)
        box[image_id].append(bbox_need)
    return box

if __name__ == "__main__":
    json_path = "/home/zsl/label_of_city_and_bdd_occlusion/cityscapes/instancesonly_filtered_gtFine_train.json"
    with open(json_path, 'r') as linefile:
        json_data = json.load(linefile)
    images = json_data['images']
    category = json_data['categories']
    images_number = len(json_data['images'])
    annotations = json_data["annotations"]
    annotations_copy = copy.deepcopy(annotations)
    
    fig_box = gather_fig(annotations_copy, images_number)
    images_need = filterate(fig_box, images)
    images_need_new = []

    for item in images_need:
        if not item in images_need_new:
            images_need_new.append(item)
    annotations_need = deleteAnnotations(images_need_new, annotations_copy)

    json_data_res = {
        'categories':category,
        'images':images_need_new,
        'annotations':annotations_need,
    }
    print(len(images_need_new)) # 468 2717
    with open('/home/zsl/label_of_city_and_bdd_occlusion/cityscapes/instancesonly_filtered_gtFine_trainocc.json', 'w') as fp:
            json.dump(json_data_res, fp,indent=4)
    print("done")