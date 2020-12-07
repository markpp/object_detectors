import numpy as np
import os
import cv2

def mask2gt(mask):
    # instances are encoded as different colors
    obj_ids = np.unique(mask)

    # first id is the background, so remove it
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set
    # of binary masks
    masks = mask == obj_ids[:, None, None]

    boxes = []
    # if no objects exists
    if len(obj_ids) == 0:
        return boxes

    for i, obj in enumerate(obj_ids):
        pos = np.where(masks[i])
        xmin, xmax = np.min(pos[1]), np.max(pos[1])
        ymin, ymax = np.min(pos[0]), np.max(pos[0])
        boxes.append([float(xmin)/mask.shape[1], float(ymin)/mask.shape[0],
                      float(xmax)/mask.shape[1], float(ymax)/mask.shape[0]])
    return boxes

def label_unique(mask):
    new_mask = np.zeros(mask.shape)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i, cont in enumerate(contours):
        cv2.drawContours(new_mask, [cont], contourIdx=-1, color=(i+1), thickness=-1)
    return new_mask

# crops the necessary number of patches at the desired resolution,
# WITH overlapping regions, in order to cover the entire image
#def crop_overlapping_patches():

# crops the necessary number of patches at the desired resolution,
# WITHOUT overlapping regions, in order to cover the entire image
def crop_patches(img, crop_size, root_dir, item, img_file, show=False):
    img_gt_bb, img_gt_cls = [], []

    mask_file = img_file.replace('jpg','png')
    mask = cv2.imread(os.path.join(root_dir,item,"masks",mask_file), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
        print("empty mask created in place of missing file {}".format(mask_file))
    else:
        mask = cv2.copyMakeBorder(mask, top=0, bottom=crop_size, left=0, right=crop_size, borderType=cv2.BORDER_CONSTANT, value=0)

    #tmp = img.copy()
    #tmp[np.where(mask>0)] = (255,0,0)
    #cv2.imwrite("test.jpg",img)

    new_mask  = label_unique(mask)

    img_gts = mask2gt(new_mask)
    for img_gt in img_gts:
        img_gt_bb.append(img_gt) #xmin, ymin, xmax, ymax
        img_gt_cls.append(0)

    patches, patch_offsets = [], []
    # divide image into appropritate crops and detect defects
    for x in range(0,img.shape[1]-crop_size,crop_size):
        for y in range(0,img.shape[0]-crop_size,crop_size):
            crop = img[y:y+crop_size, x:x+crop_size]
            patches.append(np.rollaxis(crop, 2, 0) / 255)
            patch_offsets.append([x,y])

    if show:
        output = np.zeros(img.shape, np.uint8)
        visualize_patches(output, patches, patch_offsets)

    return patches, patch_offsets, img_gt_bb, img_gt_cls

def visualize_patches(output, patches, patch_offsets, thickness=8):
    for patch, offset in zip(patches,patch_offsets):
        x, y = offset
        patch = np.array(np.rollaxis(patch, 0, 3)*255.0, dtype=np.int8)
        patch = cv2.rectangle(patch, (0+thickness//2, 0+thickness//2), (patch.shape[1]-thickness//2-1, patch.shape[0]-thickness//2-1), (0,255,0), thickness)
        output[y:y+patch.shape[0],x:x+patch.shape[1],:] = patch
    cv2.imshow("output",output)
    cv2.imwrite("output.jpg",output)
    cv2.waitKey()

if __name__ == "__main__":

    crop_size = 416
    batch_size = ((3300-crop_size)//crop_size)*(6050//crop_size)
    root_dir = '/home/markpp/datasets/bo/test/'
    #items = ['item00', 'item01', 'item02', 'item03', 'item04', 'item07', 'item08']
    items = sorted([f for f in os.listdir(root_dir) if 'item' in f])


    # iterate over each item
    for item in items[:1]:
        img_paths = sorted([f for f in os.listdir(os.path.join(root_dir,item,"rgb")) if f.endswith('.jpg')])
        print("{} has {} images".format(item,len(img_paths)))
        # iterate over each image belonging to the given item
        if not os.path.exists(os.path.join(root_dir,item,"results")):
            os.makedirs(os.path.join(root_dir,item,"results"))

        for i,img_file in enumerate(img_paths[-1:]):
            img = cv2.imread(os.path.join(root_dir,item,"rgb",img_file))
            img = cv2.copyMakeBorder(img, top=0, bottom=crop_size, left=0, right=crop_size, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                print("image {} not loaded".format(img_file))
                break

            patches, patch_offsets, img_gt_bb, img_gt_cls = crop_patches(img, crop_size, root_dir, item, img_file, show=True)
