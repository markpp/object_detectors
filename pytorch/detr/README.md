https://github.com/L4zyy/DETR_TPU


- For each image, you need to have a target dictionnary with the keys "labels" (which in your case will be a vector of 1s) and "boxes" which contains as many boxes as you have elements in the label vector, with format (center_x, center_y, h, w), normalized by the image size.

- num_classes should be equal to the maximal class label + 1. So if your labels are 0 and 1, you need to use num_classes = 2
- The +1 here correspond to the no-object class (DETR reserves the last class id for it)
- target_classes_o are the ground truth labels for the existing objects in the image, in order defined by the hungarian matcher
- Since DETR predicts always 100 objects, if there are n "true" objects in the image (ie len(target_classes_o) = n, we must supervise the 100 - n remaining outputs to predict "no-object". See that target_classes is initialized with 100 "no-objects" labels, and then we fill the "true" objects with their correct label coming from target_classes_o
- On average, the number of true objects on the image is much smaller than the total number of predicted objects (100), so the "no-object" class is the majority. empty_weight is used to rebalance the classification loss and diminish the relative weight of the "no-object" class.
- Since DETR also predicts the "no-object" class, you'll effectively have 3 classes, thus you need to keep the cross_entropy as is.
