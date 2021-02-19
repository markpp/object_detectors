import sys, os
import numpy as np
import json, pickle

import coremltools
import onnx_coreml
from coremltools.proto import NeuralNetwork_pb2, FeatureTypes_pb2

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# from utils.post_process import _topk, hmap__bboxes, tennis_hmap__bboxes

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def load_onnx_model( fn_onnx ):
    import onnx
    
    # Load the ONNX GraphProto object. Graph is a standard Python protobuf object
    onnx_model = onnx.load(fn_onnx)
    
    onnx.checker.check_model(onnx_model)
    return onnx_model

    
if __name__ == '__main__':

    onnx_model = load_onnx_model('trained_models/onnx/model.onnx')

    image_size = 512

    use_img_input=True
    fn_mlmodel_path = 'trained_models/onnx/model.mlmodel'
    if 1:
        convert_params = dict(
            predicted_feature_name = [],
            minimum_ios_deployment_target='13',
        )
        if use_img_input:
            convert_params.update( dict(
                image_input_names =  ['input.1' ],
                preprocessing_args = {
                    'image_scale': 1/255.0,
                },
            ) )
        
        mlmodel = onnx_coreml.convert(
            onnx_model,
            **convert_params,
        )
        #print(dir(mlmodel))
        spec = mlmodel.get_spec()
        # print(spec.description)        

        # https://machinethink.net/blog/coreml-image-mlmultiarray/
        if mlmodel != None:        
            input = spec.description.input[0]
            input.type.imageType.colorSpace = FeatureTypes_pb2.ImageFeatureType.RGB
            input.type.imageType.height = image_size
            input.type.imageType.width = image_size

            # print(spec.description)

            mlmodel = coremltools.models.MLModel(spec)

        if mlmodel != None:
            mlmodel.save( fn_mlmodel_path )


    '''
    model.author = 'mp'
    model.short_description = 'This model does 9 mushroom classes'
    #model.input_description['image'] = 'Image mushroom' No feature with name image.
    model.output_description['classLabelProbs'] = 'Confidence and label of predicted mushroom.'
    model.output_description['classLabel'] = 'Labels of predictions'
    # Save the CoreML model
    model.save('{}.mlmodel'.format(model_name))
    '''
'''
USAGE:
python scripts/convert_coreml.py 'resnet' 'hc64'
python scripts/convert_coreml.py 'mobilenetv2' 'hc64'

'''


