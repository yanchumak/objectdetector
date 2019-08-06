
import numpy as np
from keras_retinanet.models import load_model
from keras_retinanet.models import convert_model
from keras_retinanet.models import check_training_model
from keras import backend as K
import tensorflow as tf
import sys

if (len(sys.argv) != 3):
    print("Wrong arguments\nUsage: \n    first argument: keras model file path\n    second argument: tensorflow model output folder path")
    exit(1)

keras_model_path = sys.argv[1]
output_folder_path = sys.argv[2]

model = load_model(keras_model_path, backbone_name='resnet50')
check_training_model(model)
model = convert_model(model)

def build_signature(model):
    inputs = {}
    for input in model.inputs: 
        inputs[input.name] = input
    outputs = {}
    for output in model.outputs: 
        outputs[output.name] = output

    return tf.saved_model.signature_def_utils.predict_signature_def(inputs=inputs, outputs=outputs)
	
builder = tf.saved_model.builder.SavedModelBuilder(output_folder_path)
builder.add_meta_graph_and_variables(
    sess=K.get_session(),                                      
    tags=[tf.saved_model.tag_constants.SERVING], signature_def_map={                                      
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:                           
            build_signature(model)
    })                                      
builder.save()
