import numpy as np
from termcolor import colored

from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense 

# Compare the two inputs
def comparator(learner, instructor):
    if len(learner) != len(instructor):
        raise AssertionError(f"The number of layers in the proposed model does not agree with the expected model: expected {len(instructor)}, got {len(learner)}.") 
    for a, b in zip(learner, instructor):
        if tuple(a) != tuple(b):
            print(colored("Test failed", attrs=['bold']),
                  "\n Expected value \n\n", colored(f"{b}", "green"), 
                  "\n\n does not match the input value: \n\n", 
                  colored(f"{a}", "red"))
            raise AssertionError("Error in test") 
    print(colored("All tests passed!", "green"))

# extracts the description of a given model
def summary(model):
    result = []
    def get_keras_shape(obj, layer_name):
        # 1. Extract raw shapes from tensors (handles Keras 3 lists/nodes safely)
        if isinstance(obj, list):
            if isinstance(obj[0], list) or isinstance(obj[0], tuple): 
                # Shared layer with multiple tensors (e.g., LSTM returning state)
                shapes = [tuple(t.shape) for t in obj[0]]
            else:
                # Normal list of tensors (e.g., Dense in a loop, or multiple inputs)
                shapes = [tuple(t.shape) for t in obj if hasattr(t, 'shape')]
        else:
            # Single isolated tensor
            shapes = [tuple(obj.shape)]
            
        # 2. Format to match Coursera's exact Keras 2 expectations
        if layer_name == 'InputLayer':
            return shapes # Grader expects a list containing a tuple: [(None, ...)]
        elif layer_name == 'LSTM':
            return shapes # Grader expects a list of 3 tuples: [(None, ...), (None, ...), (None, ...)]
        elif layer_name == 'GRU':
            return shapes    
        else:
            return shapes[0] # Grader expects just a single tuple: (None, ...)
    # ----------------------------------------------
    
    for layer in model.layers:
        layer_name = layer.__class__.__name__
        
        # Safely get the output shape without triggering AttributeError
        out_shape = get_keras_shape(layer.output, layer_name)
        
        descriptors = [layer_name, out_shape, layer.count_params()]
        if (type(layer) == Dense):
            descriptors.append(layer.activation.__name__.replace("_v2", ""))
        if (type(layer) == GRU):
            descriptors.append(f"return_sequences={layer.return_sequences}")
            descriptors.append(f"return_state={layer.return_state}")
        result.append(descriptors)
    return result