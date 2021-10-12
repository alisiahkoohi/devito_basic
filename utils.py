import h5py
import os

MODEL_NAME = 'overthrust_model.h5'
MODEL_PATH = 'model/'


def LoadOverthrustModel(model_path=MODEL_PATH):
    """
    Loads Overthrust model.

    Args:
        model_path: String containing (relative) path to the directory
            containing the Overthrust model, named 'Overthrust_model.h5'.

    Returns:
        m: A 2D float32 numpy array containing squared slowness Overthrust
            model in (s^2 / km^2).
    """
    # Download model if does not exist.
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = os.path.join(model_path, MODEL_NAME)
    if not os.path.exists(model_path):
        os.system("wget https://github.com/slimgroup/JUDI.jl/raw/master/data/"
                  "overthrust_model.h5 -O " +  model_path)

    # Load the squared-slowness model.
    m = h5py.File(model_path, 'r')['m'][...].T
    return m
