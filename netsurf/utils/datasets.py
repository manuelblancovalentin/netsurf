""" Basic modules """
from time import time

""" Requests for downloading files """
import requests

""" tqdm for progress """
from tqdm import tqdm

""" Import numpy """
import numpy as np

# Custom utils 
from . import log

# manuelbv
def download_file(url, local_filename):
    try:
        # Send a GET request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful

        # Get final size 
        total_size = int(response.headers.get('content-length', 0))

        # Open a local file in write-binary mode
        with open(local_filename, 'wb') as f:
            # Write the content of the response to the local file
            for chunk in tqdm(response.iter_content(chunk_size=8192), total = total_size/8192, unit = 'KB'):
                f.write(chunk)

        log._info(f"Downloaded {local_filename} from {url}")
    except requests.exceptions.RequestException as e:
        log._info(f"Error downloading {url}: {e}")

""" Prepare food """
def prepare_data(bmk, subset = 'train', nsample_mod = -1):
    # Get dataset
    bmk.assert_dataset_is_loaded()
    dset = bmk.dataset.dataset

    # First, make sure subset exists in dset 
    if subset not in dset:
        raise ValueError(f'Subset {subset} not found in dataset')
    
    # Get subset 
    data = dset[subset]

    # Check what kind of data this is
    if isinstance(data, tuple):
        # Apply processing
        XTrain, YTrain = data
        # Apply pre-processing to data, if needed 
        if hasattr(bmk.model, 'preprocess_input'):
            s0 = time()
            log._info(f'Preprocessing input data...', end = '')
            XTrain = bmk.model.preprocess_input(XTrain)
            print(f'done in {time() - s0:.2f} seconds')

        if hasattr(bmk.model, 'preprocess_output'):
            s0 = time.time()
            log._info(f'Preprocessing output data...', end = '')
            YTrain = bmk.model.preprocess_output(YTrain)
            print(f'done in {time() - s0:.2f} seconds')
        
        # Make sure data is a numpy array
        if not isinstance(XTrain, np.ndarray):
            XTrain = np.array(XTrain)
        if not isinstance(YTrain, np.ndarray):
            YTrain = np.array(YTrain)

        if nsample_mod > 0:
            XTrain = XTrain[:-(XTrain.shape[0] % nsample_mod)]
            YTrain = YTrain[:-(YTrain.shape[0] % nsample_mod)]

        # Set data back
        data = (XTrain, YTrain)

    if not isinstance(data, tuple):
        data = (data,)

    # Return data 
    return data