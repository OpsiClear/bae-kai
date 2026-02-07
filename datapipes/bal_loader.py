"""
This file contains the pipeline for the Bundle Adjustment in the Large dataset.

The dataset is from the following paper:
Sameer Agarwal, Noah Snavely, Steven M. Seitz, and Richard Szeliski.
Bundle adjustment in the large.
In European Conference on Computer Vision (ECCV), 2010.

Link to the dataset: https://grail.cs.washington.edu/projects/bal/
"""

import torch
import os
import bz2
import re
import urllib.request
from functools import partial
import pypose as pp

DTYPE = torch.float64

# only export __all__
__ALL__ = ['build_pipeline', 'read_bal_data', 'DATA_URL', 'ALL_DATASETS', 'get_problem']

# base url for the BAL dataset, used to download the problem files
DATA_URL = 'https://grail.cs.washington.edu/projects/bal/'

# all dataset names in the BAL dataset, used to check if the dataset name is valid
ALL_DATASETS = ['ladybug', 'trafalgar', 'dubrovnik', 'venice', 'final']


def _get_problem_urls(dataset, cache_dir):
    """Get list of problem URLs from the dataset HTML page."""
    os.makedirs(cache_dir, exist_ok=True)

    html_url = DATA_URL + dataset + '.html'
    cache_file = os.path.join(cache_dir, dataset + '.html')

    # Download HTML if not cached
    if not os.path.exists(cache_file):
        urllib.request.urlretrieve(html_url, cache_file)

    # Parse HTML to find .bz2 links using regex (avoids bs4 dependency)
    with open(cache_file, 'r') as f:
        html = f.read()

    # Find all href attributes ending with .bz2
    pattern = r'href=["\']([^"\']*\.bz2)["\']'
    matches = re.findall(pattern, html)

    # Build full URLs and sort by number of images
    urls = [DATA_URL + m for m in matches]
    urls.sort(key=lambda x: int(os.path.basename(x).split('-')[1]))

    return urls


def _download_and_decompress(url, cache_dir):
    """Download and decompress a .bz2 file, return path to decompressed file."""
    os.makedirs(cache_dir, exist_ok=True)

    filename = os.path.basename(url)
    compressed_path = os.path.join(cache_dir, filename)
    decompressed_path = compressed_path.replace('.bz2', '')

    # Return cached decompressed file if exists
    if os.path.exists(decompressed_path):
        return decompressed_path

    # Download if not cached
    if not os.path.exists(compressed_path):
        urllib.request.urlretrieve(url, compressed_path)

    # Decompress
    with bz2.open(compressed_path, 'rb') as f_in:
        with open(decompressed_path, 'wb') as f_out:
            f_out.write(f_in.read())

    return decompressed_path


def read_bal_data(file_name: str, use_quat=False) -> dict:
    """
    Read a Bundle Adjustment in the Large dataset.

    Referenced Scipy's BAL loader: https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

    According to BAL official documentation, each problem is provided as a text file in the following format:

    <num_cameras> <num_points> <num_observations>
    <camera_index_1> <point_index_1> <x_1> <y_1>
    ...
    <camera_index_num_observations> <point_index_num_observations> <x_num_observations> <y_num_observations>
    <camera_1>
    ...
    <camera_num_cameras>
    <point_1>
    ...
    <point_num_points>

    Where, there camera and point indices start from 0. Each camera is a set of 9 parameters - R,t,f,k1 and k2. The rotation R is specified as a Rodrigues' vector.

    Parameters
    ----------
    file_name : str
        The decompressed file of the dataset.

    Returns
    -------
    dict
        A dictionary containing the following fields:
        - problem_name: str
            The name of the problem.
        - camera_params: torch.Tensor (n_cameras, 9 or 10)
            contains camera parameters for each camera. If use_quat is True, the shape is (n_cameras, 10).
        - points_3d: torch.Tensor (n_points, 3)
            contains initial estimates of point coordinates in the world frame.
        - points_2d: torch.Tensor (n_observations, 2)
            contains measured 2-D coordinates of points projected on images in each observations.
        - camera_index_of_observations: torch.Tensor (n_observations,)
            contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
        - point_index_of_observations: torch.Tensor (n_observations,)
            contains indices of points (from 0 to n_points - 1) involved in each observation.
    """
    with open(file_name, "r") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())

        camera_indices = torch.empty(n_observations, dtype=torch.int64)
        point_indices = torch.empty(n_observations, dtype=torch.int64)
        points_2d = torch.empty((n_observations, 2), dtype=DTYPE)

        for i in range(n_observations):
            tmp_line = file.readline()
            camera_index, point_index, x, y = tmp_line.split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i, 0] = float(x)
            points_2d[i, 1] = float(y)

        camera_params = torch.empty(n_cameras * 9, dtype=DTYPE)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        points_3d = torch.empty(n_points * 3, dtype=DTYPE)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))

    if use_quat:
        # convert Rodrigues vector to unit quaternion for camera rotation
        r = pp.so3(camera_params[:, :3])
        q = r.Exp()
        camera_params = torch.cat([camera_params[:, 3:6], q, camera_params[:, 6:]], axis=1)
    else:
        camera_params = torch.cat([camera_params[:, 3:6], camera_params[:, :3], camera_params[:, 6:]], axis=1)

    # convert camera_params to torch.Tensor
    camera_params = camera_params.clone().detach().to(DTYPE)

    return {'problem_name': os.path.splitext(os.path.basename(file_name))[0],
            'camera_params': camera_params,
            'points_3d': points_3d,
            'points_2d': points_2d,
            'camera_index_of_observations': camera_indices,
            'point_index_of_observations': point_indices,
            }


class BALDataset:
    """Iterator over BAL dataset problems."""

    def __init__(self, dataset='ladybug', cache_dir='bal_data', use_quat=False):
        self.urls = _get_problem_urls(dataset, cache_dir)
        self.cache_dir = cache_dir
        self.use_quat = use_quat
        self._index = 0

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self.urls):
            raise StopIteration
        url = self.urls[self._index]
        self._index += 1
        file_path = _download_and_decompress(url, self.cache_dir)
        return read_bal_data(file_path, use_quat=self.use_quat)

    def __len__(self):
        return len(self.urls)


def build_pipeline(dataset='ladybug', cache_dir='bal_data', use_quat=False):
    """
    Build a pipeline for the Bundle Adjustment in the Large dataset.

    Parameters
    ----------
    dataset : str, optional
        The name of the dataset, by default 'ladybug'.
        Must be one of ['ladybug', 'trafalgar', 'dubrovnik', 'venice', 'final'].
    cache_dir : str, optional
        The directory to cache the downloaded files, by default 'bal_data'.
    use_quat : bool, optional
        Whether to use quaternion for rotation, by default False.

    Returns
    -------
    BALDataset
        An iterable over the dataset problems.
    """
    print(f"Streaming data for {dataset}...")
    assert dataset in ALL_DATASETS, f"dataset_name must be one of {ALL_DATASETS}"
    return BALDataset(dataset=dataset, cache_dir=cache_dir, use_quat=use_quat)


def get_problem(problem_name, dataset, cache_dir='bal_data', use_quat=False):
    """Get a specific problem from the BAL dataset."""
    print(f"Streaming data for {dataset}...")
    assert dataset in ALL_DATASETS, f"dataset_name must be one of {ALL_DATASETS}"

    urls = _get_problem_urls(dataset, cache_dir)

    # Find matching URL
    for url in urls:
        basename = os.path.basename(url)
        if basename in {problem_name + '.txt.bz2', problem_name + '.bz2'}:
            file_path = _download_and_decompress(url, cache_dir)
            return read_bal_data(file_path, use_quat=use_quat)

    raise ValueError(f"Problem {problem_name} not found in dataset {dataset}.")


def _test():
    dp = build_pipeline()
    print("Testing dataset pipeline with use_quat=False...")
    for i in dp:
        point_indices = i['point_index_of_observations']
        camera_indices = i['camera_index_of_observations']
        points = i['points_3d'][point_indices]
        pixels = i['points_2d']
        camera_params = i['camera_params'][camera_indices]
        problem_name = i['problem_name']
        # check shape as in pp.reprojerr
        assert points.size(-1) == 3 and pixels.size(-1) == 2 and camera_params.size(-1) == 9, "Shape not compatible."
        # check shape at index 0, should be n_observation
        assert points.size(0) == pixels.size(0) == camera_params.size(0), "Shape not compatible."
        # check dtype is float64
        assert DTYPE == points.dtype == pixels.dtype == camera_params.dtype, "dtype not float64."
        print(problem_name, 'ok')

    for dataset in ALL_DATASETS:
        dp = build_pipeline(dataset=dataset, use_quat=True)
        print("Testing dataset pipeline with use_quat=True...")
        for i in dp:
            camera_params = i['camera_params']
            assert camera_params.size(-1) == 10, "Shape not compatible."
            assert DTYPE == camera_params.dtype, "dtype not float64."
            print(i['problem_name'], 'ok')
        print("All tests passed!")


if __name__ == '__main__':
    _test()
