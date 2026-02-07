import os
import torch
import numpy as np
import pypose as pp
import torch.utils.data as Data
import urllib.request
import zipfile
import tempfile


def _download_and_extract(url, root):
    """Download and extract a zip file."""
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    filepath = os.path.join(root, filename)
    extract_dir = os.path.join(root, filename.replace('.zip', ''))

    # Skip if already extracted
    if os.path.exists(extract_dir):
        return

    # Download
    if not os.path.exists(filepath):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, filepath)

    # Extract
    print(f"Extracting {filepath}...")
    with zipfile.ZipFile(filepath, 'r') as zf:
        zf.extractall(root)


class G2OPGO(Data.Dataset):
    '''
    This data is from the following paper:
    L. Carlone, R. Tron, K. Daniilidis, and F. Dellaert. Initialization Techniques for 3D
    SLAM: a Survey on Rotation Estimation and its Use in Pose Graph Optimization. In IEEE
    Intl. Conf. on Robotics and Automation (ICRA), pages 4597-4604, 2015.
    '''
    link = 'https://github.com/pypose/pypose/releases/download/v0.4.0/parking-garage.zip'
    def __init__(self, root, dataname, device='cpu', download=True):
        super().__init__()

        if download:
            _download_and_extract(self.link, root)

        def info2mat(info):
            mat = np.zeros((6,6))
            ix = 0
            for i in range(mat.shape[0]):
                mat[i,i:] = info[ix:ix+(6-i)]
                mat[i:,i] = info[ix:ix+(6-i)]
                ix += (6-i)
            return mat

        self.dtype = torch.get_default_dtype()
        filename = os.path.join(root, dataname)
        ids, nodes, edges, poses, infos = [], [], [], [], []
        with open(filename) as f:
            for line in f:
                line = line.split()
                if line[0] == 'VERTEX_SE3:QUAT':
                    ids.append(torch.tensor(int(line[1]), dtype=torch.int64))
                    nodes.append(pp.SE3(np.array(line[2:], dtype=np.float64)))
                elif line[0] == 'EDGE_SE3:QUAT':
                    edges.append(torch.tensor(np.array(line[1:3], dtype=np.int64)))
                    poses.append(pp.SE3(np.array(line[3:10], dtype=np.float64)))
                    infos.append(torch.tensor(info2mat(np.array(line[10:], dtype=np.float64))))

        self.ids = torch.stack(ids)
        self.nodes = torch.stack(nodes).to(self.dtype).to(device)
        self.edges = torch.stack(edges).to(device) # have to be LongTensor
        self.poses = torch.stack(poses).to(self.dtype).to(device)
        self.infos = torch.stack(infos).to(self.dtype).to(device)
        assert self.ids.size(0) == self.nodes.size(0) \
               and self.edges.size(0) == self.poses.size(0) == self.infos.size(0)

    def init_value(self):
        return self.nodes.clone()

    def __getitem__(self, i):
        return self.edges[i], self.poses[i], self.infos[i]

    def __len__(self):
        return self.edges.size(0)
