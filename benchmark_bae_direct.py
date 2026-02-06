"""Direct BAE benchmark on BAL datasets."""
import sys
import warnings
warnings.filterwarnings('ignore')

from time import perf_counter
import torch
import pypose as pp

from ba_helpers import Reproj, least_square_error
from datapipes.bal_loader import get_problem
from bae.optim import LM
from bae.utils.pysolvers import PCG

def benchmark(dataset_name, problem_name, n_iterations=100):
    print(f"\n{'='*70}")
    print(f"BAE Benchmark: {dataset_name}/{problem_name}")
    print(f"{'='*70}")
    
    DEVICE = 'cuda'
    dataset = get_problem(problem_name, dataset_name, use_quat=True)
    
    n_cams = dataset['camera_params'].shape[0]
    n_pts = dataset['points_3d'].shape[0]
    n_obs = dataset['points_2d'].shape[0]
    print(f"Cameras: {n_cams}, Points: {n_pts}, Observations: {n_obs}")
    
    trimmed = {k: v.to(DEVICE) for k, v in dataset.items() if type(v) == torch.Tensor}
    
    input = {
        'points_2d': trimmed['points_2d'],
        'camera_indices': trimmed['camera_index_of_observations'],
        'point_indices': trimmed['point_index_of_observations']
    }
    
    model = Reproj(
        trimmed['camera_params'][:, :10].clone(),
        trimmed['points_3d'].clone()
    ).to(DEVICE)
    
    solver = PCG(tol=1e-4, maxiter=50)
    strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
    optimizer = LM(model, strategy=strategy, solver=solver, reject=10)
    
    initial_chi2 = optimizer.model.loss(input, None).item() * 2
    print(f"Initial Chi2: {initial_chi2:.2f}")
    
    # Warmup
    torch.cuda.synchronize()
    
    start = perf_counter()
    for i in range(n_iterations):
        loss = optimizer.step(input)
        if i % 20 == 0:
            torch.cuda.synchronize()
            print(f"  Iter {i}: Chi2 = {loss.item()*2:.2f}")
    
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    
    final_chi2 = loss.item() * 2
    print(f"\nFinal Chi2: {final_chi2:.2f}")
    print(f"Time: {elapsed:.4f}s ({n_iterations} iterations)")
    print(f"Per iteration: {elapsed/n_iterations*1000:.2f} ms")
    
    return elapsed / n_iterations * 1000

if __name__ == '__main__':
    # Benchmark ladybug-49
    ms1 = benchmark('ladybug', 'problem-49-7776-pre', n_iterations=100)
    
    # Benchmark ladybug-138  
    ms2 = benchmark('ladybug', 'problem-138-19878-pre', n_iterations=100)
    
    print(f"\n{'='*70}")
    print("BAE SUMMARY")
    print(f"{'='*70}")
    print(f"  ladybug-49:  {ms1:.2f} ms/iter")
    print(f"  ladybug-138: {ms2:.2f} ms/iter")
