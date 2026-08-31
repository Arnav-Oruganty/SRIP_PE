# %%
import numpy as np
from scipy.linalg import sqrtm, inv, dft
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import pickle

# %%
import os
import sys

if not os.path.exists('LA-MCTS'):
    print("Cloning LA-MCTS repository...")
    !git clone https://github.com/facebookresearch/LA-MCTS.git
else:
    print("LA-MCTS repository already exists.")

%cd LA-MCTS

print("\nInstalling LA-MCTS and its dependencies...")
!pip install -q -e .

if '/content/LA-MCTS' not in sys.path:
    sys.path.append('/content/LA-MCTS')

%cd /content/

print("\n Installation complete. You can now import and use the 'lamcts' library.")


# this cell is to patch the nevergrad_sampler.py file to use np.all instead of np.alltrue (np.alltrue -> np.all)
nevergrad_path = "LA-MCTS/lamcts/sampler/nevergrad_sampler.py"
if os.path.exists(nevergrad_path):
    with open(nevergrad_path, "r") as f:
        code = f.read()
    code = code.replace("np.alltrue", "np.all")
    with open(nevergrad_path, "w") as f:
        f.write(code)
    print("Patched nevergrad_sampler.py to use np.all instead of np.alltrue.")
else:
    print("nevergrad_sampler.py not found, patch not applied.")

# %%
import numpy as np
import torch
import random
import math
import time
import logging
import pickle
from typing import Tuple, Optional, Dict
from scipy.linalg import sqrtm, inv, dft

from lamcts import MCTS, Func, StatsFuncWrapper, ObjectFactory
from lamcts.config import SamplerEnum, ClassifierEnum, get_mcts_params, GreedyType
from lamcts.utils import set_log_level

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
def setup_ris_system(K=4, M=20, L=16, seed=42, B=None):
    """
    Sets up the RIS system parameters, channels, and fixed matrices.
    Assumes NO direct UE-BS path.
    """
    np.random.seed(seed)

    # --- System Parameters ---
    c_ue = 0.2
    c_ris = 0.4
    c_bs = 0.6
    b_min = 0.2
    alpha_ris = 2.0
    delta_ris = 0.43 * np.pi
    Pk = 1.0

    if B is None:
        B = M
    tau = K

    # --- Generate Correlation Matrices ---
    psi_bs = np.array([[c_bs ** abs(i - j) for j in range(L)] for i in range(L)], dtype=float)
    psi_ue = np.array([[c_ue ** abs(i - j) for j in range(K)] for i in range(K)], dtype=float)
    psi_irs = np.array([[c_ris ** abs(i - j) for j in range(M)] for i in range(M)], dtype=float)

    # --- Generate Base Channels ---
    H_rbar = (1 / np.sqrt(2)) * (np.random.randn(M, K) + 1j * np.random.randn(M, K))
    G_bar = (1 / np.sqrt(2)) * (np.random.randn(L, M) + 1j * np.random.randn(L, M))

    # --- Apply Spatial Correlations ---
    H_r = sqrtm(psi_irs) @ H_rbar @ sqrtm(psi_ue.T)
    G = sqrtm(psi_bs) @ G_bar @ sqrtm(psi_irs.T)

    # --- Build Equivalent Channel Correlation Matrix ---
    R_Gamma = np.kron(L * np.multiply(psi_irs, psi_irs), psi_ue)

    try:
        R_Gamma_inv = inv(R_Gamma)
    except np.linalg.LinAlgError:
        print("Warning: R_Gamma is singular or ill-conditioned. Using pseudo-inverse.")
        R_Gamma_inv = np.linalg.pinv(R_Gamma)

    # --- Generate DFT-based Orthogonal Training Sequences ---
    X = np.zeros((K, tau), dtype=complex)
    pilot_dft_base = dft(tau)

    for k_idx in range(K):
        dk_col = pilot_dft_base[:, k_idx]
        norm_sq_dk = np.linalg.norm(dk_col) ** 2
        xk_col = np.sqrt(Pk / norm_sq_dk) * dk_col
        X[k_idx, :] = xk_col.T

    # --- Non-ideal RIS Amplitude Function ---
    def amplitude_func(theta_phase):
        return ((1 - b_min) * ((np.sin(theta_phase - delta_ris) + 1) / 2) ** alpha_ris) + b_min

    return {
        'K': K,
        'M': M,
        'L': L,
        'B': B,
        'tau': tau,
        'Pk': Pk,
        'psi_bs': psi_bs,
        'psi_ue': psi_ue,
        'psi_irs': psi_irs,
        'H_r': H_r,
        'G': G,
        'R_Gamma': R_Gamma,
        'R_Gamma_inv': R_Gamma_inv,
        'X': X,
        'amplitude_func': amplitude_func,
        'b_min': b_min,
        'alpha_ris': alpha_ris,
        'delta_ris': delta_ris
    }

# %%
def calculate_nmse(system_params, theta_ris_phases, snr_db): 
    """ Calculates the Normalized LMMSE (NMSE) for given system parameters, RIS phases, and SNR. """ 
    from numpy.linalg import inv 
    from scipy.linalg import sqrtm 
    
    M = system_params['M'] 
    K = system_params['K'] 
    L = system_params['L'] 
    B = system_params['B'] 
    R_Gamma_inv = system_params['R_Gamma_inv'] 
    X = system_params['X'] 
    amplitude_func = system_params['amplitude_func'] 
    
    # Convert SNR to linear scale 
    snr_linear = 10**(snr_db / 10) 
    sigma2 = system_params['Pk'] / snr_linear 
    
    # Calculate V (RIS reflection coefficient matrix) 
    beta_values = amplitude_func(theta_ris_phases) 
    V = beta_values * np.exp(1j * theta_ris_phases) # (M, B) 
    
    # --- FIXED S CONSTRUCTION (valid for any B) --- 
    V_kron_IK = np.kron(np.eye(K), V) # (K*M, K*B) 
    IB_kron_X = np.kron(np.eye(B), X) # (B*K, B*tau) 
    S = V_kron_IK @ IB_kron_X # (K*M, B*tau) 
    
    # Calculate matrices for NMSE computation 
    SS_H = S @ S.conj().T # (K*M, K*M) 
    matrix_to_invert = R_Gamma_inv + (1 / (sigma2 * L)) * SS_H 
    
    try: 
        inverted_matrix = inv(matrix_to_invert) 
    except np.linalg.LinAlgError: 
        print(f"Warning: Matrix inversion failed at SNR={snr_db}dB. Using pseudo-inverse.") 
        inverted_matrix = np.linalg.pinv(matrix_to_invert) 
    
    mmse = np.trace(inverted_matrix).real 
    nmse = mmse / (L * K * M) 
    
    return nmse

# %%
def generate_quantized_phases(M, B, n): 
    """ Generates quantized RIS phases using 2^n discrete levels. Args: M (int): Number of RIS elements. B (int): Number of time slots/beams. n (int): Number of quantization bits (e.g., n=3 gives 8 levels). Returns: np.ndarray: (M, B) array of quantized phase values. """ 
    L = 2 ** n # Number of discrete phase levels 
    phase_levels = np.linspace(0, 2 * np.pi, num=L, endpoint=False) # Discrete phase options 
    indices = np.random.randint(0, L, size=(M, B)) # Randomly choose indices 
    return phase_levels[indices]

# %%
class RISOptimizeFunc(Func):
    """ Optimization wrapper class for a Reconfigurable Intelligent Surface (RIS) system with an MxB configuration. This class defines a function compatible with LA-MCTS optimization framework, where each variable corresponds to a quantized RIS phase shift element. """
    
    def __init__(self, system_params, n_bits, snr_db, init_matrix: Optional[np.ndarray] = None): 
        """ Initialize the optimization function. Args: system_params (dict): System parameters containing keys: - 'M': Number of RIS rows - 'B': Number of RIS columns - 'K': Number of users - 'L': Number of quantization levels n_bits (int): Number of quantization bits (controls number of phase levels) snr_db (float): Signal-to-noise ratio in dB init_matrix (np.ndarray, optional): Optional initial RIS phase matrix (radians) """ 
        
        # Store system configuration 
        self._system_params = system_params 
        self._M = self._system_params['M'] 
        self._B = self._system_params['B'] 
        self._K = self._system_params['K'] 
        self._L = self._system_params['L'] # Number of quantized levels in the system 
        
        # Quantization and signal parameters 
        self._n_bits = n_bits 
        self._snr_db = snr_db 
        self._num_levels = 2 ** self._n_bits # Total number of discrete phase levels 
        
        # --- Problem dimensions ---  
        # Each RIS element (M×B total) is a variable with discrete phase index 
        self._dims = self._M * self._B 

        # Lower and upper bounds of discrete indices 
        self._lb = np.zeros(self._dims, dtype=float) 
        self._ub = np.full(self._dims, self._num_levels - 1, dtype=float) 
        
        # Define quantized phase values in radians (uniformly spaced from 0 to 2π) 
        self._quantized_phases_rad = np.linspace( 0, 2 * np.pi * (1 - 1/self._num_levels), self._num_levels ) 
        
        # --- Initialize phase indices --- 
        if init_matrix is not None:
            # If a continuous initial phase matrix is given: 
            # Map each element to the nearest quantized phase index
            indices_matrix = np.argmin( np.abs(init_matrix[..., None] - self._quantized_phases_rad), axis=-1 ) 
            
            # Flatten to vector format for optimization interface
            self.init_x = indices_matrix.flatten() 
        else: # Default initialization: all zeros (first quantization level) 
            self.init_x = np.zeros(self._dims, dtype=int) 
    
    @property
    def dims(self): return self._dims

    @property
    def lb(self): return self._lb

    @property
    def ub(self): return self._ub

    @property
    def is_discrete(self): return np.full(self.dims, True)

    @property
    def is_minimizing(self): return True

    def __call__(self, x: np.ndarray):
        batch_size = x.shape[0]
        nmse_results = np.zeros(batch_size)
        for i in range(batch_size):
            indices_vector = x[i].astype(int)
            theta = self._quantized_phases_rad[indices_vector].reshape(self._M, self._B)
            nmse_results[i] = calculate_nmse(self._system_params, theta, self._snr_db)
        return nmse_results, None

    def mcts_params(self, sampler, classifier) -> Dict:
        params = get_mcts_params(sampler, classifier)
        params["params"]["cp"] = 0.05
        params["params"]["leaf_size"] = self.dims * 2
        params["params"]["num_init_samples"] = self.dims * 5
        params["params"]["num_samples_per_sampler"] = self.dims * 10
        return params

    def __str__(self):
        return f"RISOptimizeFunc(K={self._K}, M={self._M}, B={self._B}, L={self._L}, n_bits={self._n_bits})"

# %%
def optimize_ris_phases_sa(system_params, snr_db, n,
                           base_iter=500, T_start=1.0, T_end=1e-3,
                           alpha=None, num_monte_carlo=1,
                           initial_phase_matrix=None):
    """
    Adaptive Simulated Annealing (SA) for quantized RIS phase optimization.
    """

    # --- Extract system dimensions ---
    M, B = system_params['M'], system_params['B']
    L = 2 ** n  # Number of discrete phase levels (quantization levels)

    # --- Define quantized phase values (uniformly spaced) ---
    phase_levels = np.linspace(0, 2 * np.pi, L, endpoint=False)

    # --- Adaptive hyperparameters ---
    max_iter = base_iter * n  # More iterations for finer quantization
    if alpha is None:
        alpha = 1 - (0.01 / n)  # Slower cooling for larger n
    T_start = T_start * (1 + 0.2 * (n - 1))  # Scale starting temperature with n

    # --- Initialize tracking variables ---
    all_nmse_results = []
    best_overall_nmse = float('inf')
    best_overall_theta = None

    # --- Run multiple Monte Carlo trials ---
    for _ in range(num_monte_carlo):

        # --- Step 1: Initialize RIS phase matrix ---
        if initial_phase_matrix is not None:
            current_theta = initial_phase_matrix.copy()
        else:
            current_theta = np.random.choice(phase_levels, size=(M, B))

        # Compute initial NMSE
        current_nmse = calculate_nmse(system_params, current_theta, snr_db)

        # Track best solution in this run
        best_theta_run = current_theta.copy()
        best_nmse_run = current_nmse

        # Set current temperature
        T = T_start

        # --- Step 2: Main Simulated Annealing loop ---
        for _ in range(max_iter):

            # --- Generate a neighboring solution ---
            candidate_theta = current_theta.copy()

            for _ in range(np.random.randint(1, 4)):
                i = np.random.randint(0, M)
                j = np.random.randint(0, B)
                candidate_theta[i, j] = np.random.choice(phase_levels)

            # --- Evaluate the new candidate ---
            candidate_nmse = calculate_nmse(system_params, candidate_theta, snr_db)

            # --- Step 3: Acceptance criterion ---
            delta = current_nmse - candidate_nmse
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                current_theta = candidate_theta
                current_nmse = candidate_nmse

                if candidate_nmse < best_nmse_run:
                    best_nmse_run = candidate_nmse
                    best_theta_run = candidate_theta.copy()

            # --- Step 4: Cooling schedule ---
            T *= alpha
            if T < T_end:
                break

        # --- Record the best NMSE from this Monte Carlo run ---
        all_nmse_results.append(best_nmse_run)

        # --- Update global best across all Monte Carlo runs ---
        if best_nmse_run < best_overall_nmse:
            best_overall_nmse = best_nmse_run
            best_overall_theta = best_theta_run.copy()

    # --- Step 5: Return best and averaged results ---
    average_nmse = np.mean(all_nmse_results)

    return best_overall_theta, average_nmse

# %%
def optimize_ris_phases_pso(system_params, snr_db, n,
                            num_particles=30, max_iter=200,
                            w=0.7, c1=1.5, c2=1.5,
                            num_monte_carlo=1,
                            initial_phase_matrix=None):
    """
    Particle Swarm Optimization (PSO) for quantized RIS phase configuration.
    """

    # --- Extract system dimensions ---
    M, B = system_params['M'], system_params['B']
    L = 2 ** n

    # Define quantized phase levels uniformly between [0, 2π)
    phase_levels = np.linspace(0, 2 * np.pi, L, endpoint=False)

    # --- Initialize results tracking ---
    all_nmse_results = []
    best_overall_nmse = float('inf')
    best_overall_position = None

    # --- Run multiple Monte Carlo trials ---
    for _ in range(num_monte_carlo):

        # --- Step 1: Initialize particles ---
        particles = []
        if initial_phase_matrix is not None:
            particles.append(initial_phase_matrix.copy())
            for _ in range(num_particles - 1):
                particles.append(generate_quantized_phases(M, B, n))
        else:
            particles = [generate_quantized_phases(M, B, n) for _ in range(num_particles)]

        # Initialize velocities
        velocities = [np.zeros((M, B)) for _ in range(num_particles)]

        # --- Step 2: Initialize personal & global bests ---
        pbest_positions = [p.copy() for p in particles]
        pbest_scores = [calculate_nmse(system_params, p, snr_db) for p in particles]

        gbest_index = np.argmin(pbest_scores)
        gbest_position_run = pbest_positions[gbest_index].copy()
        gbest_score_run = pbest_scores[gbest_index]

        # --- Step 3: Main PSO iteration loop ---
        for _ in range(max_iter):
            for i in range(num_particles):

                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (pbest_positions[i] - particles[i]) +
                                 c2 * r2 * (gbest_position_run - particles[i]))

                particles[i] += velocities[i]

                particles[i] = phase_levels[
                    np.argmin(np.abs(phase_levels[:, None, None] - particles[i]), axis=0)
                ]

                score = calculate_nmse(system_params, particles[i], snr_db)

                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_positions[i] = particles[i].copy()

                    if score < gbest_score_run:
                        gbest_score_run = score
                        gbest_position_run = particles[i].copy()

        # --- Step 4: Record results from this Monte Carlo run ---
        all_nmse_results.append(gbest_score_run)

        if gbest_score_run < best_overall_nmse:
            best_overall_nmse = gbest_score_run
            best_overall_position = gbest_position_run.copy()

    # --- Step 5: Compute average NMSE ---
    average_nmse = np.mean(all_nmse_results)

    return best_overall_position, average_nmse

# %%
def optimize_ris_phases_coordinate_descent(
        system_params,
        snr_db,
        n,
        max_iter=1,
        num_monte_carlo=1,
        initial_phase_matrix=None):
    """
    Cyclic Coordinate Descent (CCD)

    Sequentially updates each RIS element while keeping all
    other elements fixed.

    Returns:
        best_theta, best_nmse
    """

    M, B = system_params['M'], system_params['B']

    phase_levels = np.linspace(
        0,
        2*np.pi,
        2**n,
        endpoint=False
    )

    best_overall_nmse = np.inf
    best_overall_theta = None

    for mc in range(num_monte_carlo):

        if initial_phase_matrix is not None:
            theta = initial_phase_matrix.copy()
        else:
            theta = generate_quantized_phases(M, B, n)

        current_nmse = calculate_nmse(
            system_params,
            theta,
            snr_db
        )

        for iteration in range(max_iter):

            improvement = False

            for i in range(M):
                for j in range(B):

                    best_phase = theta[i, j]
                    best_nmse = current_nmse

                    for phase in phase_levels:

                        candidate_theta = theta.copy()
                        candidate_theta[i, j] = phase

                        candidate_nmse = calculate_nmse(
                            system_params,
                            candidate_theta,
                            snr_db
                        )

                        if candidate_nmse < best_nmse:
                            best_nmse = candidate_nmse
                            best_phase = phase

                    if best_nmse < current_nmse:
                        theta[i, j] = best_phase
                        current_nmse = best_nmse
                        improvement = True

            if not improvement:
                break

        if current_nmse < best_overall_nmse:
            best_overall_nmse = current_nmse
            best_overall_theta = theta.copy()

    return best_overall_theta, best_overall_nmse

# %%
# System Parameters
M = 64
SNR_values = [10.0]
N_values = [2]

# LA-MCTS Specific Configuration
# Scale LA-MCTS budget with n
base_budget = 4000
budget_scale = {2: base_budget, 3: base_budget * 3}  # adjust factors as needed

SAMPLER = SamplerEnum.NEVERGRAD_SAMPLER
CLASSIFIER = ClassifierEnum.KMEAN_SVM_CLASSIFIER
OUTPUT_DIR = "ris_optimization_results"
os.makedirs(OUTPUT_DIR, exist_ok=True) # Create output directory

# Monte Carlo Runs per Algorithm
num_monte_carlo_baseline = 100
num_monte_carlo_sa = 20
num_monte_carlo_pso = 50
# A single detailed run for LA-MCTS is often sufficient given its high budget
num_monte_carlo_lamcts = 5

nmse_results_n_snr = {}

# Main loop over quantization bits
for n in N_values:
    print(f"\n{'='*20}\n--- Quantization Bits (n) = {n} ---\n{'='*20}")
    nmse_results_n_snr[n] = {}
    
    system_params_n = setup_ris_system(K=4, M=M, L=16)

    # Initialize lists to store average NMSE for each SNR value
    nmse_baseline_snr_list, nmse_sa_snr_list, nmse_pso_snr_list, nmse_lamcts_snr_list, nmse_cd_snr_list = [], [], [], [], []

    # Loop over each SNR value
    for snr in SNR_values:
        print(f"\n--- Running for SNR = {snr} dB ---")
        initial_phase_matrix = generate_quantized_phases(system_params_n['M'], system_params_n['B'], n)

        # --- Baseline (Random) ---
        # nmse_baseline_mc_run = [np.mean([calculate_nmse(system_params_n, generate_quantized_phases(system_params_n['M'], system_params_n['B'], n), snr) for _ in range(num_monte_carlo_baseline)])]
        # avg_nmse_baseline = nmse_baseline_mc_run[0]
        # nmse_baseline_snr_list.append(avg_nmse_baseline)
        # print(f"  Random (Baseline) NMSE: {avg_nmse_baseline:.6f}")

        # --- SA ---
        # nmse_sa_mc_run = [optimize_ris_phases_sa(system_params_n, snr_db=snr, n=n, num_monte_carlo=1, initial_phase_matrix=initial_phase_matrix)[1] for _ in range(num_monte_carlo_sa)]
        # avg_nmse_sa = np.mean(nmse_sa_mc_run)
        # nmse_sa_snr_list.append(avg_nmse_sa)
        # print(f"  SA NMSE:                {avg_nmse_sa:.6f}")

        # --- PSO ---
        # nmse_pso_mc_run = [optimize_ris_phases_pso(system_params_n, snr_db=snr, n=n, num_monte_carlo=1, initial_phase_matrix=initial_phase_matrix)[1] for _ in range(num_monte_carlo_pso)]
        # avg_nmse_pso = np.mean(nmse_pso_mc_run)
        # nmse_pso_snr_list.append(avg_nmse_pso)
        # print(f"  PSO NMSE:               {avg_nmse_pso:.6f}")

        # ----- Coordinate Descent -----
        cd_start = time.time()

        nmse_cd_mc_run = [
            optimize_ris_phases_coordinate_descent(
                system_params_n,
                snr_db=snr,
                n=n,
                max_iter=5,
                num_monte_carlo=1,
                initial_phase_matrix=initial_phase_matrix
            )[1]
            for _ in range(10)
        ]

        cd_end = time.time()

        avg_nmse_cd = np.mean(nmse_cd_mc_run)
        total_cd_time = cd_end - cd_start
        avg_cd_time = total_cd_time / 10

        nmse_cd_snr_list.append(avg_nmse_cd)

        print(f"  Coordinate Descent NMSE: {avg_nmse_cd:.6f}")
        print(f"  Coordinate Descent Avg Time: {avg_cd_time:.2f} sec/run")
        print(f"  Coordinate Descent Total Time: {total_cd_time:.2f} sec/run")
        
        # --- LA-MCTS (Detailed Run) ---
        print("\n  --- Starting LA-MCTS Search ---")
        nmse_lamcts_mc_run = []
        for i in range(num_monte_carlo_lamcts):
            ris_func = RISOptimizeFunc(
                system_params=system_params_n,
                n_bits=n,
                snr_db=snr,
                init_matrix=initial_phase_matrix
            )
            func_wrapper = StatsFuncWrapper(ris_func)
            
            # Step 1: Calculate and store the initial NMSE to use as a baseline
            fval, _ = ris_func(ris_func.init_x.reshape(1, -1))
            initial_nmse = fval[0]
            print(f"  LA-MCTS Baseline NMSE (from initial matrix): {initial_nmse:.6f}")

            mcts_params = ris_func.mcts_params(sampler=SAMPLER, classifier=CLASSIFIER)
            mcts = MCTS.create_mcts(func_wrapper, func_wrapper, mcts_params)
            
            start_time = time.time()
            try:
                CALL_BUDGET = budget_scale[n]
                print(f"  Optimizing with a budget of {CALL_BUDGET} evaluations...")
                stats = mcts.search(greedy=GreedyType.Best, call_budget=CALL_BUDGET)
            except Exception as e:
                print(f"  An error occurred during LA-MCTS search: {e}")
                stats = func_wrapper.stats # Retrieve stats collected so far
            
            end_time = time.time()
            print(f"  Search finished in {end_time - start_time:.2f} seconds.")

            # Step 2: Implement the "Safety Net". Initialize the best overall result with the baseline.
            best_nmse_overall = initial_nmse

            if stats and len(stats.call_history) > 0:
                best_result_from_search = stats.call_history[-1].fx
                print(f"  Total function calls: {stats.total_calls}")
                print(f"  Best NMSE found by LA-MCTS Search: {best_result_from_search:.6f}")
                
                # Compare the search result with the initial baseline and keep the minimum
                best_nmse_overall = min(initial_nmse, best_result_from_search)
                
                sampler_name = SAMPLER.name.lower()
                output_file = os.path.join(OUTPUT_DIR, f"ris_stats_n{n}_snr{snr}_{sampler_name}.pkl")
                with open(output_file, "wb") as f:
                    pickle.dump(stats, f)
                print(f"  Full LA-MCTS stats saved to '{output_file}'")
            else:
                print("  LA-MCTS search did not produce any results. Using baseline NMSE as the final result.")
                # No action needed, best_nmse_overall is already set to the initial_nmse

            # Step 3: Append the guaranteed best result to the list
            nmse_lamcts_mc_run.append(best_nmse_overall)
            print(f"  ---------------------------------")
            print(f"  Final NMSE recorded for LA-MCTS (guaranteed best): {best_nmse_overall:.6f}")
            print(f"  ---------------------------------")

        avg_nmse_lamcts = np.mean(nmse_lamcts_mc_run)
        nmse_lamcts_snr_list.append(avg_nmse_lamcts)

        # Store the collected results for the current n
        nmse_results_n_snr[n] = {
            'Random': nmse_baseline_snr_list,
            'SA': nmse_sa_snr_list,
            'CD': nmse_cd_snr_list,
            'PSO': nmse_pso_snr_list,
            'LA-MCTS': nmse_lamcts_snr_list
        }

print(f"\n\n{'='*20}\n--- FINAL SUMMARY ---\n{'='*20}")
# Pretty print the final dictionary
import json
print(json.dumps(nmse_results_n_snr, indent=2))

# %%
n = 3

plt.figure(figsize=(8,5))

plt.plot(SNR_values, nmse_results_n_snr[n]['Random'],
         marker='x', linewidth=2.5, label='Random')

plt.plot(SNR_values, nmse_results_n_snr[n]['CD'],
         marker='o', linewidth=2.5, label='CD')

plt.plot(SNR_values, nmse_results_n_snr[n]['SA'],
         marker='s', linewidth=2.5, label='SA')

plt.plot(SNR_values, nmse_results_n_snr[n]['PSO'],
         marker='^', linewidth=2.5, label='PSO')

plt.plot(SNR_values, nmse_results_n_snr[n]['LA-MCTS'],
         marker='d', linewidth=2.5, label='LA-MCTS-PO')

plt.yscale('log')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()