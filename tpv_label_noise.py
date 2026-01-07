import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle as pkl
import os

# ----------------------------
# Configs/Hyperparams
# ----------------------------

d = 20           # input dimension
n_train = 1000   # train samples
n_test = 5000    # test samples

width_list = [128, 256, 512, 800, 1024, 1600]
sigma_list = [0.01,  0.1] 
depth = 3
R = 50          # Monte Carlo runs per (width, sigma)
n_epochs_clean = 800
max_epochs_noisy = 500  # maximum epochs for noisy training (safety limit)
train_loss_thres = [0,0]  # training stops when loss goes below this threshold
lr_clean = 5e-3
lr_noisy = 5e-3
CENTER_PENALTY_LAMBDA = 0 # 1e-3  # strength of quadratic penalty around w*
compute_theoretical_TPV = True  # whether to compute theoretical TPV

save_dir = 'results'
save_file_name = f'{save_dir}/tpv_label_noise_results'


# ----------------------------
# Reproducibility
# ----------------------------
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Taylor-test + center penalty config
# ----------------------------
TAYLOR_N_REF = 128    # number of reference inputs for Taylor check
TAYLOR_H = 1e-2       # finite-difference step along (w - w*)

def compute_proximity_penalty(model, ref_state_dict):
    """
    Compute L2 proximity penalty: sum of ||w - w_ref||^2 over all parameters.
    """
    penalty = 0.0
    num_params = sum(p.numel() for p in model.parameters())
    for name, param in model.named_parameters():
        if name in ref_state_dict:
            ref_param = ref_state_dict[name]
            penalty = penalty + torch.sum((param - ref_param) ** 2)
    return penalty # num_params

# ----------------------------
# Simple Deep ReLU MLP
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, width, depth=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.ReLU())
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def get_param_vector(model):
    return torch.cat([p.view(-1) for p in model.parameters()])


# ----------------------------
# Training with optional proximity penalty
# ----------------------------
def train_full_batch(model,
                     X,
                     y,
                     max_epochs=1000,
                     train_loss_thres=0.001,
                     lr=1e-2,
                     wd=0.0,
                     center_lambda=0.0,
                     ref_state_dict=None,
                     print_stats=False,
                     X_test=None,
                     y_test=None):
    """
    Full-batch training with cosine LR and optional quadratic penalty
    around a reference parameter vector (center_params ~ w*).
    
    Training stops when training loss goes below train_loss_thres or max_epochs is reached.
    
    If X_test and y_test are provided, computes and returns the test loss.
    
    Returns: (test_loss or None, epochs_trained)
    """
    model.eval()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.MSELoss()

    epochs_trained = 0
    for epoch in range(max_epochs):
        epochs_trained = epoch + 1
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)

        # Quadratic penalty around w*
        if ref_state_dict is not None and center_lambda > 0.0:
            penalty = compute_proximity_penalty(model, ref_state_dict)
            loss = loss + center_lambda * penalty

        if not torch.isfinite(loss):
            print(f"Non-finite loss encountered: {loss.item()}")
            break

        loss.backward()
        # Mild gradient clipping to avoid explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        # scheduler.step()

        # Check training loss (MSE only, without penalty) after each epoch
        model.eval()
        with torch.no_grad():
            train_preds = model(X)
            current_train_loss = criterion(train_preds, y).item()
        # model.train()

        # Stop if training loss is below threshold
        if current_train_loss < train_loss_thres:
            if print_stats:
                print(f"  Training loss {current_train_loss:.6f} < threshold {train_loss_thres}, stopping at epoch {epoch}")
            break

    if print_stats:
        print(f"  Final training loss: {current_train_loss:.6f} after {epochs_trained} epochs")

    # NaN check
    for p in model.parameters():
        if torch.isnan(p).any():
            print("Warning: NaN detected in model parameters after training.")

    # Compute test loss if test data is provided
    if X_test is not None and y_test is not None:
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test)
            test_loss = criterion(test_preds, y_test).item()
        if print_stats:
            print(f"  Final test loss: {test_loss:.6f}")
        return test_loss, epochs_trained
    
    return None, epochs_trained


# ----------------------------
# First-order Taylor check
# ----------------------------
def compute_taylor_rel_err(model_run,
                           input_dim,
                           width,
                           depth,
                           base_state_dict,
                           X_ref,
                           f_star_ref,
                           h=TAYLOR_H,
                           device=device):
    """
    Compute relative first-order Taylor error of model_run around w*.

    We test along the displacement Δ = w - w* using finite differences:
        d(x) = f_w(x) - f_{w*}(x)
        g(x) = [f(w* + h Δ; x) - f_{w*}(x)] / h

    and return
        rel_err = E[(d-g)^2] / (E[d^2] + eps).
    """
    X_ref = X_ref.to(device)
    f_star_ref = f_star_ref.to(device)

    model_run = model_run.to(device)
    model_run.eval()
    with torch.no_grad():
        f_run_ref = model_run(X_ref)

    # Build finite-difference model at w* + h (w - w*)
    model_fd = MLP(input_dim, width, depth=depth).to(device)

    state_star = {k: v.to(device) for k, v in base_state_dict.items()}
    state_run = model_run.state_dict()

    state_fd = {}
    for k in state_star.keys():
        v_star = state_star[k]
        v_run = state_run[k]
        delta = v_run - v_star
        state_fd[k] = v_star + h * delta

    model_fd.load_state_dict(state_fd)
    model_fd.eval()
    with torch.no_grad():
        f_fd_ref = model_fd(X_ref)

    d = f_run_ref - f_star_ref
    g = (f_fd_ref - f_star_ref) / h

    num = ((d - g) ** 2).mean().item()
    denom = (d ** 2).mean().item() + 1e-12
    rel_err = num / denom
    return rel_err


# ----------------------------
# Jacobian and TPV helpers
# ----------------------------
def compute_train_jacobian(model, X):
    """
    J: (n_train, P) where each row is grad f(x_i) wrt flattened params.
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    n = X.shape[0]
    with torch.no_grad():
        P = sum(p.numel() for p in params)
    J = torch.zeros(n, P, device=device)
    for i in range(n):
        model.zero_grad(set_to_none=True)
        x = X[i:i+1]
        out = model(x)
        (out.sum()).backward(retain_graph=False)
        grads = []
        for p in params:
            grads.append(p.grad.view(-1))
        g = torch.cat(grads)
        J[i] = g
    if torch.isnan(J).any():
        print("Warning: NaN detected in Jacobian after training.")
    return J


def compute_G_diag(model, V, X_test):
    """
    Compute diag(G) where G = V^T H_eff V, H_eff = E_x[g(x) g(x)^T].
    G_ii = E_x[(g(x)^T v_i)^2].
    V: (P, r) right singular vectors from SVD of J.
    Returns: G_diag (r,)
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    r = V.shape[1]
    n_test = X_test.shape[0]
    G_accum = torch.zeros(r, device=device)

    for i in range(n_test):
        model.zero_grad(set_to_none=True)
        x = X_test[i:i+1]
        out = model(x)
        (out.sum()).backward(retain_graph=False)
        grads = []
        for p in params:
            grads.append(p.grad.view(-1))
        g = torch.cat(grads)  # (P,)
        a = V.t() @ g         # (r,)
        G_accum += a * a

    G_diag = G_accum / n_test
    return G_diag


def compute_theoretical_base_T(J, model, X_test):
    """
    Compute T = sum_i G_ii / s_i^2 (the sigma^2-independent part)
    using the SVD of J and test Jacobian-based G_diag.
    """
    J_cpu = J.detach().float().cpu()
    U, S, Vh = torch.linalg.svd(J_cpu, full_matrices=False)
    U = U.to(J.device)
    S = S.to(J.device)
    Vh = Vh.to(J.device)

    V = Vh.t()  # (P, r)
    G_diag = compute_G_diag(model, V, X_test)  # (r,)
    T = 0.0
    for gii, si in zip(G_diag, S):
        if si.item() > 0:
            T += gii / (si * si)
    
    mask = S>1e-5
    alpha = (G_diag[mask] * n_train) / (S[mask]**2)
    return T.item(), alpha.cpu().numpy()


# ----------------------------
# Empirical TPV estimation (with proximity penalty + Taylor errors)
# ----------------------------
def estimate_empirical_TPV(width, sigma, R, X_train, y_clean_train,
                           X_test, y_clean_test, f_star_test, f_star_train, depth=3,
                           max_epochs=1000, train_loss_thres=0.001, lr=1e-2, init_state=None,
                           center_lambda=0.0):
    """
    For fixed width and noise std sigma:
    - Fix an initialization (w*).
    - For each of R runs, add label noise, train from same init with
      a proximity penalty around w*, and measure prediction fluctuations
      around f_star on train and test.
    
    Training stops when training loss goes below train_loss_thres or max_epochs is reached.

    Also compute a first-order Taylor relative error per run and return
    the mean over runs.
    """
    preds_runs_test = []
    preds_runs_train = []
    train_losses = []
    test_losses = []
    taylor_rel_errs = []

    # Build reference set for Taylor test (shared across runs)
    n_ref = min(TAYLOR_N_REF, X_train.shape[0])
    X_ref = X_train[:n_ref].detach()
    with torch.no_grad():
        base_model = MLP(d, width, depth=depth).to(device)
        base_model.load_state_dict(init_state)
        base_model.eval()
        # f_star_ref = base_model(X_ref).detach()
    ref_state_dict = init_state

    for r in range(R):
        model = MLP(d, width, depth=depth).to(device)
        model.load_state_dict(init_state)

        # noisy labels
        eps = torch.randn_like(y_clean_train) * sigma
        y_noisy = y_clean_train + eps

        _, epochs_trained = train_full_batch(
            model,
            X_train,
            y_noisy,
            max_epochs=max_epochs,
            train_loss_thres=train_loss_thres,
            lr=lr,
            wd=0.0,
            center_lambda=center_lambda,
            ref_state_dict=ref_state_dict,
            print_stats=False,
        )

        # Compute final training and test losses
        model.eval()
        with torch.no_grad():
            train_preds = model(X_train)
            test_preds = model(X_test)
            criterion = nn.MSELoss()
            final_train_loss = criterion(train_preds, y_noisy).item()
            final_test_loss = criterion(test_preds, y_clean_test).item()
            train_losses.append(final_train_loss)
            test_losses.append(final_test_loss)

        preds_runs_test.append(test_preds.squeeze(-1).cpu().numpy())
        preds_runs_train.append(train_preds.squeeze(-1).cpu().numpy())

    preds_runs_test = np.stack(preds_runs_test, axis=0)   # (R, n_test)
    preds_runs_train = np.stack(preds_runs_train, axis=0) # (R, n_train)
    f_star_np = f_star_test.squeeze(-1).cpu().numpy()     # (n_test,)
    f_star_train_np = f_star_train.squeeze(-1).cpu().numpy()  # (n_train,)

    diffs_test = preds_runs_test - f_star_np[None, :]
    empirical_TPV = np.mean(diffs_test ** 2)

    diffs_train = preds_runs_train - f_star_train_np[None, :]
    empirical_TPV_train = np.mean(diffs_train ** 2)

    avg_train_loss = np.mean(train_losses)
    avg_test_loss = np.mean(test_losses)
    std_train_loss = np.std(train_losses)
    std_test_loss = np.std(test_losses)
    mean_taylor_rel_err = float(np.mean(taylor_rel_errs))

    return (empirical_TPV,
            empirical_TPV_train,
            avg_train_loss,
            avg_test_loss,
            std_train_loss,
            std_test_loss,
            mean_taylor_rel_err)


# ----------------------------
# Synthetic linear data
# y = x^T w_true  (no noise)
# ----------------------------


X_train = torch.randn(n_train, d, device=device)
X_test  = torch.randn(n_test, d, device=device)

w_true = torch.randn(d, 1, device=device)
y_clean_train = X_train @ w_true      # (n_train, 1)
y_clean_test  = X_test @ w_true       # (n_test, 1)

# ----------------------------
# Main experiment
# ----------------------------


# Store TPV values: shape (len(widths), len(sigmas))
theoretical_TPV = np.zeros((len(width_list), len(sigma_list)))
empirical_TPV = np.zeros((len(width_list), len(sigma_list)))
empirical_TPV_train = np.zeros((len(width_list), len(sigma_list)))

# training/test losses
train_loss_noisy = np.zeros((len(width_list), len(sigma_list)))
test_loss_noisy = np.zeros((len(width_list), len(sigma_list)))
train_loss_std = np.zeros((len(width_list), len(sigma_list)))
test_loss_std = np.zeros((len(width_list), len(sigma_list)))
test_loss_ref_model = np.zeros((len(width_list),))

J_rank = np.zeros(len(width_list))
alpha_all = {}
T_base_all = np.zeros(len(width_list))

# Taylor relative error (mean over runs) per (width, sigma)
taylor_rel_err = np.zeros((len(width_list), len(sigma_list)))

init_state = {}
for wi, width in enumerate(width_list):
    print(f"=== Width {width} ===")

    # 1) Train clean reference model to get W*
    model_clean = MLP(d, width, depth=depth).to(device)
    test_loss_ref_model[wi], _ = train_full_batch(
        model_clean,
        X_train,
        y_clean_train,
        max_epochs=n_epochs_clean,
        train_loss_thres=0.0,  # no early stopping for clean reference model
        lr=lr_clean,
        wd=0.0,
        center_lambda=0.0,
        print_stats=True,
        X_test=X_test,
        y_test=y_clean_test,
    )
    model_clean.eval()
    print(f"Reference model test loss (clean): {test_loss_ref_model[wi]:.6f}")

    init_state[width] = model_clean.state_dict()

    with torch.no_grad():
        f_star_test = model_clean(X_test)    # (n_test, 1)
        f_star_train = model_clean(X_train)  # (n_train, 1)

    if compute_theoretical_TPV:
        # 2) Compute J (train Jacobian) at W*
        print("  Computing train Jacobian...")
        J = compute_train_jacobian(model_clean, X_train)  # (n_train, P)

        try:
            J_rank[wi] = torch.linalg.matrix_rank(J, tol=1e-5).item()
        except:
            print("  SVD failed for J; setting rank to NaN.")
            J_rank[wi] = np.nan

        # 3) Compute theoretical base T = sum_i G_ii / s_i^2
        print("  Computing theoretical TPV base term...")
        T_base, alpha = compute_theoretical_base_T(J, model_clean, X_test)  # scalar
        alpha_all[width] = alpha
        T_base_all[wi] = T_base
        print(f"    Rank(J) = {J_rank[wi]/n_train}, T_base = {T_base:.4f}, alpha_min = {alpha.min():.4f}, alpha_max = {alpha.max():.4f}")
    
    # 4) For each sigma, compute theoretical + empirical TPV
    for si, sigma in enumerate(sigma_list):
        print(f"    sigma = {sigma:.3f}")
        if compute_theoretical_TPV:
            theoretical_TPV[wi, si] = (sigma ** 2) * T_base

        (emp_TPV,
         emp_TPV_train,
         avg_train_loss,
         avg_test_loss,
         std_train,
         std_test,
         mean_taylor_err) = estimate_empirical_TPV(
            width=width,
            sigma=sigma,
            R=R,
            X_train=X_train,
            y_clean_train=y_clean_train,
            X_test=X_test,
            y_clean_test=y_clean_test,
            f_star_test=f_star_test,
            f_star_train=f_star_train,
            depth=depth,
            max_epochs=max_epochs_noisy,
            train_loss_thres=train_loss_thres[si],
            lr=lr_noisy,
            init_state=init_state[width],
            center_lambda=CENTER_PENALTY_LAMBDA,
        )

        empirical_TPV[wi, si] = emp_TPV
        empirical_TPV_train[wi, si] = emp_TPV_train
        train_loss_noisy[wi, si] = avg_train_loss
        test_loss_noisy[wi, si] = avg_test_loss
        train_loss_std[wi, si] = std_train
        test_loss_std[wi, si] = std_test
        taylor_rel_err[wi, si] = mean_taylor_err

print("\nTheoretical TPV:\n", theoretical_TPV)
print("\nEmpirical TPV (Test):\n", empirical_TPV)
print("\nEmpirical TPV (Train):\n", empirical_TPV_train)
print("\nTraining Loss (Noisy):\n", train_loss_noisy)
print("\nTest Loss (Noisy):\n", test_loss_noisy)
print("\nTraining Loss Std:\n", train_loss_std)
print("\nTest Loss Std:\n", test_loss_std)
print("\nReference Model Test Loss (Clean):\n", test_loss_ref_model)

import pickle as pkl
with open(f"{save_file_name}.pkl", "wb") as f: # 6 is final version; 7 turned off lr scheduler
    pkl.dump({
        "width_list": width_list,
        "sigma_list": sigma_list,
        "theoretical_TPV": theoretical_TPV,
        "empirical_TPV": empirical_TPV,
        "empirical_TPV_train": empirical_TPV_train,
        "train_loss_noisy": train_loss_noisy,
        "test_loss_noisy": test_loss_noisy,
        "train_loss_std": train_loss_std,
        "test_loss_std": test_loss_std,
        "test_loss_ref_model": test_loss_ref_model,
        "taylor_rel_err": taylor_rel_err,
        "R": R,
        "CENTER_PENALTY_LAMBDA": CENTER_PENALTY_LAMBDA,
        "TAYLOR_N_REF": TAYLOR_N_REF,
        "TAYLOR_H": TAYLOR_H,
        "alpha_all": alpha_all,
        "J_rank": J_rank,
        "T_base": T_base_all,
    }, f)





# ----------------------------
# Plotting
# ----------------------------

# Create plots directory if it doesn't exist
if not os.path.exists(f"{save_dir}/plots"):
    os.makedirs(f"{save_dir}/plots")

# load results
with open(f"{save_file_name}.pkl", "rb") as f:
    data = pkl.load(f)
    width_list = data["width_list"]
    sigma_list = data["sigma_list"]
    theoretical_TPV = data["theoretical_TPV"]
    empirical_TPV_train = data["empirical_TPV_train"]
    empirical_TPV = data["empirical_TPV"]
    train_loss_noisy = data["train_loss_noisy"]
    test_loss_noisy = data["test_loss_noisy"]
    train_loss_std = data["train_loss_std"]
    test_loss_std = data["test_loss_std"]
    taylor_rel_err = data.get("taylor_rel_err", None)
    T_base = data.get("T_base", None)
    test_loss_ref_model = data.get("test_loss_ref_model", None)
print(f'taylor_rel_err exists: {taylor_rel_err is not None}')

gen_gap = test_loss_noisy - train_loss_noisy
test_loss_degradation = test_loss_noisy - test_loss_ref_model[:, np.newaxis]

sigma_idx_for_width = sigma_idx_for_loss = 0  # index for sigma_list
sigma_val = sigma_list[sigma_idx_for_width]

# ----------------------------
# Font size settings
# ----------------------------
LABEL_FONTSIZE = 30    # Font size for axis labels
TICK_FONTSIZE = 25     # Font size for tick values
TITLE_FONTSIZE = 25    # Font size for plot titles
LEGEND_FONTSIZE = 20  # Font size for legends


theoretical_TPV[theoretical_TPV > 100] = np.nan


# plot T_base vs width
if T_base is not None:
    plt.figure(figsize=(8,5))
    plt.plot(width_list, T_base, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Width', fontsize=LABEL_FONTSIZE)
    plt.ylabel('T_base', fontsize=LABEL_FONTSIZE)
    plt.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    # plot gen_gap on secondary y-axis
    ax2 = plt.twinx()
    for sigma_idx, sigma in enumerate(sigma_list[:1]):  # plot for first two sigma values
        ax2.plot(width_list, gen_gap[:,sigma_idx], marker='s', color='orange', linewidth=2, markersize=6, label=f'$\sigma$={sigma:.2f}')
    ax2.set_ylabel(f'Generalization Gap', color='orange', fontsize=LABEL_FONTSIZE)
    ax2.tick_params(axis='y', labelcolor='orange', labelsize=TICK_FONTSIZE)
    plt.title('T_base vs Width', fontsize=TITLE_FONTSIZE)
    plt.tight_layout()
    legend = plt.legend(fontsize=LEGEND_FONTSIZE)
    legend.get_frame().set_alpha(0)
    plt.savefig(f'{save_dir}/plots/T_base_vs_width.pdf', bbox_inches='tight')


# sigma_idx_for_width = sigma_idx_for_loss = 0  # index for sigma_list
# sigma_val = sigma_list[sigma_idx_for_width]

for sigma_idx_for_width in [0,1]:
    sigma_idx_for_loss = sigma_idx_for_width
    sigma_val = sigma_list[sigma_idx_for_width]
    fig, ax1 = plt.subplots(figsize=(8, 5))

    print(width_list)
    print(theoretical_TPV[:, sigma_idx_for_width]/(sigma_val**2))
    # Plot theoretical TPV on left y-axis
    color1 = 'tab:red'
    ax1.set_xlabel('Width', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel('Theoretical TPV',  fontsize=LABEL_FONTSIZE) # color=color1,
    line1 = ax1.plot(width_list, theoretical_TPV[:, sigma_idx_for_width],
                    marker='o',  label='Theory', color=color1,
                    linewidth=2, markersize=6) #
    ax1.tick_params(axis='y', labelsize=TICK_FONTSIZE) # , labelcolor=color1
    ax1.tick_params(axis='x', labelsize=TICK_FONTSIZE)

    # Create second y-axis for empirical TPV
    ax2 = ax1#.twinx()
    color2 = 'tab:blue'
    color3 = 'tab:green'
    ax2.set_ylabel('TPV',  fontsize=LABEL_FONTSIZE)# color=color2,
    line2 = ax2.plot(width_list, empirical_TPV[:, sigma_idx_for_width],
                    marker='s', color=color2, label='Empirical TPV (test)',
                    linewidth=2, markersize=6)
    line3 = ax2.plot(width_list, empirical_TPV_train[:, sigma_idx_for_width],
                    marker='^', color=color3, label='Empirical TPV (train)',
                    linewidth=2, markersize=6)
    ax2.tick_params(axis='y',  labelsize=TICK_FONTSIZE)# labelcolor=color2,

    plt.title(f"TPV vs Width (sigma = {sigma_val})", fontsize=TITLE_FONTSIZE)

    # Combine legends from both axes
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    legend = ax1.legend(lines, labels, loc='upper right', fontsize=LEGEND_FONTSIZE)
    legend.get_frame().set_alpha(0)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/plots/tpv_vs_width_sigma-{sigma_val}.pdf', bbox_inches='tight')



    # Training/Test Loss and TPV with dual y-axes and error bars (for fixed sigma)
    sigma_val_loss = sigma_list[sigma_idx_for_loss]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot both training and test loss with error bars on left y-axis
    for sigma_idx in range(test_loss_noisy.shape[1]):
        print(f'sigma: {sigma_list[sigma_idx]}, Test Loss: {test_loss_noisy[:, sigma_idx]}')
    color1 = 'tab:red'
    color2 = 'tab:orange'
    color5 = 'tab:brown'
    ax1.set_xlabel('Width', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel('Loss', color='black', fontsize=LABEL_FONTSIZE)
    line1 = ax1.errorbar(width_list, train_loss_noisy[:, sigma_idx_for_loss], 
                        yerr=train_loss_std[:, sigma_idx_for_loss],
                        marker='o', color=color1, label='Training Loss (Noisy Model)',
                        capsize=5, capthick=2)
    line2 = ax1.errorbar(width_list, test_loss_noisy[:, sigma_idx_for_loss], 
                        yerr=test_loss_std[:, sigma_idx_for_loss],
                        marker='s', color=color2, label='Test Loss (Noisy Model)',
                        capsize=5, capthick=2)
    line5 = ax1.plot(width_list, test_loss_ref_model,
                        marker='s', color=color5, label='Test Loss (Ref. Model)')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=TICK_FONTSIZE)
    ax1.tick_params(axis='x', labelsize=TICK_FONTSIZE)

    # Create second y-axis for empirical TPV
    ax2 = ax1.twinx()
    color3 = 'tab:blue'
    ax2.set_ylabel('Empirical TPV', color=color3, fontsize=LABEL_FONTSIZE)
    line3 = ax2.plot(width_list, empirical_TPV[:, sigma_idx_for_loss], 
                    marker='^', color=color3, label='Empirical TPV (test)',
                    linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color3, labelsize=TICK_FONTSIZE)

    # Add title and legend
    plt.title(f'Loss and TPV vs Width (sigma = {sigma_val_loss})', fontsize=TITLE_FONTSIZE)

    # Combine legends from both axes
    lines = [line1, line2, ] + line5 + line3 #+ line4
    labels = [l.get_label() for l in lines]
    legend = ax1.legend(lines, labels, loc='upper right', fontsize=LEGEND_FONTSIZE)
    legend.get_frame().set_alpha(0)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/plots/loss_and_tpv_vs_width_sigma-{sigma_val_loss}.pdf', bbox_inches='tight')

    plt.show()