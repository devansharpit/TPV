from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


class TPVBase(ABC):
    def __init__(self, device=None, seed=0):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

    @abstractmethod
    def compute_tpv(self, *args, **kwargs):
        pass

    def compute_hutchinson_trace(self, model, X, y, loss_fn=None, n_samples=200):
        """
        Estimate Tr(∇²L(w*)) via Rademacher Hutchinson trace estimator.

        For each v ~ {±1}^p:
            v^T H v = v^T ∇_w (∇_w L · v)  (double backward / HVP)
        Average over n_samples gives Tr(H).

        loss_fn defaults to CrossEntropyLoss for long targets, MSELoss otherwise.

        Reference: compute_sharpness_hutchinson() in
        applications/label_noise_sensitivity_label-noise-tvp_vs_sgd-noise-tpv.py
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss() if y.dtype == torch.long else nn.MSELoss()

        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]
        trace_estimate = 0.0

        for _ in range(n_samples):
            vs = [
                torch.randint(0, 2, p.shape, device=self.device).float() * 2 - 1
                for p in params
            ]

            model.zero_grad()
            loss = loss_fn(model(X), y)
            grads = torch.autograd.grad(loss, params, create_graph=True)

            gv = sum((g * v).sum() for g, v in zip(grads, vs))

            model.zero_grad()
            hvs = torch.autograd.grad(gv, params, retain_graph=False)

            vhv = sum((v * hv).sum().item() for v, hv in zip(vs, hvs))
            trace_estimate += vhv

        return trace_estimate / n_samples

    def compute_hutchinson_trace_heff(self, model, X, n_samples=200):
        """
        Estimate Tr(H_eff) = E_x[||J(x)||_F^2] directly, where
            J(x) = ∂f(x; w) / ∂w  ∈ R^{K x p}
        and the expectation is over the (training or test) input distribution.

        This is the quantity the TPV trace stability theorem (Thm 3.1) actually
        governs. It differs from compute_hutchinson_trace, which estimates
        Tr(∇²L(w*)) — the full loss Hessian — and which only coincides with
        Tr(H_eff) under squared loss with small residuals (Appendix E.2).

        Estimator: a doubly-stochastic Hutchinson estimate. Each iteration
        draws a single sample x ~ Uniform(X) and a single Rademacher vector
        v ∈ {±1}^K, then computes
            g = ∇_w <v, f(x; w)> = J(x)^T v   (one backward pass)
            ||g||^2  is an unbiased estimator of ||J(x)||_F^2  (over v),
        and the outer average over x is an unbiased estimator of
        E_x[||J(x)||_F^2] = Tr(H_eff). One backward pass per sample, matching
        the cost of compute_hutchinson_trace.

        Why single-sample (not batched): processing a mini-batch with a single
        backward of Σ_b <v_b, f(x_b)> would yield ||Σ_b J(x_b)^T v_b||^2, which
        contains cross-terms Σ_{a≠b} <J(x_a)^T v_a, J(x_b)^T v_b>. We want the
        sum (not the squared norm of the sum), so each (x, v) pair must be
        processed with its own backward. The forward, however, can be batched
        (forward_batch_size below) since it is shared across the v draws.

        Notes:
            * Label-free: y is not used. This isolates the geometric quantity
              from any residual / loss-shape effects.
            * Variance scales with both Var_x[||J(x)||_F^2] and the per-sample
              Rademacher variance; a few hundred draws is typically enough for
              a stable estimate when |X| ≳ 1000.

        Args:
            model:       network at w* (eval mode is set internally).
            X:           input tensor (N, ...) on self.device. Treated as the
                         empirical distribution from which x is sampled
                         uniformly without replacement (with replacement once
                         exhausted).
            n_samples:   number of (x, v) Monte-Carlo draws.

        Returns:
            float — estimate of Tr(H_eff(w*; X)) = (1/|X|) Σ_x ||J(x)||_F^2.
        """
        model.eval()
        params = [p for p in model.parameters() if p.requires_grad]

        N = X.shape[0]
        running_sum = 0.0

        # Sample indices uniformly from X. For n_samples ≤ N we sample
        # without replacement; otherwise fall back to with-replacement.
        if n_samples <= N:
            perm = torch.randperm(N, device=self.device)[:n_samples]
        else:
            perm = torch.randint(0, N, (n_samples,), device=self.device)

        for i in range(n_samples):
            idx = perm[i].item()
            x = X[idx : idx + 1].to(self.device)  # keep batch dim of size 1

            model.zero_grad()
            out = model(x)  # (1,) or (1, K)
            out = out.squeeze(0)  # (K,) or scalar

            if out.ndim == 0:
                v = torch.randint(0, 2, (), device=self.device).float() * 2 - 1
                scalar = v * out
            else:
                v = torch.randint(0, 2, out.shape, device=self.device).float() * 2 - 1
                scalar = (v * out).sum()

            grads = torch.autograd.grad(scalar, params, retain_graph=False)
            running_sum += sum(g.pow(2).sum().item() for g in grads)

        return running_sum / n_samples

    @torch.no_grad()
    def compute_f_star(self, model, X, batch_size=256):
        """Batched no-grad forward pass returning logits on device."""
        model.eval()
        logits_list = []
        for i in range(0, X.shape[0], batch_size):
            batch = X[i : i + batch_size].to(self.device)
            logits_list.append(model(batch))
        return torch.cat(logits_list, dim=0)

    def compute_tpv_from_runs(self, preds_runs, f_star):
        """
        Aggregate empirical TPV across Monte Carlo runs.

        preds_runs: list of length R, each element (n, K) or (n,) numpy array
        f_star:     (n, K) or (n,) numpy array of clean reference predictions

        Returns: E_{runs, x}[||z_r(x) - z*(x)||²]

        Reference: aggregation in investigation/tpv_label_noise_cifar.py:369-385
        """
        preds = np.stack(preds_runs, axis=0)  # (R, n, K) or (R, n)
        f_star_np = f_star if isinstance(f_star, np.ndarray) else f_star.cpu().numpy()
        diffs = preds - f_star_np[None]
        if diffs.ndim == 3:
            return float(np.mean(np.sum(diffs ** 2, axis=-1)))
        return float(np.mean(diffs ** 2))

    def compute_proximity_penalty(self, model, ref_state_dict):
        """L2 penalty: Σ ||w - w_ref||² over all parameters.

        Reference: investigation/tpv_label_noise_cifar.py:58-68
        """
        penalty = torch.tensor(0.0, device=self.device)
        for name, param in model.named_parameters():
            if name in ref_state_dict:
                penalty = penalty + torch.sum((param - ref_state_dict[name]) ** 2)
        return penalty