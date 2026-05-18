import copy

import torch

from tpv.tpv_base import TPVBase


class QuantizationTPV(TPVBase):
    """
    TPV estimator for uniform quantization noise.

    From the paper: TPV_quant ≈ sigma_q² * Tr(∇²L(w*))

    For uniform quantization with n_bits, the rounding error variance per weight is:
        sigma_q² = delta² / 12
    where delta = (w_max - w_min) / (2^n_bits - 1) is the quantization step size,
    computed from the full weight range across all parameters.

    Two estimators are provided:

    * `compute_tpv` — analytic: TPV_quant ≈ sigma_q² · Tr(H_eff). Tr(H_eff) is
      estimated via the Jacobian Hutchinson trace estimator inherited from
      TPVBase (compute_hutchinson_trace_heff). This is the form derived in
      Remark 4.4.

    * `compute_tpv_empirical` — direct Monte Carlo: sample δw from the
      quantization noise distribution, perturb the parameters, measure
      E_x[(f_{w*+δw}(x) − f_*(x))²]. Bypasses the first-order linearization
      and the Tr(H_eff) ≈ Tr(∇²L) bridge.
    """

    def compute_tpv(
        self,
        model,
        X,
        y,
        n_bits=8,
        n_hutchinson_samples=200,
    ):
        """
        Compute TPV for uniform quantization noise.

        Args:
            model:                 trained model at w* (nn.Module, eval-mode)
            X:                     inputs (N, ...)
            y:                     targets (N,) long or (N, K) float
            n_bits:                quantization bit-width (default 8)
            n_hutchinson_samples:  number of Rademacher vectors for Hutchinson estimator

        Returns:
            tpv:     float — TPV estimate: sigma_q² * Tr(∇²L)
            trace_H: float — raw Hutchinson trace Tr(∇²L(w*))
        """
        all_weights = torch.cat([p.detach().flatten() for p in model.parameters()])
        w_min, w_max = all_weights.min().item(), all_weights.max().item()
        delta = (w_max - w_min) / (2 ** n_bits - 1)
        sigma_q_sq = (delta ** 2) / 12.0

        trace_H = self.compute_hutchinson_trace_heff(model, X,
                                                     n_samples=n_hutchinson_samples)
        return sigma_q_sq * trace_H, trace_H

    def compute_tpv_empirical(
        self,
        model,
        X_train,
        X_test=None,
        n_bits=8,
        n_runs=50,
        forward_batch_size=256,
        seed=None,
    ):
        """
        Empirical quantization-noise TPV via Monte Carlo over per-coordinate
        noise draws around w*.

        For each of n_runs realizations: sample δw with δw_j ~ Unif(-δ/2, δ/2)
        (matching the noise model in Remark 4.4 / Appendix F), copy w* + δw
        into the model parameters, forward-pass on X_train (and X_test, if
        given), record E_x[||f_{w*+δw}(x) − f_*(x)||²], and restore w*. The
        returned TPV is the mean across runs and inputs.

        Bypasses the first-order linearization (Eq. 6) and the
        Tr(H_eff) ≈ Tr(∇²L) bridge used by `compute_tpv`. Because
        quantization noise is data-independent and i.i.d. per coordinate,
        this estimator has no SGD-trajectory / momentum / center-penalty
        baggage — it is a near-direct probe of Eq. 2.

        We deliberately do not implement deterministic rounding to a fixed
        grid: in that case δw is a deterministic function of w* (zero-
        variance over runs), so a single realization is the entire signal,
        and the resulting per-realization perturbation does not match the
        zero-mean i.i.d. covariance model assumed in Remark 4.4. Using
        i.i.d. uniform additive noise is the apples-to-apples empirical
        counterpart of `compute_tpv` here. (Subtractive dithering gives a
        statistically equivalent draw and is therefore not exposed
        separately.)

        Args:
            model:               trained model at w* (nn.Module). Set to
                                 eval() internally and kept there.
            X_train:             inputs from the training distribution (N, ...).
            X_test:              optional inputs from the test distribution.
            n_bits:              quantization bit-width; matches `compute_tpv`.
            n_runs:              number of Monte Carlo perturbation realizations.
            forward_batch_size:  mini-batch size for the perturbed forward pass.
            seed:                optional RNG seed for the noise draws (None
                                 falls back to the default torch generator).

        Returns:
            dict with:
                tpv_train:               float
                tpv_test:                float | None
                tpv_train_per_run:       list[float]
                tpv_test_per_run:        list[float] | None
                n_runs:                  int
                delta:                   float — quantization step size used
                sigma_q_sq:              float — δ² / 12
        """
        model.eval()

        # Match compute_tpv exactly: global weight range, δ = range / (2^bits − 1).
        all_weights = torch.cat([p.detach().flatten() for p in model.parameters()])
        w_min, w_max = all_weights.min().item(), all_weights.max().item()
        delta = (w_max - w_min) / (2 ** n_bits - 1)
        sigma_q_sq = (delta ** 2) / 12.0

        # Cache clean reference logits.
        f_star_train = self.compute_f_star(
            model, X_train, batch_size=forward_batch_size
        ).detach()
        f_star_test = (
            self.compute_f_star(model, X_test, batch_size=forward_batch_size).detach()
            if X_test is not None
            else None
        )

        # Cache clean parameter values once, indexed by parameter object so
        # we can restore in place without going through state_dict.
        clean_params = [
            (p, p.detach().clone()) for p in model.parameters() if p.requires_grad
        ]

        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(int(seed))
        else:
            # Use a fresh per-call seed so repeated calls don't lock-step.
            gen.seed()

        tpv_train_per_run = []
        tpv_test_per_run = [] if X_test is not None else None

        try:
            for _ in range(n_runs):
                # 1. Draw δw ~ Unif(-δ/2, δ/2) per coordinate and write
                #    w* + δw into the live parameters.
                with torch.no_grad():
                    for p, w_clean in clean_params:
                        noise = (
                            torch.rand(p.shape, generator=gen, device=self.device)
                            - 0.5
                        ) * delta
                        p.copy_(w_clean + noise)

                # 2. Forward on train (and test) under the perturbed weights.
                #    Aggregate per the vector-output TPV convention (App. B):
                #    TPV(X) = E_x[||f_{w*+δw}(x) − f_*(x)||²], i.e. sum over
                #    output dimensions, mean over samples. This is what
                #    matches σ_q² · Tr(H_eff) under the linear approximation.
                preds_train = self.compute_f_star(
                    model, X_train, batch_size=forward_batch_size
                )
                diff_train = preds_train - f_star_train
                if diff_train.ndim == 1:
                    sq_train = diff_train.pow(2)
                else:
                    sq_train = diff_train.pow(2).sum(dim=-1)
                tpv_train_per_run.append(float(sq_train.mean().item()))
                if X_test is not None:
                    preds_test = self.compute_f_star(
                        model, X_test, batch_size=forward_batch_size
                    )
                    diff_test = preds_test - f_star_test
                    if diff_test.ndim == 1:
                        sq_test = diff_test.pow(2)
                    else:
                        sq_test = diff_test.pow(2).sum(dim=-1)
                    tpv_test_per_run.append(float(sq_test.mean().item()))

                # 3. Restore w* before the next draw.
                with torch.no_grad():
                    for p, w_clean in clean_params:
                        p.copy_(w_clean)
        finally:
            # Defensive: even if something raised mid-loop, make sure the
            # caller's model is left at w*.
            with torch.no_grad():
                for p, w_clean in clean_params:
                    p.copy_(w_clean)

        result = {
            "tpv_train": float(sum(tpv_train_per_run) / len(tpv_train_per_run)),
            "tpv_train_per_run": tpv_train_per_run,
            "n_runs": n_runs,
            "delta": delta,
            "sigma_q_sq": sigma_q_sq,
        }
        if X_test is not None:
            result["tpv_test"] = float(sum(tpv_test_per_run) / len(tpv_test_per_run))
            result["tpv_test_per_run"] = tpv_test_per_run
        else:
            result["tpv_test"] = None
            result["tpv_test_per_run"] = None

        return result