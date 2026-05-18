import numpy as np
import torch

from tpv.tpv_base import TPVBase


class LabelTPV(TPVBase):
    """
    TPV estimator for label noise, computed empirically via Monte Carlo.

    For each of R runs:
      1. Sample eps ~ N(0, noise_std²) and form noisy logit targets = f_star + eps
      2. Initialize a fresh model from base_state_dict
      3. Fine-tune using train_fn (MSE on noisy logits)
      4. Record predictions on train / test sets

    TPV = E_{runs, x}[||z_r(x) - z*(x)||²]
    """

    def compute_tpv(
        self,
        model_factory,
        base_state_dict,
        X_train,
        X_test,
        noise_std,
        R,
        train_fn,
        loader_seed=12345,
        batch_size=None,
    ):
        """
        Compute empirical TPV for label noise.

        Args:
            model_factory:         callable () -> nn.Module (fresh uninitialized model)
            base_state_dict:       clean reference weights w* (OrderedDict from state_dict())
            X_train:               training inputs (N, ...) on device
            X_test:                test inputs (M, ...) on device
            noise_std:             std of Gaussian noise added to logits each run
            R:                     number of Monte Carlo runs
            train_fn:              callable (model, X_train, y_noisy_logits) -> None
                                   caller controls optimizer, epochs, proximity penalty.
                                   Only label noise differs between runs; everything else
                                   should be deterministic (fixed seeds, no augmentation).
            loader_seed:           fixed seed for DataLoader shuffling so mini-batch order
                                   is identical across runs (only noise differs)
            batch_size:            batch size for all model forward passes (f_star
                                   computation and per-run prediction collection).
                                   None processes the full tensor in one shot (original
                                   behaviour, fine for small datasets). Set to a finite
                                   value when X_train / X_test are too large to fit
                                   through the model at once.

        Returns:
            dict:
                empirical_TPV_train: float
                empirical_TPV_test:  float
        """
        torch.manual_seed(loader_seed)

        preds_train = []
        preds_test = []

        # Resolve effective batch sizes: None → full tensor in one shot.
        _bs_train = batch_size if batch_size is not None else len(X_train)
        _bs_test  = batch_size if batch_size is not None else len(X_test)

        model = model_factory()
        model.load_state_dict(base_state_dict)
        model = model.to(self.device)
        f_star_train_logits = self.compute_f_star(model, X_train, batch_size=_bs_train)
        f_star_test_logits  = self.compute_f_star(model, X_test,  batch_size=_bs_test)

        f_star_train_np = f_star_train_logits.cpu().numpy()
        f_star_test_np = f_star_test_logits.cpu().numpy()

        logit_scale = f_star_train_logits.abs().mean().item()

        for _ in range(R):
            model = model_factory()
            model.load_state_dict(base_state_dict)
            model = model.to(self.device)

            # Noisy logit targets — only stochastic element between runs
            eps = torch.randn_like(f_star_train_logits) * noise_std * logit_scale
            y_noisy = f_star_train_logits + eps

            train_fn(model, X_train, y_noisy)

            preds_train.append(self.compute_f_star(model, X_train, batch_size=_bs_train).cpu().numpy())
            preds_test.append( self.compute_f_star(model, X_test,  batch_size=_bs_test).cpu().numpy())

        tpv_train = self.compute_tpv_from_runs(preds_train, f_star_train_np)
        tpv_test = self.compute_tpv_from_runs(preds_test, f_star_test_np)

        return dict(empirical_TPV_train=tpv_train, empirical_TPV_test=tpv_test)
