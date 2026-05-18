import copy

import torch
import torch.nn as nn

from tpv.tpv_base import TPVBase


class SgdTPV(TPVBase):
    """
    TPV estimator for SGD noise.

    Two estimators are provided:

    * `compute_tpv` — analytic: TPV_sgd ≈ (lr / (2 * batch_size)) * Tr(∇²L(w*)).
      The scaling factor comes from the SGD stationary-distribution (SDE)
      approximation: near convergence, SGD noise covariance C_sgd ∝
      (lr / batch_size) · H, so Tr(H_eff · C_sgd) ∝ (lr / (2 · batch_size)) · Tr(∇²L).
      Tr(∇²L) is estimated via the Hutchinson trace estimator inherited from TPVBase.

    * `compute_tpv_empirical` — trajectory: simulate SGD around w* with an L2
      proximity penalty and average E_x[(f_w(x) − f_*(x))²] over post-burn-in
      snapshots. Bypasses the SDE/second-order approximation.
    """

    def compute_tpv(
        self,
        model,
        X,
        y,
        lr,
        batch_size,
        n_hutchinson_samples=200,
    ):
        """
        Compute TPV for SGD noise.

        Args:
            model:                 trained model at w* (nn.Module, eval-mode)
            X:                     inputs (N, ...)
            y:                     targets (N,) long or (N, K) float
            lr:                    SGD learning rate used during training
            batch_size:            mini-batch size used during training
            n_hutchinson_samples:  number of Rademacher vectors for Hutchinson estimator

        Returns:
            tpv:     float — TPV estimate: (lr / (2 * batch_size)) * Tr(∇²L)
            trace_H: float — raw Hutchinson trace Tr(∇²L(w*))
        """

        trace_H = self.compute_hutchinson_trace_heff(model, X,
                                                     n_samples=n_hutchinson_samples)
        noise_var = lr / (2.0 * batch_size)
        return noise_var * trace_H, trace_H

    def compute_tpv_empirical(
        self,
        model,
        X_train,
        y_train,
        X_test=None,
        lr=1e-3,
        batch_size=128,
        momentum=0.9,
        sgd_steps=1000,
        burn_in=200,
        snapshot_every=20,
        center_lambda=1e-3,
        loss_fn=None,
        loader_seed=12345,
    ):
        """
        Empirical SGD-noise TPV via trajectory snapshots around w*.

        Runs SGD with momentum starting from the supplied (clean) model,
        anchored to w* by an L2 proximity penalty (strength `center_lambda`).
        After `burn_in` steps, every `snapshot_every` steps records
        E_x[(f_w(x) - f_*(x))^2] on X_train (and X_test, if given). The
        returned TPV is the mean across snapshots.

        Bypasses the second-order/SDE approximation used by `compute_tpv`
        and matches the procedure in `estimate_tpv_sgd_noise` in
        investigation/tpv_trace_synth_universal_scatter.py.

        Args:
            model:           trained model at w* (nn.Module)
            X_train, y_train: training data on self.device
            X_test:          optional test inputs; if given, also returns test TPV
            lr:              SGD learning rate
            batch_size:      mini-batch size for the trajectory
            momentum:        SGD momentum (0.9 matches the reference script)
            sgd_steps:       total number of SGD steps to run around w*
            burn_in:         steps to discard before sampling snapshots
            snapshot_every:  sampling cadence after burn-in
            center_lambda:   L2 proximity-penalty strength around w*; 0 disables
            loss_fn:         callable (logits, targets) -> scalar.
                             Auto-detected from y_train.dtype if None.
            loader_seed:     fixed seed for the DataLoader generator

        Returns:
            dict with:
                tpv_train:               float
                tpv_test:                float | None
                tpv_train_per_snapshot:  list[float]
                tpv_test_per_snapshot:   list[float] | None
                n_snapshots:             int
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss() if y_train.dtype == torch.long else nn.MSELoss()

        center_state = copy.deepcopy(model.state_dict())

        f_star_train = self.compute_f_star(model, X_train).detach()
        f_star_test = (
            self.compute_f_star(model, X_test).detach() if X_test is not None else None
        )

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=torch.Generator().manual_seed(loader_seed),
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        tpv_train_per_snap = []
        tpv_test_per_snap = [] if X_test is not None else None

        data_iter = iter(loader)
        step = 0
        while step < sgd_steps:
            try:
                X_batch, y_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                X_batch, y_batch = next(data_iter)

            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)

            if center_lambda > 0.0:
                loss = loss + center_lambda * self.compute_proximity_penalty(model, center_state)

            loss.backward()
            optimizer.step()

            step += 1

            if step >= burn_in and (step - burn_in) % snapshot_every == 0:
                train_preds = self.compute_f_star(model, X_train)
                tpv_train_per_snap.append(
                    float((train_preds - f_star_train).pow(2).mean().item())
                )
                if X_test is not None:
                    test_preds = self.compute_f_star(model, X_test)
                    tpv_test_per_snap.append(
                        float((test_preds - f_star_test).pow(2).mean().item())
                    )

        model.load_state_dict(center_state)

        if not tpv_train_per_snap:
            raise RuntimeError(
                "No snapshots collected; adjust burn_in / snapshot_every / sgd_steps."
            )

        result = {
            "tpv_train": float(sum(tpv_train_per_snap) / len(tpv_train_per_snap)),
            "tpv_train_per_snapshot": tpv_train_per_snap,
            "n_snapshots": len(tpv_train_per_snap),
        }
        if X_test is not None:
            result["tpv_test"] = float(sum(tpv_test_per_snap) / len(tpv_test_per_snap))
            result["tpv_test_per_snapshot"] = tpv_test_per_snap
        else:
            result["tpv_test"] = None
            result["tpv_test_per_snapshot"] = None

        return result
