import abc
import warnings
import torch
import torch.nn as nn
from torch_pruning import importance 
from torch_pruning.pruner import function
import typing
from torch_pruning.dependency import Group
import torch_pruning as tp


import numpy as np
from scipy.spatial import distance
from copy import deepcopy


class GroupJacobianImportance_accumulate(importance.GroupMagnitudeImportance):
    """
    Memory-efficient variant:

    Instead of storing all Jacobian rows J (one per backward call) in self._jacobian
    and later computing w^T J^T J w, we stream the computation:

      For each backward pass (with gradient g for layer.weight and layer.bias),
      we compute the batch contribution to:
          sum_b (j_b w)^2
      and accumulate it directly into self._importance[layer_name + '_out'/'_in'].

    This avoids keeping J in memory and only keeps per-layer 1D importance vectors.

    IMPORTANT: For exact equivalence to the original implementation, the weights
    must be frozen while you call `accumulate_grad` multiple times and only
    `accumulate_score` at the end. If you keep updating weights in between, this
    version will no longer match w^T J^T J w with the *final* w.
    """

    def __init__(
        self,
        group_reduction: str = "mean",
        normalizer: str = "mean",  # 'parameters' or anything supported by GroupMagnitudeImportance
        bias: bool = False,
        target_types: list = [
            nn.modules.conv._ConvNd,
            nn.Linear,
            nn.modules.batchnorm._BatchNorm,
            nn.LayerNorm,
        ],
    ):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias

        # We no longer store full Jacobians
        self._jacobian = {}
        self._counter = {}  # kept for API compatibility, but unused
        self._importance = {}
        self._memory = {}

    def zero_grad(self):
        # For compatibility with the original API; does nothing meaningful now.
        self._jacobian = {}
        self._counter = {}

    def zero_score(self):
        self._importance = {}

    def _accumulate_conv_linear(self, layer_name: str, layer: nn.Module):
        """
        Streaming accumulation for Conv/Linear layers:
        - Out-channel importance: sum_b (g_w · w + g_b * b)^2 per out-channel
        - In-channel importance:  same idea but over incoming channels and transposed weights
        """
        if layer.weight.grad is None:
            warnings.warn(
                f"The gradient of {layer_name}.weight is None; please do loss.backward() first; "
                f"otherwise no score for {layer_name} prune_xx_out_channels",
                UserWarning,
            )
            return

        if hasattr(layer, "transposed") and layer.transposed:
            # Not supported in original code either
            raise NotImplementedError(
                "Transposed convolutions are not supported in GroupJacobianImportance_accumulate."
            )

        # ---- Out-channels ----
        # weight: (o, ihw) for conv or (o, i) for fc after flatten
        w = layer.weight.data.flatten(1)  # (o, D)
        g_w = layer.weight.grad.data.flatten(1)  # (o, D)

        # Contribution from weight term
        jw = (g_w * w).sum(dim=1)  # (o,)

        # If bias exists, include it as extra dimension as in original code
        if hasattr(layer, "bias") and layer.bias is not None and layer.bias.grad is not None:
            b = layer.bias.data  # (o,)
            g_b = layer.bias.grad.data  # (o,)
            jw = jw + g_b * b  # (o,)

        local_imp_out = jw * jw  # (o,)

        key_out = layer_name + "_out"
        if key_out not in self._importance:
            self._importance[key_out] = local_imp_out.data.clone()
        else:
            self._importance[key_out] = self._importance[key_out] + local_imp_out.data.clone()

        # ---- In-channels ----
        # Same idea but with transposed view (i, ohw) or (i, o)
        w_in = layer.weight.data.transpose(0, 1).flatten(1)  # (i, D)
        g_w_in = layer.weight.grad.data.transpose(0, 1).flatten(1)  # (i, D)

        jw_in = (g_w_in * w_in).sum(dim=1)  # (i,)
        local_imp_in = jw_in * jw_in  # (i,)

        key_in = layer_name + "_in"
        if key_in not in self._importance:
            self._importance[key_in] = local_imp_in.clone()
        else:
            self._importance[key_in] = self._importance[key_in] + local_imp_in.clone()

    def _accumulate_batchnorm(self, layer_name: str, layer: nn.modules.batchnorm._BatchNorm):
        if not layer.affine:
            return
        if layer.weight.grad is None or layer.bias.grad is None:
            warnings.warn(
                f"The gradient of {layer_name}.weight or .bias is None; "
                f"please do loss.backward() first; otherwise no score for "
                f"{layer_name} prune_batchnorm_out_channels",
                UserWarning,
            )
            return

        # w, b : (o,)
        w = layer.weight.data
        b = layer.bias.data
        g_w = layer.weight.grad.data
        g_b = layer.bias.grad.data

        # For each channel: dot([gw, gb], [w, b]) = gw*w + gb*b
        jw = g_w * w + g_b * b  # (o,)
        local_imp = jw * jw  # (o,)

        key_out = layer_name + "_out"
        if key_out not in self._importance:
            self._importance[key_out] = local_imp.clone()
        else:
            self._importance[key_out] = self._importance[key_out] + local_imp.clone()

    def _accumulate_layernorm(self, layer_name: str, layer: nn.LayerNorm):
        if not layer.elementwise_affine:
            return
        if layer.weight.grad is None or layer.bias.grad is None:
            warnings.warn(
                f"The gradient of {layer_name}.weight or .bias is None; "
                f"please do loss.backward() first; otherwise no score for "
                f"{layer_name} prune_layernorm_out_channels",
                UserWarning,
            )
            return

        w = layer.weight.data
        b = layer.bias.data
        g_w = layer.weight.grad.data
        g_b = layer.bias.grad.data

        jw = g_w * w + g_b * b  # (o,)
        local_imp = jw * jw  # (o,)

        key_out = layer_name + "_out"
        if key_out not in self._importance:
            self._importance[key_out] = local_imp.clone()
        else:
            self._importance[key_out] = self._importance[key_out] + local_imp.clone()

    def accumulate_grad(self, model, transposed: bool = False):
        """
        Streaming accumulation of J^T J in the quadratic form w^T J^T J w,
        without storing J explicitly.

        model: the model from which gradients are read (after loss.backward()).
        transposed: kept for API compatibility; transposed conv not supported.
        """
        assert transposed is False

        for layer_name, layer in model.named_modules():
            if not isinstance(layer, tuple(self.target_types)):
                continue

            # Conv / Linear
            if isinstance(layer, (nn.modules.conv._ConvNd, nn.Linear)):
                self._accumulate_conv_linear(layer_name, layer)

            # BatchNorm
            elif isinstance(layer, nn.modules.batchnorm._BatchNorm):
                self._accumulate_batchnorm(layer_name, layer)

            # LayerNorm
            elif isinstance(layer, nn.LayerNorm):
                self._accumulate_layernorm(layer_name, layer)

        # We no longer store gradients across calls, so zero_grad is trivial
        self.zero_grad()

    def accumulate_score(self, model):
        """
        In the streaming variant, all the work is already done in accumulate_grad.
        This is kept only for API compatibility with the original implementation.
        """
        # Nothing to do; scores are already in self._importance.
        return

    @torch.no_grad()
    def __call__(self, group):
        group_imp = []
        group_idxs = []
        group_parameter_numel = 0  # total params per group

        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            layer_name = dep.target.name[: dep.target.name.find(" ")]
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs

            if not isinstance(layer, tuple(self.target_types)):
                continue

            # Conv/Linear out_channels
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed:
                        raise NotImplementedError(
                            "Transposed convolutions are not supported."
                        )
                    local_imp = self._importance[layer_name + "_out"][idxs]
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    group_parameter_numel += (
                        layer.weight.data.numel() / layer.weight.data.shape[0]
                    )

            # Conv/Linear in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed:
                        raise NotImplementedError(
                            "Transposed convolutions are not supported."
                        )
                    local_imp = self._importance[layer_name + "_in"][idxs]

                    # Same special case as original implementation
                    if (
                        prune_fn == function.prune_conv_in_channels
                        and layer.groups != layer.in_channels
                        and layer.groups != 1
                    ):
                        local_imp = local_imp.repeat(layer.groups)

                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    group_parameter_numel += (
                        layer.weight.data.numel() / layer.weight.data.shape[1]
                    )

            # BN
            elif prune_fn == function.prune_batchnorm_out_channels:
                if layer.affine and layer.weight.grad is not None:
                    local_imp = self._importance[layer_name + "_out"][idxs]
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    group_parameter_numel += layer.weight.numel() * 2  # w and bias

            # LN
            elif prune_fn == function.prune_layernorm_out_channels:
                if layer.elementwise_affine and layer.weight.grad is not None:
                    local_imp = self._importance[layer_name + "_out"][idxs]
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    group_parameter_numel += layer.weight.numel() * 2  # w and bias

        if len(group_imp) == 0:  # skip groups without parameterized layers
            return None

        group_imp = self._reduce(group_imp, group_idxs)

        if self.normalizer == "parameters":
            group_imp = group_imp / group_parameter_numel
        else:
            group_imp = self._normalize(group_imp, self.normalizer)

        return group_imp



# define a class called GroupJBR that inherits everything from GroupJacobianImportance_accumulate
class GroupJBRImportance_accumulate(GroupJacobianImportance_accumulate):
    def __init__(self, 
                 group_reduction:str="mean", 
                 normalizer:str='mean',  # 'num_parameters'
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm, nn.modules.LayerNorm]):
        super().__init__(group_reduction, normalizer, bias, target_types)
        print('Using GroupJBRImportance_accumulate for JBR importance scoring.')






class WHCImportance(tp.importance.GroupMagnitudeImportance):

    def __init__(self, p=2, group_reduction="mean", normalizer='mean', bias=False):
        super().__init__(p=p, group_reduction=group_reduction, normalizer=normalizer, bias=bias)

    @torch.no_grad()
    def __call__(self, group, **kwargs):
        group_imp = []
        group_idxs = []
        # Iterate over all groups and estimate group importance
        for i, (dep, idxs) in enumerate(group):
            layer = dep.layer
            prune_fn = dep.pruning_fn
            root_idxs = group[i].root_idxs
            if not isinstance(layer, tuple(self.target_types)):
                continue
            ####################
            # Conv/Linear Output
            ####################
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)

                # L2 norm
                l2_norm = w.norm(p=2, dim=1)
                l2_norm_diag = torch.diag(l2_norm)

                # Cosine similarity matrix
                w_normed = torch.nn.functional.normalize(w, p=2, dim=1)
                similar_matrix = 1 - torch.abs(torch.matmul(w_normed, w_normed.T))  # Cosine similarity

                # Multiply with L2 norm diagonal matrix
                similar_matrix = l2_norm_diag @ similar_matrix @ l2_norm_diag
                similar_sum = similar_matrix.sum(dim=0)

                group_imp.append(similar_sum)
                group_idxs.append(root_idxs)

            ####################
            # Conv/Linear Input
            ####################
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.flatten(1)
                else:
                    w = layer.weight.data.transpose(0, 1).flatten(1)

                # L2 norm
                w = w.cpu()
                l2_norm = w.norm(p=2, dim=1)
                l2_norm_diag = torch.diag(l2_norm)

                # Cosine similarity matrix
                w_normed = torch.nn.functional.normalize(w, p=2, dim=1)
                similar_matrix = 1 - torch.abs(torch.matmul(w_normed, w_normed.T))  # Cosine similarity

                # Multiply with L2 norm diagonal matrix
                similar_matrix = l2_norm_diag @ similar_matrix @ l2_norm_diag
                local_imp = similar_matrix.sum(dim=0)

                # repeat importance for group convolutions
                if prune_fn == function.prune_conv_in_channels and layer.groups != layer.in_channels and layer.groups != 1:
                    local_imp = local_imp.repeat(layer.groups)

                local_imp = local_imp[idxs]
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

        if len(group_imp) == 0:  # skip groups without parameterized layers
            return None

        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp