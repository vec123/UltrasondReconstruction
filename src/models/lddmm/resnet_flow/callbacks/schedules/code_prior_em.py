"""EM of a Gaussian prior on the leading code entries (loss.code_prior_em)."""

import json
import os
import time

import numpy as np
import torch

from src.models.lddmm.learning.callbacks.base import Callback


class CodePriorEM(Callback):
    """Every ``every`` steps, fit ``z ~ N(m, C)`` to the codes' Laplace posteriors; make it the code penalty.

    The code is ``[z | w]``. The first ``n_em`` entries (z, read by a frozen prior field,
    whose weights fix the scale of z) get the learned prior; the rest (w, read by the
    trained residual) keep the isotropic prior they started with, so the scale of w
    against the network cannot drift. Per member i, in the units of the training
    objective times the batch size B (data averaged over the batch, penalty summed):

        -log p(c_i | data_i) = data_i + 1/2 (c_i - center)^T Pi (c_i - center),
        Pi = 2 B lambda P     (lambda = code_reg_weight; P = I for the plain L2 at start)

    E-step: a_i = z_i, the current code (training keeps it near its MAP), and
            S_i = (H_i + C^-1)^-1, with H_i the Hessian of data_i in z from central
            differences of the batch gradient. Member i's data term reads only its own
            code, so 2 n_em batch passes give every H_i; each pass draws the same
            subsample. H_i is made symmetric and positive semi-definite.
    M-step: m = sum_i w_i a_i / sum_i w_i,
            C = (nu C0 + sum_i w_i ((a_i - m)(a_i - m)^T + S_i)) / (nu + sum_i w_i),
            with C0 = I / (2 B lambda), the starting prior, and w_i = 1 / (scans of
            member i's fetus) under ``weighting: fetus`` (fetus = name up to the first
            "_"), or 1 under ``weighting: scan``.
    The penalty then has center [m | 0] and precision blockdiag(C^-1 / (2 B lambda), I).

    Runs at the first step, every ``every`` steps after it, and once more at train end,
    so the last prior written matches the final flow and codes. Each run writes
    ``<output_dir>/code_prior_em/step_<k>.npz`` (names, weights, a, H, S, m, C) and
    appends to ``history.json`` there. Needs every member in every batch (batch = cohort).
    """

    def __init__(self, stepper, names, n_em: int, code_reg_weight: float, output_dir: str,
                 every: int = 250, nu: float = 10.0, weighting: str = "fetus", rel_eps: float = 0.1):
        super().__init__(every_n_steps=every)
        if every <= 0:
            raise ValueError(f"code_prior_em.every must be > 0, got {every}")
        if not 0 < n_em <= stepper.code_source.n_z:
            raise ValueError(f"code_prior_em needs 0 < n_em <= n_z={stepper.code_source.n_z}, got {n_em}")
        if code_reg_weight <= 0:
            raise ValueError("code_prior_em needs loss.code_reg_weight > 0: the prior is that penalty")
        if weighting not in ("fetus", "scan"):
            raise ValueError(f"code_prior_em.weighting must be 'fetus' or 'scan', got {weighting!r}")
        if len(names) != stepper.code_source.num_shapes:
            raise ValueError(f"{len(names)} member names for {stepper.code_source.num_shapes} codes")
        self.stepper = stepper
        self.names = list(names)
        self.n_em = int(n_em)
        self.code_reg_weight = float(code_reg_weight)
        self.output_dir = os.path.join(output_dir, "code_prior_em")
        self.every = int(every)
        self.nu = float(nu)
        self.weighting = weighting
        self.rel_eps = float(rel_eps)
        fetuses = [name.split("_")[0] for name in self.names]
        self.weights = (np.array([1.0 / fetuses.count(f) for f in fetuses]) if weighting == "fetus"
                        else np.ones(len(self.names)))
        self.m = self.C = self.C0 = None
        self.last_batch = None
        self.history = []

    def on_step_end(self, ctx, step, metrics, batch, pred):
        self.last_batch = batch
        if (step - ctx.start_step) % self.every == 0:
            self._update(step, batch, metrics)

    def on_train_end(self, ctx):
        if self.last_batch is not None:
            self._update("final", self.last_batch, None)

    def _update(self, step, batch, metrics):
        start = time.time()
        codes = self.stepper.code_source
        table = codes.codes.weight
        sample = batch[1]
        n_members = len(torch.unique(sample.shape_ids.view(-1)))
        if n_members != codes.num_shapes:
            raise ValueError(f"code_prior_em needs every member in every batch: {n_members} of "
                             f"{codes.num_shapes} (set train.batch to the cohort size)")
        scale = 2.0 * sample.shape_ids.numel() * self.code_reg_weight   # Pi = scale * P
        n = self.n_em
        if self.C is None:
            center = codes.penalty_center
            self.m = np.zeros(n) if center is None else center[:n].detach().cpu().double().numpy()
            self.C0 = np.eye(n) / scale
            self.C = self.C0.copy()

        H = self._hessians(batch, step)
        a = table.detach()[:, :n].cpu().double().numpy()
        S = np.linalg.inv(H + np.linalg.inv(self.C)[None])
        S = 0.5 * (S + np.transpose(S, (0, 2, 1)))

        w = self.weights
        m = w @ a / w.sum()
        d = a - m
        C = (self.nu * self.C0 + (w[:, None] * d).T @ d + np.einsum("i,ijk->jk", w, S)) / (self.nu + w.sum())
        C = 0.5 * (C + C.T)

        precision = np.eye(codes.n_z)
        precision[:n, :n] = np.linalg.inv(C) / scale
        center = np.zeros(codes.n_z)
        if codes.penalty_center is not None:
            center[n:] = codes.penalty_center[n:].detach().cpu().double().numpy()
        center[:n] = m
        codes.set_penalty_precision(torch.as_tensor(center, dtype=table.dtype),
                                    torch.as_tensor(precision, dtype=table.dtype))

        os.makedirs(self.output_dir, exist_ok=True)
        tag = step if isinstance(step, str) else f"{step:06d}"
        np.savez(os.path.join(self.output_dir, f"step_{tag}.npz"), names=np.array(self.names),
                 weights=w, a=a, H=H, S=S, m=m, C=C, n_em=n, scale=scale)
        mahalanobis = np.einsum("ij,jk,ik->i", d, np.linalg.inv(C), d)
        record = {"step": step, "seconds": time.time() - start,
                  "trace_C": float(np.trace(C)), "trace_C0": float(np.trace(self.C0)),
                  "logdet_C": float(np.linalg.slogdet(C)[1]),
                  "mean_shift_norm": float(np.linalg.norm(m - self.m)),
                  "mahalanobis_sq_median": float(np.median(mahalanobis)),
                  "posterior_trace_median": float(np.median(np.trace(S, axis1=1, axis2=2)))}
        self.history.append(record)
        with open(os.path.join(self.output_dir, "history.json"), "w") as handle:
            json.dump(self.history, handle, indent=2)
        print(f"[code_prior_em] step {step}: trace C {record['trace_C']:.4g} (start {record['trace_C0']:.4g}), "
              f"mean moved {record['mean_shift_norm']:.4g}, Mahalanobis^2 median "
              f"{record['mahalanobis_sq_median']:.1f} of {n}, {record['seconds']:.0f} s")
        if metrics is not None:
            metrics["code_prior_em/trace_C"] = record["trace_C"]
        self.m, self.C = m, C

    def _hessians(self, batch, step):
        """[members, n_em, n_em] Hessians of each member's data term in z, PSD, float64."""
        table = self.stepper.code_source.codes.weight
        n = self.n_em
        batch_size = batch[1].shape_ids.numel()
        eps = torch.as_tensor(self.rel_eps * np.sqrt(np.diag(self.C)), dtype=table.dtype, device=table.device)
        seed = len(self.history)
        base = table.detach()[:, :n].clone()
        H = np.zeros((table.shape[0], n, n))
        for k in range(n):
            grads = []
            for sign in (1.0, -1.0):
                with torch.no_grad():
                    table[:, k] = base[:, k] + sign * eps[k]
                grads.append(self._data_gradient(batch, seed))
            with torch.no_grad():
                table[:, k] = base[:, k]
            H[:, :, k] = (batch_size * (grads[0] - grads[1]) / (2.0 * eps[k])).double().cpu().numpy()
        H = 0.5 * (H + np.transpose(H, (0, 2, 1)))
        values, vectors = np.linalg.eigh(H)
        return np.einsum("sij,sj,skj->sik", vectors, np.clip(values, 0.0, None), vectors)

    def _data_gradient(self, batch, seed):
        """[members, n_em] gradient of the batch data term in z, on the subsample ``seed`` draws."""
        table = self.stepper.code_source.codes.weight
        devices = [table.device.index or 0] if table.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            _, values = self.stepper._values(*batch)
            grad, = torch.autograd.grad(values["data"], table)
        return grad[:, :self.n_em].detach()
