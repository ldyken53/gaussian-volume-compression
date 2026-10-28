"""Densification: when to grow the model, by how much, and where the new Gaussians go.

Three strategies: sample-error-based (default; new Gaussians at samples drawn with probability
~ error^alpha), 3DGS clone/split by position gradient, and MCMC relocation.
"""

import math

import torch
from simple_knn._C import distCUDA2

from veggies.model import NEW_WEIGHT, VALUE_EPS
from veggies.utils import build_rotation, build_scaling_rotation, inverse_sigmoid


def event_count(n0, n_final, iterations, densify_from_iter, interval, growth):
    """(events, latest): enough events that each grows the model by about `growth`, at least 3,
    done by half-way so the last ones still train."""
    latest = max(3, (iterations // 2 - densify_from_iter) // interval)
    want = math.ceil(math.log(max(n_final, n0) / n0) / math.log(growth))
    return int(min(latest, max(3, want))), latest


def planned_events(model, n_final, args):
    if args.densify_growth <= 1.0:
        return 3
    n0 = max(1, model.count)
    events, _ = event_count(
        n0,
        n_final,
        args.iterations,
        args.densify_from_iter,
        args.densification_interval,
        args.densify_growth,
    )
    print(f"densify_events {events} ({n0:,} -> {n_final:,} Gaussians)")
    return events


def budget(model, n_final, prune_threshold):
    """Gaussians still to place: those missing from n_final plus the dead ones."""
    return int(
        n_final - model.count + torch.count_nonzero(model.weights < prune_threshold)
    )


def share(budget, events_left):
    return min(-(-budget // max(1, events_left)), budget)


def pick(err, k, alpha):
    """k samples drawn without replacement with probability ~ err^alpha (Gumbel top-k); alpha <= 0
    takes the plain top k."""
    if alpha <= 0:
        return torch.topk(err, k).indices
    logits = alpha * torch.log(err + 1e-12)
    u = torch.rand_like(err).clamp_min(1e-20)
    return torch.topk(logits - torch.log(-torch.log(u)), k).indices


def run_event(model, args, data, values, gt, n_final, events_left, remove=None):
    """One densification event. remove: indices of Gaussians to remove, set below the prune
    threshold so that the strategy treats them as dead and re-places their slots."""
    dead = args.prune_threshold
    if remove is not None and remove.numel() > 0:
        model._weights.data[remove] = inverse_sigmoid(
            torch.tensor(dead * 0.1, device=model._weights.device)
        )
    left = budget(model, n_final, dead)
    events_left = max(1, events_left)
    if args.densify == "mcmc":
        mcmc_relocate(model, (model.weights < dead).squeeze(-1))
        mcmc_add(model, args.cap_max)
    elif args.densify == "3dgs":
        k = max(0, share(left, events_left))
        grads = model._means.grad
        if k > 0 and grads is not None:
            cloned = clone(model, grads, args.densify_grad_threshold, 1.0, max_new=k)
            rest = max(0, k - int(cloned or 0))
            if rest > 0:
                split(model, grads, args.densify_grad_threshold, 1.0, max_new=rest)
        model.prune((model.weights < dead).squeeze(-1))
    else:
        defined = gt.ravel() != -1
        err = torch.abs(values.ravel()[defined] - gt.ravel()[defined])
        k = max(0, min(share(left, events_left), err.numel()))
        if k > 0:
            chosen = pick(err, k, args.densify_alpha)
            idx = torch.nonzero(defined, as_tuple=False).squeeze(1)[chosen]
            at_samples(model, data.positions(idx), gt.ravel()[idx].reshape(-1, 1), dead)


# Sample-error-based.


def at_samples(model, points, values, prune_threshold, new_weight=NEW_WEIGHT):
    """New Gaussians at points with the given values; they take the dead Gaussians' rows, and
    dead ones left over are pruned."""
    new = _new_gaussians(model, points, values, new_weight)
    k = points.shape[0]
    slots = torch.nonzero((model.weights < prune_threshold).squeeze(-1)).squeeze(-1)
    m = min(k, slots.numel())
    if m > 0:
        model.overwrite(slots[:m], {n: v[:m] for n, v in new.items()})
    if k > m:
        model.append(
            *(new[n][m:] for n in ("means", "weights", "scales", "rotations", "values"))
        )
    elif slots.numel() > m:
        mask = torch.zeros(model.count, dtype=torch.bool, device=model._means.device)
        mask[slots[m:]] = True
        model.prune(mask)


def _new_gaussians(model, points, values, new_weight):
    dist2 = torch.clamp_min(_neighbour_dist2(model, points) * 0.1, 0.0000001)
    n = points.shape[0]
    rotations = torch.zeros((n, 4), device="cuda")
    rotations[:, 0] = 1
    return {
        "means": points,
        "weights": inverse_sigmoid(
            new_weight * torch.ones((n, 1), dtype=torch.float, device="cuda")
        ),
        "scales": torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3),
        "rotations": rotations,
        "values": inverse_sigmoid(torch.clamp(values, VALUE_EPS, 1.0 - VALUE_EPS)),
    }


def _neighbour_dist2(model, new_points):
    """Mean squared distance to the 3 nearest other points, for the new points only (distCUDA2
    over the whole set, without recomputing the existing points)."""
    old = model._means.detach()
    n_old, k = old.shape[0], new_points.shape[0]

    def full():
        return distCUDA2(torch.cat([old, new_points], dim=0))[n_old:]

    if k == 0 or 8 * k > n_old:
        return full()
    dev = old.device
    pts = torch.cat([old, new_points], dim=0)
    P = pts.shape[0]
    lo = pts.min(dim=0).values
    ext = (pts.max(dim=0).values - lo).clamp_min(1e-12)
    spacing = float((torch.prod(ext) / P) ** (1.0 / 3.0))
    offsets = torch.stack(
        torch.meshgrid(*[torch.arange(-1, 2, device=dev)] * 3, indexing="ij"), -1
    ).reshape(-1, 3)
    out = torch.empty(k, device=dev)
    todo = torch.arange(k, device=dev)
    for R in (2.0 * spacing, 4.0 * spacing):
        G = (ext / R).floor().long() + 1
        c = ((pts - lo) / R).floor().long().clamp_min(0).minimum(G - 1)
        lin = (c[:, 2] * G[1] + c[:, 1]) * G[0] + c[:, 0]
        order = torch.argsort(lin)
        counts = torch.bincount(lin, minlength=int(torch.prod(G)))
        starts = torch.cumsum(counts, 0) - counts
        chunk = max(1024, int(6e7 // (27 * max(1, int(counts.max())))))
        unresolved = []
        for ch in torch.split(todo, chunk):
            raw = c[n_old + ch][:, None, :] + offsets[None]
            inside = ((raw >= 0) & (raw < G)).all(-1)
            qc = raw.clamp_min(0).minimum(G - 1)
            nb = (qc[..., 2] * G[1] + qc[..., 1]) * G[0] + qc[..., 0]
            cnt = counts[nb] * inside
            M = int(cnt.max())
            if M == 0:
                unresolved.append(ch)
                continue
            ar = torch.arange(M, device=dev)
            slot = (starts[nb][..., None] + ar).clamp(max=P - 1)
            idx = order[slot].reshape(ch.shape[0], -1)
            valid = (ar < cnt[..., None]).reshape(ch.shape[0], -1)
            valid &= idx != (n_old + ch)[:, None]
            d2 = ((pts[idx] - new_points[ch][:, None, :]) ** 2).sum(-1)
            d2 = d2.masked_fill(~valid, float("inf"))
            if d2.shape[1] < 3:
                unresolved.append(ch)
                continue
            best = torch.topk(d2, 3, dim=1, largest=False, sorted=True).values
            ok = best[:, 2] <= R * R
            out[ch[ok]] = (best[ok, 0] + best[ok, 1] + best[ok, 2]) / 3.0
            unresolved.append(ch[~ok])
        todo = torch.cat(unresolved)
        if todo.numel() == 0:
            return out
    out[todo] = full()[todo]
    return out


# 3DGS clone/split.


def _select_by_grad(model, grads, threshold, large, extent, max_parents):
    """Gaussians whose position gradient reaches threshold and that are larger (split) or not
    (clone) than percent_dense * extent; the max_parents largest gradients if given."""
    grad_mag = torch.zeros(model.count, device=model._means.device)
    grad_mag[: grads.shape[0]] = torch.linalg.norm(grads, dim=-1)
    size = torch.max(model.scales, dim=1).values
    limit = model.percent_dense * extent
    mask = (grad_mag >= threshold) & (size > limit if large else size <= limit)
    idx = mask.nonzero(as_tuple=True)[0]
    if max_parents is not None:
        if max_parents <= 0 or idx.numel() == 0:
            return None
        if idx.numel() > max_parents:
            idx = idx[torch.topk(grad_mag[idx], k=max_parents, sorted=False).indices]
        mask = torch.zeros_like(mask)
        mask[idx] = True
    return mask


def split(model, grads, threshold, extent, n=2, max_new=None):
    mask = _select_by_grad(
        model,
        grads,
        threshold,
        True,
        extent,
        None if max_new is None else int(max_new) // n,
    )
    if mask is None or n * mask.sum().item() == 0:
        return 0
    added = n * mask.sum().item()
    stds = model.scales[mask].repeat(n, 1)
    samples = torch.normal(
        mean=torch.zeros((stds.size(0), 3), device=model._means.device), std=stds
    )
    rotations = build_rotation(model._rotations[mask]).repeat(n, 1, 1)
    model.append(
        torch.bmm(rotations, samples.unsqueeze(-1)).squeeze(-1)
        + model.means[mask].repeat(n, 1),
        model._weights[mask].repeat(n, 1),
        torch.log(model.scales[mask].repeat(n, 1) / (0.8 * n)),
        model._rotations[mask].repeat(n, 1),
        model._values[mask].repeat(n, 1),
    )
    model.prune(
        torch.cat(
            (
                mask,
                torch.zeros(
                    n * mask.sum(), device=model._means.device, dtype=torch.bool
                ),
            )
        )
    )
    return added


def clone(model, grads, threshold, extent, max_new=None):
    mask = _select_by_grad(
        model,
        grads,
        threshold,
        False,
        extent,
        None if max_new is None else int(max_new),
    )
    if mask is None or mask.sum().item() == 0:
        return 0
    model.append(
        model._means[mask],
        model._weights[mask],
        model._scales[mask],
        model._rotations[mask],
        model._values[mask],
    )
    return mask.sum().item()


# MCMC relocation.


def _relocated(model, idxs, ratio):
    n = (ratio[idxs, 0] + 1).float()
    weight = torch.clamp(
        (model.weights[idxs, 0] / n).unsqueeze(-1),
        max=1.0 - torch.finfo(torch.float32).eps,
        min=0.005,
    )
    scales = model.scales[idxs]
    rotation = build_rotation(model._rotations[idxs])
    noise = torch.randn_like(scales) * scales * 0.5
    means = model._means[idxs] + torch.bmm(rotation, noise.unsqueeze(-1)).squeeze(-1)
    return (
        means,
        inverse_sigmoid(weight),
        model._scales[idxs],
        model._rotations[idxs],
        model._values[idxs],
    )


def _sample_alive(probs, num, alive=None):
    probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
    picked = torch.multinomial(probs, num, replacement=True)
    if alive is not None:
        picked = alive[picked]
    return picked, torch.bincount(picked).unsqueeze(-1)


def mcmc_relocate(model, dead):
    """Move the dead Gaussians onto live ones sampled by weight, splitting their weight."""
    if dead.sum() == 0:
        return
    dead_idx = dead.nonzero(as_tuple=True)[0]
    alive_idx = (~dead).nonzero(as_tuple=True)[0]
    if alive_idx.shape[0] <= 0:
        return
    source, ratio = _sample_alive(
        model.weights[alive_idx, 0], dead_idx.shape[0], alive_idx
    )
    (
        model._means[dead_idx],
        model._weights[dead_idx],
        model._scales[dead_idx],
        model._rotations[dead_idx],
        model._values[dead_idx],
    ) = _relocated(model, source, ratio)
    model._weights[source] = model._weights[dead_idx]
    model._scales[source] = model._scales[dead_idx]
    changed = torch.cat([source, dead_idx]).unique()
    model.restart_ema(changed)
    model.rewrap(changed)


def mcmc_add(model, cap):
    current = model._weights.shape[0]
    n_new = max(0, min(cap, int(1.5 * current)) - current)
    if n_new <= 0:
        return 0
    source, ratio = _sample_alive(model.weights.squeeze(-1), n_new)
    means, weights, scales, rotations, values = _relocated(model, source, ratio)
    model._weights[source] = weights
    model._scales[source] = scales
    model.restart_ema(source)
    model.append(means, weights, scales, rotations, values)
    model.rewrap(source)
    return n_new


def mcmc_noise(model, noise_lr, means_lr):
    """MCMC's position noise, gated to Gaussians of low weight and shaped by their covariance."""
    L = build_scaling_rotation(model.scales, model.rotations)
    covariance = L @ L.transpose(1, 2)
    gate = 1 / (1 + torch.exp(-100 * ((1 - model.weights) - 0.995)))
    noise = torch.randn_like(model._means) * gate * noise_lr * means_lr
    noise = torch.bmm(covariance, noise.unsqueeze(-1)).squeeze(-1)
    model._means.add_(noise)
