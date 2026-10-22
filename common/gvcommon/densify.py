"""The error-proportional densification schedule: how many growth events, how many Gaussians
each adds, and which samples get them."""

import math

import torch


def event_count(n0, n_final, iterations, densify_from_iter, interval, growth):
    """(events, latest): enough events that each grows the model by about `growth`, at least 3,
    done by half-way so the last ones still train; latest is the most that fit."""
    latest = max(3, (iterations // 2 - densify_from_iter) // interval)
    want = math.ceil(math.log(max(n_final, n0) / n0) / math.log(growth))
    return int(min(latest, max(3, want))), latest


def budget(gaussians, n_final, dead_weight):
    """Gaussians still to place: those missing from n_final plus the dead (weight < dead_weight)."""
    return int(
        n_final
        - gaussians.get_values.shape[0]
        + torch.count_nonzero(gaussians.get_weight < dead_weight)
    )


def share(budget, events_left):
    """This event's even share of the budget, rounded up."""
    return min(-(-budget // max(1, events_left)), budget)


def pick(err, k, alpha):
    """k samples drawn without replacement with probability ~ err^alpha (Gumbel top-k); plain
    top-k (alpha <= 0) spends the budget on a thin tail."""
    if alpha <= 0:
        return torch.topk(err, k).indices
    logits = alpha * torch.log(err + 1e-12)
    u = torch.rand_like(err).clamp_min(1e-20)
    return torch.topk(logits - torch.log(-torch.log(u)), k).indices
