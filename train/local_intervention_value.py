"""Joint task potentials learned from outcomes and single-vessel contrasts.

All labels come from terminal continuations of a frozen predecessor.  The
potential is a function of a complete intervention, not a sum of local gains.
Local consequences constrain a factual-anchored joint potential. The learned
single-vessel contrasts are antisymmetric; reconstruction makes the final
complete-plan value comparisons cycle consistent.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from itertools import combinations, product
import copy
import hashlib
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def response_basis(context, controls, timing=None, dual=False):
    """Public motion descriptors for all ten boat choices.

    These are input features, not task labels or a collision-cost controller.
    With timing, each candidate command acts only within its finite window;
    the observed factual command applies before and after that window.
    The dual basis retains the sustained-command descriptors alongside it.
    """
    if dual and timing is None:
        raise ValueError('The dual motion basis requires an intervention window.')
    b, n = controls.shape[:2]
    rows = context[:, :8*n].reshape(b, n, 8)
    coefficients = context.new_tensor([[-1., -1.]]+[[a, z] for a in (0., .5, 1.) for z in (0., .5, 1.)])
    source, rule, factual = controls.split(2, -1)
    command = coefficients[None, None]*source[:, :, None]+(1-coefficients[None, None])*rule[:, :, None]
    command[:, :, 0] = factual
    force = 750.*command+250.
    if timing is not None:
        # Intervention clocks count 0.2 s control steps. Each contains the
        # four 0.05 s integrations used by the public WAM-V model.
        window_start = 4*timing[:, 0]
        window_end = 4*timing.sum(-1)
    heading = torch.atan2(rows[..., 5], rows[..., 4])
    state = torch.cat((30.*rows[..., :2], heading[..., None], 3.*rows[..., 2:4], rows[..., 6:7]), -1)
    state = state[:, :, None].expand(-1, -1, 10, -1).clone()
    snapshots = []
    for step in range(80):
        applied = force
        if timing is not None:
            active = (step >= window_start) & (step < window_end)
            applied = torch.where(active[:, None, None, None], force, force[:, :, :1])
        x, y, yaw, vx, vy, angular = state.unbind(-1)
        x = x+.05*vx; y = y+.05*vy; yaw = yaw+.05*angular
        cosine, sine = yaw.cos(), yaw.sin()
        u = cosine*vx+sine*vy; v = -sine*vx+cosine*vy
        du = (400.*angular*v-(100.+150.*u.abs())*u+applied.sum(-1))/380.
        dv = (-380.*angular*u-(100.+150.*v.abs())*v)/400.
        dr = (-20.*u*v-(980.+950.*angular.abs())*angular+1.25*(applied[..., 0]-applied[..., 1]))/1430.
        u = u+.05*du; v = v+.05*dv
        state = torch.stack((x, y, yaw, cosine*u-sine*v, sine*u+cosine*v, angular+.05*dr), -1)
        if step in (19, 39, 79):
            snapshots.append(state)
    path = torch.stack(snapshots, -2)
    target = 5.*context[:, -3:-1]
    time = context.new_tensor((1., 2., 4.))
    position = path[..., :2]-target[:, None, None, None]*time[None, None, None, :, None]
    body = torch.stack((path[..., 2].cos(), path[..., 2].sin()), -1)
    absolute = torch.cat((position/30., path[..., 3:5]/5., body, path[..., 5:6]/2.), -1)
    relative = torch.cat(((path[..., :2]-path[:, :, :1, :, :2])/5.,
        (path[..., 3:5]-path[:, :, :1, :, 3:5])/3., body-body[:, :, :1],
        (path[..., 5:6]-path[:, :, :1, :, 5:6])/2.), -1)
    result = (torch.cat((absolute, relative), -1).flatten(-2), path)
    if dual:
        persistent = response_basis(context, controls)
        result = tuple(torch.cat((finite, sustained), -1)
                       for finite, sustained in zip(result, persistent))
    return result if timing is None else (*result, timing)


def response_pair_geometry(left, right, dual=False):
    if dual:
        return torch.cat((response_pair_geometry(left[..., :6], right[..., :6]),
                          response_pair_geometry(left[..., 6:], right[..., 6:])), -1)
    displacement = left[..., :2]-right[..., :2]
    velocity = left[..., 3:5]-right[..., 3:5]
    separation = displacement.square().sum(-1).clamp_min(1e-8).sqrt()
    approach = (displacement*velocity).sum(-1)/separation
    return torch.cat((displacement/30., velocity/5., separation[..., None]/10.,
                      approach[..., None]/5.), -1).flatten(-2)


class JointTaskPotential(nn.Module):
    def __init__(self, width=64, peer_conditioned=True, output_width=1, context_conditioned=False,
                 local_contrast_head=False, motion_basis=False, residual_potential=False,
                 finite_motion_basis=False, dual_motion_basis=False, identity_conditioned=False):
        super().__init__()
        self.width = width
        self.peer_conditioned = peer_conditioned
        self.context_conditioned = context_conditioned
        self.motion_basis = motion_basis
        self.finite_motion_basis = finite_motion_basis
        self.dual_motion_basis = dual_motion_basis
        self.identity_conditioned = identity_conditioned
        if dual_motion_basis and not finite_motion_basis:
            raise ValueError('The dual motion basis requires finite motion features.')
        if finite_motion_basis and (not motion_basis or local_contrast_head):
            raise ValueError('Finite motion features require the motion basis and a direct task potential.')
        self.residual_potential = residual_potential
        if residual_potential and (output_width != 1 or local_contrast_head):
            raise ValueError('The centered representation defines one shared scalar task potential.')
        self.node = nn.Sequential(nn.Linear(29+5*context_conditioned+42*motion_basis*(1+dual_motion_basis)+identity_conditioned, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.message = nn.Sequential(nn.Linear(2*width+10+18*motion_basis*(1+dual_motion_basis), width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.update = nn.Sequential(nn.Linear(3*width, width), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(3*width+8, width), nn.SiLU(), nn.Linear(width, output_width))
        self.effect_head = (nn.Sequential(nn.Linear(2*(3*width+8), width), nn.SiLU(),
            nn.Linear(width, 1)) if residual_potential else None)
        self.contrast = (nn.Sequential(nn.Linear(3*width+8, width), nn.SiLU(),
            nn.Linear(width, 6*width)) if local_contrast_head else None)
        self.contrast_categorical = local_contrast_head == 'categorical'
        self.contrast_agreement = (nn.Sequential(nn.Linear(3*width+8, width), nn.SiLU(),
            nn.Linear(width, 2)) if self.contrast_categorical else None)
        self.register_buffer('coefficients', torch.tensor([[-1., -1.]] +
            [[a, b] for a in (0., .5, 1.) for b in (0., .5, 1.)]))

    def encode(self, context, controls, programs, timing, responses=None):
        n = programs.shape[-1]
        rows = context[:, :8*n].reshape(-1, n, 8)
        source, rule, factual = controls.split(2, -1)
        active = programs > 0
        coefficients = self.coefficients[programs]
        command = torch.where(active[..., None], coefficients*source+(1-coefficients)*rule, factual)
        features = torch.cat((rows, controls, F.one_hot(programs, 10).to(rows.dtype),
                              command, command-factual, active[..., None].to(rows.dtype)), -1)
        if self.context_conditioned:
            features = torch.cat((features, context[:, None, -5:].expand(-1, n, -1)), -1)
        if self.identity_conditioned:
            if context.shape[-1] != 9*n+5:
                raise ValueError('The task potential requires measured physical vessel identities.')
            features = torch.cat((features, context[:, 8*n:9*n, None]), -1)
        if self.motion_basis:
            if getattr(self, 'finite_motion_basis', False) and (responses is None or len(responses) == 2):
                responses = response_basis(context, controls, timing, getattr(self, 'dual_motion_basis', False))
            elif responses is None:
                responses = response_basis(context, controls)
            batch = torch.arange(len(context), device=context.device)[:, None]
            boats = torch.arange(n, device=context.device)[None]
            selected_response = responses[0][batch, boats, programs]
            features = torch.cat((features, selected_response), -1)
        own = self.node(features)
        if self.peer_conditioned:
            peer = own
        else:
            reference = torch.cat((rows, controls, F.one_hot(torch.zeros_like(programs), 10).to(rows.dtype),
                                   factual, torch.zeros_like(factual), torch.zeros_like(active[..., None], dtype=rows.dtype)), -1)
            if self.context_conditioned:
                reference = torch.cat((reference, context[:, None, -5:].expand(-1, n, -1)), -1)
            if self.identity_conditioned:
                reference = torch.cat((reference, context[:, 8*n:9*n, None]), -1)
            if self.motion_basis:
                reference = torch.cat((reference, responses[0][:, :, 0]), -1)
            peer = self.node(reference)
        difference = rows[:, :, None]-rows[:, None]
        distance = difference[..., :2].square().sum(-1).clamp_min(1e-8).sqrt()
        closing = (difference[..., :2]*difference[..., 2:4]).sum(-1)/distance
        geometry = torch.cat((difference, distance[..., None], closing[..., None]), -1)
        if self.motion_basis:
            own_path = responses[1][batch, boats, programs]
            peer_path = own_path if self.peer_conditioned else responses[1][:, :, 0]
            geometry = torch.cat((geometry, response_pair_geometry(own_path[:, :, None], peer_path[:, None],
                getattr(self, 'dual_motion_basis', False))), -1)
        message = self.message(torch.cat((own[:, :, None].expand(-1, -1, n, -1),
                                         peer[:, None].expand(-1, n, -1, -1), geometry), -1))
        mask = ~torch.eye(n, dtype=torch.bool, device=rows.device)[None, :, :, None]
        encoded = self.update(torch.cat((own, (message*mask).sum(2)/max(n-1, 1),
                                        message.masked_fill(~mask, -1e4).amax(2)), -1))
        changed = (encoded*active[..., None]).sum(1)/active.sum(1, keepdim=True).clamp_min(1)
        # A factual continuation has no intervention duration or delay.
        clock = timing.to(rows.dtype).log1p()/6.*active.any(-1, keepdim=True)
        global_state = torch.cat((context[:, -5:], clock, rows.new_full((len(rows), 1), n/6.)), -1)
        representation = torch.cat((encoded.mean(1), encoded.amax(1), changed, global_state), -1)
        return encoded, representation

    def forward(self, context, controls, programs, timing, responses=None):
        representation = self.encode(context, controls, programs, timing, responses)[1]
        if not self.residual_potential:
            return self.head(representation).squeeze(-1)
        reference = self.encode(context, controls, torch.zeros_like(programs), torch.zeros_like(timing), responses)[1]
        # A complete plan is evaluated relative to its own factual state.
        # The effect vanishes continuously with its physical duration, so a
        # one-tick choice does not inherit a full persistent-maneuver effect.
        effect = self.effect_head(torch.cat((reference, representation-reference), -1)).squeeze(-1)
        exposure = -torch.expm1(-.2*timing[:, 1].to(context.dtype))*(programs > 0).any(-1)
        return self.head(reference).squeeze(-1)+exposure*effect

    def contrast_from_encodings(self, first, second, changed_boat, global_state, return_distribution=False):
        """Predict the two factual-stratum consequences of one boat change.

        Symmetric context selects a signed linear form. Swapping the complete
        endpoint plans therefore reverses the prediction exactly, including
        the effects of the changed boat on its unchanged teammates.
        """
        index = torch.arange(len(first), device=first.device)
        left = torch.cat((first[index, changed_boat], first.mean(1), first.amax(1)), -1)
        right = torch.cat((second[index, changed_boat], second.mean(1), second.amax(1)), -1)
        symmetric = torch.cat(((left+right)*.5, global_state), -1)
        coefficients = self.contrast(symmetric)
        coefficients = coefficients.reshape(-1, 2, 3*self.width)
        signed = (coefficients*(right-left)[:, None]).sum(-1)/(3*self.width)**.5
        if self.contrast_categorical:
            logits = torch.stack((-signed, self.contrast_agreement(symmetric), signed), -1)
            probabilities = logits.softmax(-1)
            delta = probabilities[..., 2]-probabilities[..., 0]
            return (delta, logits) if return_distribution else delta
        return signed.tanh()

    def local_contrasts(self, context, controls, programs, timing, return_distribution=False, responses=None):
        """The batch contains adjacent endpoint pairs with common futures."""
        encoded, representation = self.encode(context, controls, programs, timing, responses)
        pair = programs.reshape(-1, 2, programs.shape[-1])
        changed = pair[:, 0] != pair[:, 1]
        if not bool((changed.sum(-1) == 1).all()):
            raise ValueError('A local comparison must change exactly one vessel.')
        clocks = timing.reshape(-1, 2, 2).amax(1).to(context.dtype).log1p()/6.
        global_state = torch.cat((context[::2, -5:], clocks,
            context.new_full((len(pair), 1), programs.shape[-1]/6.)), -1)
        delta = self.contrast_from_encodings(encoded[::2], encoded[1::2],
            changed.to(torch.long).argmax(-1), global_state, return_distribution)
        return delta, self.head(representation)

    def catalog_contrasts(self, context, encoded, pairs, boats, inverse, clock):
        """Factor the same local comparison across its intervention windows.

        The final coefficient projection is contracted with the signed plan
        features before expanding over windows. This is the same bilinear
        contrast as contrast_from_encodings, without repeating that large
        matrix multiplication for every delay and duration.
        """
        first = encoded[:, pairs[:, 0]]; second = encoded[:, pairs[:, 1]]
        index = torch.arange(len(pairs), device=encoded.device)
        left = torch.cat((first[:, index, boats], first.mean(2), first.amax(2)), -1)
        right = torch.cat((second[:, index, boats], second.mean(2), second.amax(2)), -1)
        global_state = torch.cat((context[:, None, -5:].expand(-1, len(pairs), -1),
            context.new_zeros((len(context), len(pairs), 2)),
            context.new_full((len(context), len(pairs), 1), encoded.shape[2]/6.)), -1)
        symmetric = torch.cat(((left+right)*.5, global_state), -1)
        difference = right-left
        projection = torch.einsum('buf,sfh->bush', difference,
            self.contrast[2].weight.reshape(2, 3*self.width, self.width))
        offset = torch.einsum('buf,sf->bus', difference, self.contrast[2].bias.reshape(2, 3*self.width))
        hidden = self.contrast[1](self.contrast[0](symmetric)[:, inverse]+
            F.linear(clock, self.contrast[0].weight[:, -3:-1])[None])
        signed = ((projection[:, inverse]*hidden[:, :, None]).sum(-1)+offset[:, inverse])/(3*self.width)**.5
        if self.contrast_categorical:
            agreement_hidden = self.contrast_agreement[1](self.contrast_agreement[0](symmetric)[:, inverse]+
                F.linear(clock, self.contrast_agreement[0].weight[:, -3:-1])[None])
            probabilities = torch.stack((-signed, self.contrast_agreement[2](agreement_hidden), signed), -1).softmax(-1)
            return probabilities[..., 2]-probabilities[..., 0]
        return signed.tanh()

    def forward_catalog(self, context, controls, programs, timing, return_encodings=False, responses=None):
        if getattr(self, 'finite_motion_basis', False):
            return self.finite_catalog(context, controls, programs, timing, return_encodings)
        return self.catalog_from_responses(context, controls, programs, timing, return_encodings, responses)

    def finite_catalog(self, context, controls, programs, timing, return_encodings=False):
        """Reuse exact finite-window trajectories for equivalent public clocks."""
        b, n = controls.shape[:2]
        delay = timing[:, 0].clamp(0, 20)
        duration = torch.minimum(timing[:, 1].clamp_min(0), 20-delay)
        clocks, inverse = torch.unique(torch.stack((delay, duration), -1), dim=0, return_inverse=True)
        count = len(clocks)
        expanded_context = context[:, None].expand(-1, count, -1).flatten(0, 1)
        expanded_controls = controls[:, None].expand(-1, count, -1, -1).flatten(0, 1)
        expanded_clocks = clocks[None].expand(b, -1, -1).flatten(0, 1)
        basis = response_basis(expanded_context, expanded_controls, expanded_clocks,
                               getattr(self, 'dual_motion_basis', False))
        return self.catalog_from_responses(context, controls, programs, timing,
            return_encodings, basis, window_inverse=inverse, window_count=count)

    def catalog_from_responses(self, context, controls, programs, timing, return_encodings=False, responses=None,
                               window_inverse=None, window_count=1):
        """Exactly reuse node and pair features across complete joint plans."""
        original_context, original_controls = context, controls
        if window_inverse is not None:
            context = context[:, None].expand(-1, window_count, -1).flatten(0, 1)
            controls = controls[:, None].expand(-1, window_count, -1, -1).flatten(0, 1)
        b, n = controls.shape[:2]; width = self.width
        rows = context[:, :8*n].reshape(b, n, 8)
        source, rule, factual = controls.split(2, -1)
        digits = torch.arange(10, device=context.device)
        active = digits > 0
        command = torch.where(active[None, None, :, None],
            self.coefficients[None, None]*source[:, :, None]+(1-self.coefficients[None, None])*rule[:, :, None],
            factual[:, :, None])
        features = torch.cat((rows[:, :, None].expand(-1, -1, 10, -1),
            controls[:, :, None].expand(-1, -1, 10, -1),
            F.one_hot(digits, 10).to(rows.dtype)[None, None].expand(b, n, -1, -1),
            command, command-factual[:, :, None],
            active.to(rows.dtype)[None, None, :, None].expand(b, n, -1, -1)), -1)
        if self.context_conditioned:
            features = torch.cat((features, context[:, None, None, -5:].expand(-1, n, 10, -1)), -1)
        if self.identity_conditioned:
            if context.shape[-1] != 9*n+5:
                raise ValueError('The task potential requires measured physical vessel identities.')
            features = torch.cat((features, context[:, 8*n:9*n, None, None].expand(-1, -1, 10, -1)), -1)
        if self.motion_basis:
            if responses is None:
                responses = response_basis(context, controls)
            features = torch.cat((features, responses[0]), -1)
        own = self.node(features)
        peer = own if self.peer_conditioned else own[:, :, :1].expand_as(own)
        difference = rows[:, :, None]-rows[:, None]
        distance = difference[..., :2].square().sum(-1).clamp_min(1e-8).sqrt()
        closing = (difference[..., :2]*difference[..., 2:4]).sum(-1)/distance
        geometry = torch.cat((difference, distance[..., None], closing[..., None]), -1)
        geometry = geometry[:, :, :, None, None].expand(-1, -1, -1, 10, 10, -1)
        if self.motion_basis:
            own_path = responses[1]
            peer_path = own_path if self.peer_conditioned else own_path[:, :, :1].expand_as(own_path)
            geometry = torch.cat((geometry, response_pair_geometry(own_path[:, :, None, :, None],
                peer_path[:, None, :, None], getattr(self, 'dual_motion_basis', False))), -1)
        messages = self.message(torch.cat((
            own[:, :, None, :, None].expand(-1, -1, n, -1, 10, -1),
            peer[:, None, :, None].expand(-1, n, -1, 10, -1, -1),
            geometry), -1))
        if window_inverse is not None:
            context, controls = original_context, original_controls
            b = len(context)
            rows = context[:, :8*n].reshape(b, n, 8)
            own = own.reshape(b, window_count, n, 10, width)
            messages = messages.reshape(b, window_count, n, n, 10, 10, width)
        boats = torch.arange(n, device=context.device)
        mask = (~torch.eye(n, dtype=torch.bool, device=context.device))[None, None, :, :, None]
        design = programs if window_inverse is None else torch.cat((window_inverse[:, None], programs), -1)
        distinct, inverse = torch.unique(design, dim=0, return_inverse=True)
        parts = []; encodings = []
        reference = None
        if self.residual_potential:
            factual_responses = (responses if window_inverse is None else
                tuple(item.reshape(b, window_count, *item.shape[1:])[:, 0] for item in responses))
            reference = self.encode(context, controls,
                torch.zeros((b, n), dtype=torch.long, device=context.device),
                torch.zeros((b, 2), dtype=torch.long, device=context.device), factual_responses)[1]
        for start in range(0, len(distinct), 256):
            q = distinct[start:start+256]
            if window_inverse is None:
                selected = own[:, boats[None], q]
                pair = messages[:, boats[None, :, None], boats[None, None, :], q[:, :, None], q[:, None, :]]
            else:
                window, q = q[:, 0], q[:, 1:]
                selected = own[:, window[:, None], boats[None], q]
                pair = messages[:, window[:, None, None], boats[None, :, None],
                                boats[None, None, :], q[:, :, None], q[:, None, :]]
            size = len(q); held = q > 0
            encoded = self.update(torch.cat((selected, (pair*mask).sum(3)/max(n-1, 1),
                pair.masked_fill(~mask, -1e4).amax(3)), -1))
            if return_encodings:
                encodings.append(encoded)
            changed = (encoded*held[None, :, :, None]).sum(2)/held.sum(1)[None, :, None].clamp_min(1)
            global_state = torch.cat((context[:, None, -5:].expand(-1, size, -1),
                rows.new_zeros((b, size, 2)), rows.new_full((b, size, 1), n/6.)), -1)
            representation = torch.cat((encoded.mean(2), encoded.amax(2), changed, global_state), -1)
            if self.residual_potential:
                factual = reference[:, None].expand(-1, size, -1)
                parts.append(self.effect_head[0](torch.cat((factual, representation-factual), -1)))
            else:
                parts.append(self.head[0](representation))
        # Timing enters the final conditional-value head. Reuse the exact
        # same joint-plan representation across immediate and delayed choices.
        clock = timing.to(rows.dtype).log1p()/6.*(programs > 0).any(-1, keepdim=True)
        head = self.effect_head if self.residual_potential else self.head
        hidden = torch.cat(parts, 1)[:, inverse]+F.linear(clock, head[0].weight[:, -3:-1])[None]
        values = head[2](head[1](hidden)).squeeze(-1)
        if self.residual_potential:
            exposure = -torch.expm1(-.2*timing[:, 1].to(context.dtype))*(programs > 0).any(-1)
            values = self.head(reference).squeeze(-1)[:, None]+exposure[None]*values
        return (values, torch.cat(encodings, 1)[:, inverse]) if return_encodings else values


def local_edges(options, n):
    """Undirected comparisons changing one boat, with an identical clock."""
    lookup = {tuple(option): i for i, option in enumerate(options)}
    pairs = set()
    for a, (code, delay, span) in enumerate(options):
        if code == 0:
            continue
        for boat in range(n):
            digit = (code//10**boat) % 10
            for other in range(10):
                if other == digit:
                    continue
                alternate = code+(other-digit)*10**boat
                # The factual node represents the same frozen continuation
                # for every window; changing one held boat to free is local.
                b = 0 if alternate == 0 else lookup.get((alternate, delay, span))
                if b is not None and b != a:
                    pairs.add((min(a, b), max(a, b)))
    return sorted(pairs)


def single_vessel_neighborhood(code, n):
    """Every alternative that changes exactly one component of a full plan."""
    return sorted({code+(other-code//10**boat % 10)*10**boat
        for boat in range(n) for other in range(10)
        if other != code//10**boat % 10})


def connected_programs(packet, order, interaction_order=2):
    """Include full-team plans together with every single-boat path to them."""
    n = order.shape[-1]
    codes = {0}
    for digit in range(1, 10):
        prefix = 0
        for boat in range(n):
            codes.add(digit*10**boat)
            prefix += digit*10**boat
            codes.add(prefix)
    positions = packet['motion'][0, :, :2][order[0]]
    distance = torch.cdist(positions, positions).fill_diagonal_(float('inf'))
    index = int(distance.argmin()); a, b = divmod(index, n)
    for first in (1, 3, 7, 9):
        for second in (1, 3, 7, 9):
            codes.add(first*10**a+second*10**b)
    if interaction_order >= 3:
        # Heterogeneous small-group maneuvers cannot be represented by
        # uniform fleet plans or isolated one-boat interventions. Include
        # every deletion neighbour, so their values remain linked by genuine
        # single-vessel comparisons rather than additive credit assumptions.
        for boats in combinations(range(n), 3):
            for digits in product((1, 3, 7, 9), repeat=3):
                terms = [digit*10**boat for boat, digit in zip(boats, digits)]
                code = sum(terms); codes.add(code)
                for term in terms: codes.add(code-term)
    return sorted(codes-{0})


def candidate_screen_order(options, n, interaction_order=2, strategy='fleet_first', root_seed=None):
    """Allocate branch queries across plan sizes and intervention windows."""
    def held(code):
        return int(np.count_nonzero((code//(10**np.arange(n))) % 10))
    def priority(q):
        code, _, span = options[q]
        count = held(code)
        rank = 0 if count == n else 1 if count == interaction_order else 2 if count == 2 else 3
        return rank, span, code
    pending = sorted(range(1, len(options)), key=priority)
    if strategy == 'fleet_first':
        return pending
    if strategy not in ('interleaved', 'root_randomized'):
        raise ValueError('Unknown local-comparison sampling allocation.')
    groups = {}
    for q in pending:
        code, _, span = options[q]
        groups.setdefault((held(code), span), []).append(q)
    groups = list(groups.values())
    if strategy == 'root_randomized':
        if root_seed is None:
            raise ValueError('Randomized branch allocation requires a recorded root seed.')
        rng = np.random.default_rng(root_seed)
        for group in groups:
            rng.shuffle(group)
        groups = [groups[index] for index in rng.permutation(len(groups))]
    return [group[k] for k in range(max(map(len, groups), default=0))
            for group in groups if k < len(group)]


def observed_programs(root):
    """Retain every sampled plan, including all negative outcomes."""
    used = sorted({q for q, _ in root['outcomes']})
    if len(used) == len(root['options']): return root
    index = {old: new for new, old in enumerate(used)}
    return dict(root, options=[root['options'][q] for q in used],
        outcomes={(index[q], future): value for (q, future), value in root['outcomes'].items()})


def learning_records(data):
    roots = data['roots']
    node_rows, edge_rows, anchor_rows = [], [], []
    diagnostics = []
    for root_id, root in enumerate(roots):
        n = len(root['effect_controls']) if 'effect_controls' in root else (len(root['context'])-5)//8
        index = {}; observed_futures = {}
        for (q, future), observation in sorted(root['outcomes'].items()):
            index[q, future] = len(node_rows)
            observed_futures.setdefault(q, set()).add(future)
            code, delay, span = root['options'][q]
            node_rows.append((root_id, q, future, code, delay, span, float(observation['success'])))
        edges = local_edges(root['options'], n)
        count = 0
        for a, b in edges:
            common = sorted(observed_futures.get(a, set()) & observed_futures.get(b, set()))
            for future in common:
                edge_rows.append((index[a, future], index[b, future]))
                count += 1
        for (q, future), position in index.items():
            if q and (0, future) in index:
                anchor_rows.append((index[0, future], position))
        diagnostics.append(dict(nodes=len(index), local_edges=len(edges), paired_local_observations=count))
    if not edge_rows:
        raise ValueError('The conditional learner requires observed single-vessel paired differences.')
    return np.asarray(node_rows), np.asarray(edge_rows, dtype=np.int64), np.asarray(anchor_rows, dtype=np.int64), diagnostics


def measured_response_records(roots, rows):
    """Use only measured, identity-aligned successor states as auxiliary labels."""
    indices, targets = [], []
    for index, row in enumerate(rows):
        root_id, option, future = map(int, row[:3])
        root = roots[root_id]
        outcome = root['outcomes'][option, future]
        if 'following' not in outcome or outcome.get('response_sampling') != 'physical_tick':
            continue
        before = np.asarray(root['context'], np.float32)
        after = np.asarray(outcome['following'], np.float32)
        steps = outcome.get('response_steps')
        if (after.shape != before.shape or not np.isfinite(after).all()
                or steps is None or not np.isfinite(steps) or not 0 <= steps <= 20):
            raise ValueError('Measured response supervision requires a valid aligned 20-tick successor.')
        # The collector already restores each successor boat to its root
        # identity. Early terminal responses retain their actual elapsed time.
        indices.append(index)
        targets.append(np.concatenate((after-before, np.asarray([steps/20.], np.float32))))
    if not indices:
        raise ValueError('Response supervision requires measured physical successors.')
    return np.asarray(indices, np.int64), np.asarray(targets, np.float32)


def merge_paired_data(datasets, *, allow_distinct_contexts=False, allow_opponent_mixture=False):
    """Merge observed branches without duplicating a root/program/future."""
    if not datasets:
        raise ValueError('Supply paired terminal observations.')
    predecessor = datasets[0]['predecessor_sha256']
    opponents = {root.get('opponent_sha256', data.get('opponent_sha256'))
                 for data in datasets for root in data['roots']}
    if len(opponents) > 1 and not allow_opponent_mixture:
        raise ValueError('Mixing opponent populations requires an explicit adaptation objective.')
    grouped = {}
    for data in datasets:
        if data['predecessor_sha256'] != predecessor:
            raise ValueError('Conditional and ordinary comparisons need one common frozen predecessor.')
        for original in data['roots']:
            future_weights = np.asarray(original.get('future_weights', np.ones(len(original['futures']))), np.float32)
            if (future_weights.shape != (len(original['futures']),)
                    or not np.isfinite(future_weights).all() or (future_weights < 0).any()
                    or not future_weights.any()):
                raise ValueError('Every observed future requires a finite nonnegative population weight.')
            root = original
            # At reset, a seeded factual continuation and a branch with that
            # same seed are the same experiment. Join their observations
            # before estimating contrasts, rather than fitting conflicting
            # potentials at identical public contexts.
            if original['root_steps'] == 0:
                root = copy.deepcopy(original)
                root['futures'] = [root['prefix_future'] if f is None else f for f in root['futures']]
                root['prefix_future'] = None
            opponent = root.get('opponent_sha256', data.get('opponent_sha256'))
            key = (tuple(root['scene']), root['prefix_future'], root['root_steps'], opponent)
            if allow_distinct_contexts and key in grouped:
                # Different behavior policies can reach different states at
                # the same scene/future/clock. Their frozen branch predecessor
                # is common, but their interventions are different experiments.
                base_key = key
                alternative = 0
                while key in grouped:
                    stored = grouped[key]
                    if (np.allclose(stored['context'], root['context'], atol=1e-6)
                            and np.allclose(stored['effect_controls'], root['effect_controls'], atol=1e-6)):
                        break
                    alternative += 1
                    key = base_key+(alternative,)
            if key not in grouped:
                grouped[key] = copy.deepcopy(root)
                grouped[key]['future_weights'] = future_weights.copy()
                grouped[key]['opponent_sha256'] = opponent
                continue
            old = grouped[key]
            if not np.allclose(old['context'], root['context'], atol=1e-6):
                raise ValueError('A reused paired root has changed physical context.')
            if not np.allclose(old['effect_controls'], root['effect_controls'], atol=1e-6):
                raise ValueError('A reused paired root has changed predecessor controls.')
            options = [(0, 0, 0)]+sorted((set(map(tuple, old['options'])) | set(map(tuple, root['options'])))-{(0, 0, 0)})
            futures = list(old['futures'])
            for future in root['futures']:
                if future not in futures: futures.append(future)
            position = {option: i for i, option in enumerate(options)}
            future_position = {future: i for i, future in enumerate(futures)}
            merged_weights = np.full(len(futures), np.nan, np.float32)
            outcomes = {}
            for entry in (old, root):
                entry_weights = np.asarray(entry.get('future_weights', np.ones(len(entry['futures']))), np.float32)
                for future, weight in zip(entry['futures'], entry_weights):
                    index = future_position[future]
                    if np.isfinite(merged_weights[index]) and not np.isclose(merged_weights[index], weight):
                        raise ValueError('A common future has conflicting population weights; choose a common population before merging.')
                    merged_weights[index] = weight
                for (q, f), outcome in entry['outcomes'].items():
                    index = (position[tuple(entry['options'][q])], future_position[entry['futures'][f]])
                    if index in outcomes and outcomes[index]['success'] != outcome['success']:
                        raise ValueError('Identical physical branches have conflicting terminal labels.')
                    # A later observation may omit the physical response.
                    # Retain actual response labels already collected for the
                    # same branch while merging its shared terminal outcome.
                    merged = outcomes.get(index, {}).copy()
                    merged.update(copy.deepcopy(outcome))
                    outcomes[index] = merged
            old.update(options=options, futures=futures, future_weights=merged_weights, outcomes=outcomes)
    return dict(model=copy.deepcopy(datasets[0]['model']), roots=list(grouped.values()),
        predecessor_sha256=predecessor, simulation_steps=sum(d.get('simulation_steps', 0) for d in datasets),
        opponent_sha256=next(iter(opponents)) if len(opponents) == 1 else None,
        training_opponent_populations=sorted(opponents, key=lambda identity: identity or ''),
        training_data=[d.get('training_data') for d in datasets], completed=all(d.get('completed', False) for d in datasets))


def measured_choice_options(record):
    """Identify each executed branch, including independently sampled candidates."""
    options = {'factual': (0, 0, 0)}
    for label, decision in record['decisions'].items():
        options[label] = (tuple(map(int, decision['option'][0]))
                          if bool(decision['accepted'][0]) else (0, 0, 0))
    ranking = record.get('ranking')
    if ranking is not None:
        labels = ranking['labels']; programs = np.asarray(ranking['options'])
        if (len(set(labels)) != len(labels) or programs.shape != (len(labels), 3)
                or not np.isfinite(programs).all() or not np.array_equal(programs, programs.astype(np.int64))):
            raise ValueError('Measured catalog labels must uniquely identify complete integer programs.')
        for label, program in zip(labels, programs):
            option = tuple(map(int, program))
            if label in options and options[label] != option:
                raise ValueError('A measured branch cannot name two different physical programs.')
            options[label] = option
        if any(label not in record['outcomes'] for label in labels):
            raise ValueError('Every measured catalog program needs its terminal outcomes.')
    if any(label not in options for label in record['outcomes']):
        raise ValueError('A terminal observation lacks its executed physical program.')
    return options


def measured_choices_to_data(policy, measurements, output=None):
    """Acquire every measured choice as training data, including adverse ones.

    A completed development measurement can supply factual and proposed
    terminal observations without inventing labels or simulating them again.
    Independently sampled catalog candidates retain their measured outcomes too.
    Once acquired, those states are training states; subsequent validation
    must use a different collection. Repeated measurements are deduplicated
    by their physical root, complete program, and shared future identity.
    """
    import conditional_fusion_value as c
    import develop_source_fusion as d
    import run_remote_evidence as r

    policy = Path(policy)
    predecessor = hashlib.sha256(policy.read_bytes()).hexdigest()
    model = torch.load(policy, map_location='cpu', weights_only=False)
    if d.joint_value_objectives(model):
        raise ValueError('Acquired choices need their common frozen predecessor without M2.')
    actor = d.joint_load(str(policy), policy.stat().st_mtime_ns)
    coefficients = np.asarray([[-1., -1.]]+[[a, b] for a in (0., .5, 1.) for b in (0., .5, 1.)])
    collections = []; provenance = []
    for path in map(Path, measurements):
        data = torch.load(path, map_location='cpu', weights_only=False)
        identity = data['identity']; records = data['records']
        if (not data['completed'] or identity['predecessor_sha256'] != predecessor
                or len(records) != len(identity['tasks'])):
            raise ValueError('Acquire only completed choices from the same frozen predecessor.')
        roots = []
        for record, task in zip(records, identity['tasks']):
            if (tuple(record['scene']), record['prefix_future']) != (tuple(task[0]), task[1]):
                raise ValueError('A measured root has changed its task identity.')
            futures = list(record['futures'])
            if len(futures) != identity['repetitions'] or len(set(futures)) != len(futures):
                raise ValueError('Each measured future must retain its independent identity.')
            packet = d.mechanism_tensor(record['packet'])
            factual = np.asarray(record['actions']['factual'])
            with torch.no_grad():
                native = actor(packet, deterministic=True)[0]
                context, order = c.intervention_task_context(packet, True, True)
                effect = c.paired_effect_controls(actor, packet, native)
            if not np.allclose(native[0].cpu().numpy(), factual, atol=1e-6, rtol=0.):
                raise ValueError('The acquired root changed its factual command.')
            n = len(factual); inverse = order[0].argsort().cpu().numpy()
            one = np.ones((n, 1), dtype=np.float32)
            options = [(0, 0, 0)]; outcomes = {}
            measured_options = measured_choice_options(record)
            for label, rows in record['outcomes'].items():
                option = measured_options[label]
                if option == (0, 0, 0):
                    expected = factual
                else:
                    code, delay, span = option
                    if not 0 < code < 10**n or delay < 0 or span < 1:
                        raise ValueError('A measured choice is not a finite fleet-compatible intervention.')
                    preference = coefficients[(code//(10**np.arange(n))) % 10][inverse].astype(np.float32)
                    first = preference if delay == 0 else np.full_like(preference, -1.)
                    expected = np.concatenate((factual[:, :3], -one, (delay+span)*one, -2*one,
                        np.zeros((n, 2), np.float32), delay*one, first, preference,
                        np.zeros_like(one)), -1)
                action = np.asarray(record['actions'][label])
                if action.shape != expected.shape or not np.allclose(action, expected, atol=1e-6, rtol=0.):
                    raise ValueError('A measured program does not reconstruct its actual executed command.')
                if option not in options:
                    options.append(option)
                q = options.index(option)
                if len(rows) != len(futures):
                    raise ValueError('A measured option is missing a complete shared future.')
                for f, row in enumerate(rows):
                    item = {key: row[key] for key in ('success', 'outcome', 'duration', 'physical_steps')}
                    if (float(item['success']) not in (0., 1.) or not int(item['outcome'])
                            or int(item['physical_steps']) < 1):
                        raise ValueError('Only observed terminal outcomes are training labels.')
                    if (q, f) in outcomes and outcomes[q, f] != item:
                        raise ValueError('The same measured branch has conflicting terminal outcomes.')
                    outcomes[q, f] = item
            roots.append(dict(context=context[0].cpu().numpy(),
                effect_controls=effect[0, order[0]].cpu().numpy(), options=options,
                outcomes=outcomes, futures=futures, scene=tuple(record['scene']),
                prefix_future=record['prefix_future'], root_steps=record['root_step'],
                lookback=None, order=order[0].cpu().numpy(),
                acquisition='all factual, proposed and sampled catalog choices, including neutral and harmful outcomes'))
        collections.append(dict(model=model, predecessor_sha256=predecessor,
            roots=roots, simulation_steps=data['simulation_steps'], completed=True,
            training_data=str(path)))
        provenance.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            previous_role=data.get('result_role'), identity=identity))
    merged = merge_paired_data(collections, allow_distinct_contexts=True)
    merged.update(result_role='development training acquired from completed choice measurements',
        acquisition_sources=provenance, measured_terminal_observations=sum(
            len(root['outcomes']) for root in merged['roots']),
        outcome_selection='all completed roots and all actually measured choices; no outcome filtering',
        subsequent_validation_requires_new_states=True)
    if output is not None:
        merged['training_data'] = str(output)
        r.atomic_save(merged, Path(output))
    return merged


def factual_reference_data(policy, references):
    """Fit the factual value on the entire observed development population.

    Failure-focused branch acquisition must not make ordinary successful
    states look like factual failures. These are existing complete task
    outcomes; no counterfactual label or future seed enters the representation.
    """
    import conditional_fusion_value as c
    import develop_source_fusion as d
    policy = Path(policy)
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    model = torch.load(policy, map_location='cpu', weights_only=False)
    if model.get('successor_task_value') is not None:
        raise ValueError('Factual references require the frozen F00 or F10 predecessor.')
    actor = d.joint_load(str(policy), policy.stat().st_mtime_ns)
    roots = {}; steps = 0
    for result in references:
        if not result['completed'] or result.get('policy_sha256') != digest:
            raise ValueError('Use completed evaluations of this exact predecessor.')
        for (scene, future), row in zip(result['tasks'], result['rows']):
            scene = tuple(scene)
            if scene not in roots:
                np.random.seed(scene[0]); torch.manual_seed(scene[0])
                env = d.numerical_source().TADEnv(scene[1])
                obs, _ = env.reset(agility=scene[2], noisy_agility=False)
                packet = d.mechanism_tensor(d.joint_packet(env, obs))
                with torch.no_grad():
                    action = actor(packet, deterministic=True, gate_only=True)[0][..., :3]
                    context, order = c.intervention_task_context(packet, True, True)
                    controls = c.paired_effect_controls(actor, packet, action)
                roots[scene] = dict(context=context[0].numpy(), effect_controls=controls[0, order[0]].numpy(),
                    options=[(0, 0, 0)], outcomes={}, futures=[], scene=scene, prefix_future=None,
                    root_steps=0, lookback=None, order=order[0].numpy())
            root = roots[scene]
            if future not in root['futures']: root['futures'].append(future)
            key = (0, root['futures'].index(future))
            if key in root['outcomes']:
                if root['outcomes'][key]['success'] != row['success']:
                    raise ValueError('Repeated factual evaluation has conflicting terminal outcomes.')
                continue
            root['outcomes'][key] = dict(success=float(row['success']), outcome=row['outcome'], duration=row['duration'])
            steps += int(round(row['duration']/.2))
    return dict(model=model, roots=list(roots.values()), predecessor_sha256=digest,
        simulation_steps=steps, training_data=[r['policy'] for r in references], completed=True)


def include_delayed_continuations(data, policy=None, coordinate_reference=None):
    """Compare intervening now with an observed later intervention.

    A measured branch following a factual prefix is also a complete delayed
    plan at reset. Only its unchanged (None) continuation can be reused this
    way; freshly reseeded suffixes are not initial-state experiments. An
    already terminal factual prefix is identical for every subsequent plan.
    """
    if 'measured_terminal_observations' in data:
        return copy.deepcopy(data)
    result = copy.deepcopy(data)
    measured = sum(len(root['outcomes']) for root in result['roots'])
    initial = {tuple(root['scene']): root for root in result['roots'] if root['root_steps'] == 0}
    if coordinate_reference is not None:
        if coordinate_reference['predecessor_sha256'] != data['predecessor_sha256']:
            raise ValueError('Reuse boat coordinates only for the same frozen predecessor.')
        known = {(tuple(root['scene']), root['prefix_future'], root['root_steps']): root
                 for root in coordinate_reference['roots'] if 'order' in root}
        for root in result['roots']:
            previous = known.get((tuple(root['scene']), root['prefix_future'], root['root_steps']))
            if 'order' not in root and previous is not None and np.allclose(
                    root['context'], previous['context'], atol=1e-6):
                root['order'] = np.asarray(previous['order']).copy()
    replay_steps = 0; unmatched = 0
    if any('order' not in root for root in result['roots']):
        if policy is None:
            raise ValueError('Supply the frozen predecessor to recover missing physical boat coordinates.')
        import conditional_fusion_value as c
        import develop_source_fusion as d
        policy = Path(policy)
        if hashlib.sha256(policy.read_bytes()).hexdigest() != result['predecessor_sha256']:
            raise ValueError('Delayed labels require their exact measured predecessor.')
        actor = d.joint_load(str(policy), policy.stat().st_mtime_ns)
        for root in result['roots']:
            if 'order' in root: continue
            scene = root['scene']; np.random.seed(scene[0]); torch.manual_seed(scene[0])
            env = d.numerical_source().TADEnv(scene[1]); obs, _ = env.reset(agility=scene[2], noisy_agility=False)
            if root['prefix_future'] is not None: np.random.seed(root['prefix_future'])
            elapsed = done = 0
            while elapsed < root['root_steps'] and not done:
                with torch.no_grad():
                    action = actor(d.mechanism_tensor(d.joint_packet(env, obs)), deterministic=True)[0][0].numpy()
                obs, _, done, _, count = d.joint_advance(env, obs, actor, action, root['root_steps']-elapsed)
                elapsed += count; replay_steps += count
            packet = d.mechanism_tensor(d.joint_packet(env, obs))
            with torch.no_grad(): context, order = c.intervention_task_context(packet, True, True)
            if elapsed != root['root_steps'] or not np.allclose(context[0].numpy(), root['context'], atol=1e-5, rtol=1e-5):
                unmatched += 1
                continue
            root['order'] = order[0].numpy()
    promoted = terminal = 0
    for root in result['roots']:
        if root['root_steps'] <= 0 or None not in root['futures'] or 'order' not in root:
            continue
        target = initial.get(tuple(root['scene']))
        if target is None:
            raise ValueError('A delayed plan requires the observed initial factual context.')
        if 'order' not in target: continue
        # A program's digits name the boats in its decision-time ordering.
        # Transport those physical identities to the initial ordering before
        # reusing a later branch as a delayed complete plan.
        coordinate = np.argsort(root['order'])[target['order']]
        powers = 10**np.arange(len(coordinate))
        future = root['prefix_future']; source_future = root['futures'].index(None)
        if future not in target['futures']: target['futures'].append(future)
        destination_future = target['futures'].index(future)
        options = {tuple(option): q for q, option in enumerate(target['options'])}
        for (q, f), outcome in root['outcomes'].items():
            if f != source_future: continue
            code, delay, span = root['options'][q]
            if code: code = int((((code//powers) % 10)[coordinate]*powers).sum())
            option = (code, root['root_steps']+delay, span) if code else (0, 0, 0)
            if option not in options:
                options[option] = len(target['options']); target['options'].append(option)
            index = (options[option], destination_future)
            if index in target['outcomes']:
                if target['outcomes'][index]['success'] != outcome['success']:
                    raise ValueError('A measured factual prefix has a conflicting complete-task outcome.')
                continue
            target['outcomes'][index] = dict(outcome, reused_prefix=True)
            promoted += 1
    for root in initial.values():
        facts = [(f, outcome) for (q, f), outcome in root['outcomes'].items() if q == 0]
        for q, (code, delay, span) in enumerate(root['options']):
            if not code or delay <= 0: continue
            for future, outcome in facts:
                if outcome['duration'] > .2*delay+1e-6 or (q, future) in root['outcomes']: continue
                root['outcomes'][q, future] = dict(outcome, reused_terminal_prefix=True)
                terminal += 1
    result.update(measured_terminal_observations=measured,
        reused_delayed_labels=promoted, reused_terminal_prefix_labels=terminal,
        coordinate_replay_steps=replay_steps, unmatched_factual_prefixes=unmatched,
        simulation_steps=result.get('simulation_steps', 0)+replay_steps)
    return result


def focus_development_futures(data, references, weight=16.):
    """Optimize the explicitly reusable development population in both arms."""
    result = copy.deepcopy(data); population = {}
    for reference in references:
        for scene, future in reference['tasks']:
            population.setdefault(tuple(scene), set()).add(future)
    for root in result['roots']:
        futures = population.get(tuple(root['scene']), set())
        root['future_weights'] = [float(weight) if (
            (future in futures) if root['root_steps'] == 0 else
            (future is None and root['prefix_future'] in futures)) else 1.
            for future in root['futures']]
    result['development_future_weight'] = float(weight)
    return result


def initial_plan_comparisons(data, policy, factual_bank, references):
    """Bring verified later comparisons to the initial decision horizon.

    A late intervention after a factual prefix is also an executable delayed
    complete plan at reset. Verify the observed prefix against the frozen
    predecessor trajectory before transporting its single-boat comparisons.
    Labels after a different learned prefix remain local to their own state.
    """
    if factual_bank['predecessor_sha256'] != data['predecessor_sha256']:
        raise ValueError('Initial comparisons require the same frozen predecessor.')
    reference = factual_reference_data(policy, references)
    # These task results are already represented in the common trajectory bank.
    reference['simulation_steps'] = 0
    trajectories = {(tuple(record['scene']), record['future']): record
                    for record in factual_bank['records'] if 'option' not in record}
    retained = []; verified = 0; local_only = 0
    for root in data['roots']:
        if root['root_steps'] == 0:
            retained.append(root)
            continue
        trajectory = trajectories.get((tuple(root['scene']), root['prefix_future']))
        if trajectory is None or None not in root['futures']:
            local_only += 1
            continue
        ticks = np.rint(400*trajectory['contexts'][:, -1]).astype(int)
        positions = np.flatnonzero(ticks == root['root_steps'])
        matched = any(np.allclose(root['context'], trajectory['contexts'][i], atol=1e-6, rtol=1e-6)
                      and np.allclose(root['effect_controls'], trajectory['controls'][i], atol=1e-6, rtol=1e-6)
                      for i in positions)
        # The first intervention has an unchanged predecessor prefix by
        # construction. Later learned prefixes need fresh local comparisons;
        # matching only a compressed public snapshot is insufficient.
        factual_prefix = (data.get('recipe', {}).get('context_policy') is None
                          or root.get('lookback') == 'first_intervention')
        if not matched or not factual_prefix:
            local_only += 1
            continue
        future = root['futures'].index(None)
        factual = root['outcomes'].get((0, future))
        if factual is None or factual['success'] != trajectory['values'][0]:
            raise ValueError('A verified predecessor prefix changed its factual task result.')
        retained.append(root); verified += 1
    measured = dict(data, roots=retained, simulation_steps=0)
    prepared = merge_paired_data([reference, measured])
    promoted = include_delayed_continuations(prepared, policy=policy)
    promoted['roots'] = [root for root in promoted['roots'] if root['root_steps'] == 0]
    promoted['simulation_steps'] = promoted.get('coordinate_replay_steps', 0)
    promoted.update(verified_factual_prefixes=verified, unpromoted_local_prefixes=local_only,
                    initial_factual_labels=sum(len(root['outcomes']) for root in reference['roots']),
                    training_data='verified initial-horizon comparisons from the common measured data')
    return promoted


def state_features(context, controls, time_scale=1.):
    # Public kinematics and the actual predecessor command define locality.
    n = controls.shape[-2]
    rows = context[..., :8*n].reshape(*context.shape[:-1], n, 8)
    if context.shape[-1] == 9*n+5:
        rows = torch.cat((rows, context[..., 8*n:9*n, None]), -1)
    elif context.shape[-1] != 8*n+5:
        raise ValueError('Unknown public state representation.')
    clock = torch.cat((context[..., -5:-1], time_scale*context[..., -1:]), -1)
    return torch.cat((torch.cat((rows, .25*controls), -1).flatten(-2), clock), -1)


def retain_vessel_identity(data, factual_bank=None):
    """Expose recorded physical indices to both learners without changing labels."""
    from conditional_fusion_value import append_vessel_identity
    result = copy.deepcopy(data)
    for root in result['roots']:
        if 'order' not in root:
            raise ValueError('A training state has no measured physical vessel order.')
        root['context'] = append_vessel_identity(torch.as_tensor(root['context']),
            torch.as_tensor(root['order'])).numpy()
    result['public_vessel_identity'] = True
    bank = copy.deepcopy(factual_bank)
    if bank is not None:
        if bank['predecessor_sha256'] != result['predecessor_sha256']:
            raise ValueError('Identity-conditioned data need their own frozen predecessor.')
        for record in bank['records']:
            if 'orders' not in record:
                raise ValueError('Factual trajectories require measured vessel orders; replay missing records.')
            record['contexts'] = append_vessel_identity(torch.as_tensor(record['contexts']),
                torch.as_tensor(record['orders'])).numpy()
        bank['public_vessel_identity'] = True
    return result, bank


def reconcile_root(root, prior, edge_weight=4., ridge=.5, effective_future_confidence=False):
    """Fit one integrable potential using absolute and paired local outcomes."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    options = root['options']; size = len(options)
    n = len(root['effect_controls']) if 'effect_controls' in root else (len(root['context'])-5)//8
    observations = root['outcomes']
    counts = np.zeros(size); totals = np.zeros(size); observed_futures = {}; raw_counts = np.zeros(size)
    future_weights = root.get('future_weights', np.ones(len(root['futures'])))
    for (q, future), item in observations.items():
        weight = future_weights[future]
        counts[q] += weight; totals[q] += weight*item['success']; raw_counts[q] += 1
        observed_futures.setdefault(q, set()).add(future)
    baseline = totals[0]/max(1., counts[0])
    repair = np.zeros(size); damage = np.zeros(size)
    factual_failure = np.zeros(size); factual_success = np.zeros(size)
    failure_squared_weight = np.zeros(size); success_squared_weight = np.zeros(size)
    for (q, future), item in observations.items():
        if q and (0, future) in observations:
            factual = observations[0, future]['success']
            weight = future_weights[future]
            if factual:
                factual_success[q] += weight; damage[q] += weight*(1-item['success'])
                success_squared_weight[q] += weight**2
            else:
                factual_failure[q] += weight; repair[q] += weight*item['success']
                failure_squared_weight[q] += weight**2
    rescue = (repair+.5)/(factual_failure+1.)
    injury = (damage+.5)/(factual_success+1.)
    rescue_var = rescue*(1-rescue)/(factual_failure+2.)
    injury_var = injury*(1-injury)/(factual_success+2.)
    conditional_success = np.stack((rescue, 1-injury))
    conditional_success[:, 0] = (0., 1.)
    standardized = (1-baseline)*rescue+baseline*(1-injury)
    standardized[0] = baseline
    variance = (1-baseline)**2*rescue_var+baseline**2*injury_var
    precision = np.where(counts > 0, np.minimum(100., 1./(4*variance+.01)), 0.)
    rhs = precision*standardized+ridge*np.asarray(prior)
    row_indices = []; column_indices = []; entries = []
    edge_count = 0
    for a, b in local_edges(options, n):
        futures = sorted(observed_futures.get(a, set()) & observed_futures.get(b, set()))
        if not futures:
            continue
        differences = np.array([observations[b, f]['success']-observations[a, f]['success'] for f in futures])
        weights = np.asarray([future_weights[f] for f in futures])
        strata = np.array([observations[0, f]['success'] for f in futures], dtype=int)
        # Match the factual-success distribution, rather than extrapolating
        # two sampled factual failures to every possible future.
        difference = 0.
        for stratum, population in ((0, 1-baseline), (1, baseline)):
            selected = differences[strata == stratum]
            mass = weights[strata == stratum]
            prior_difference = conditional_success[stratum, b]-conditional_success[stratum, a]
            difference += population*((selected*mass).sum()+prior_difference)/(mass.sum()+1.)
        # Each edge estimates a signed consequence, never a smoothing target.
        weight = edge_weight*weights.sum()
        row_indices.extend((a, b, a, b)); column_indices.extend((a, b, b, a))
        entries.extend((weight, weight, -weight, -weight))
        rhs[b] += weight*difference; rhs[a] -= weight*difference
        edge_count += len(futures)
    # The factual value is measured by its own complete future population.
    # Reusing that node in many noisy contrasts must not move its baseline.
    # Unobserved nodes have only their independent ridge prior. Eliminate
    # them exactly instead of solving a large empty part of the comparison.
    active = np.flatnonzero(counts[1:] > 0)+1
    matrix = sparse.diags(precision+ridge, format='csr')+sparse.coo_matrix(
        (entries, (row_indices, column_indices)), shape=(size, size)).tocsr()
    reduced = matrix[active][:, active].tocsc()
    reduced_rhs = rhs[active]-matrix[active, 0].toarray().ravel()*baseline
    free_value = spsolve(reduced, reduced_rhs) if len(active) else np.empty(0)
    if np.any((free_value < 0.) | (free_value > 1.)):
        from scipy.optimize import minimize
        solution = minimize(lambda x: (.5*x@reduced@x-reduced_rhs@x, reduced@x-reduced_rhs),
            np.clip(free_value, 0., 1.), jac=True, method='L-BFGS-B', bounds=[(0., 1.)]*len(active),
            options=dict(maxiter=200, ftol=1e-10, gtol=1e-7))
        free_value = solution.x
    value = np.asarray(prior).copy(); value[0] = baseline; value[active] = free_value
    # A symmetric variance penalty can reward an unobserved repair stratum:
    # two successful factual continuations say nothing about rescuing a
    # factual failure. Use separate one-sided bounds for rescue and harm.
    from scipy.stats import beta
    paired_count = factual_failure+factual_success
    confidence_failure = factual_failure
    confidence_success = factual_success
    confidence_repair = repair
    confidence_damage = damage
    if effective_future_confidence:
        # Importance weights specify the fitted objective, not additional
        # independent futures. Their concentration reduces the effective
        # evidence count; rescaling all weights cannot create confidence.
        confidence_failure = factual_failure**2/np.maximum(failure_squared_weight, 1e-20)
        confidence_success = factual_success**2/np.maximum(success_squared_weight, 1e-20)
        confidence_repair = np.clip(repair/np.maximum(factual_failure, 1e-20), 0., 1.)*confidence_failure
        confidence_damage = np.clip(damage/np.maximum(factual_success, 1e-20), 0., 1.)*confidence_success
    repair_lower = beta.ppf(.2, confidence_repair+.5, confidence_failure-confidence_repair+.5)
    damage_upper = beta.ppf(.8, confidence_damage+.5, confidence_success-confidence_damage+.5)
    gain_lower = (1-baseline)*repair_lower-baseline*damage_upper
    uncertainty = np.maximum(0., value-baseline-gain_lower)
    uncertainty[0] = 0.
    return value, uncertainty, counts, dict(local_difference_observations=edge_count,
        factual_value=float(baseline), factual_observations=int(raw_counts[0]),
        factual_objective_mass=float(counts[0]), counts_are_objective_weights='future_weights' in root,
        paired_factual_failure=factual_failure.tolist(), paired_factual_success=factual_success.tolist(),
        paired_repair=repair.tolist(), paired_damage=damage.tolist(), paired_count=paired_count.tolist(),
        confidence_failure_count=confidence_failure.tolist(), confidence_success_count=confidence_success.tolist(),
        confidence_counting='effective weighted futures' if effective_future_confidence else 'objective mass',
        repair_lower=repair_lower.tolist(), damage_upper=damage_upper.tolist())


class PlainJointReturnDecision(nn.Module):
    """Absolute-return argmax over the shared complete intervention catalog.

    This execution path contains only fitted networks and candidate programs.
    It has no training-state memory, paired-difference correction, factual
    calibration, uncertainty threshold, inherited value stage or support test.
    """
    def __init__(self, width, states, options, defenders, peer_conditioned=True,
                 context_conditioned=True, motion_basis=True, decision_period=5,
                 inference_device=None, residual_potential=False, finite_motion_basis=False, dual_motion_basis=False,
                 identity_conditioned=False, gain_margin=0., intervention_budget=0):
        super().__init__()
        if not np.isfinite(gain_margin) or not 0. <= gain_margin <= 1.:
            raise ValueError('The ordinary graph-return margin must lie between zero and one.')
        if not isinstance(intervention_budget, int) or intervention_budget < 0:
            raise ValueError('The ordinary graph-return intervention budget must be a nonnegative integer.')
        self.gain_margin = float(gain_margin)
        self.intervention_budget = intervention_budget
        self.models = nn.ModuleList([JointTaskPotential(width, peer_conditioned,
            context_conditioned=context_conditioned, motion_basis=motion_basis,
            residual_potential=residual_potential, finite_motion_basis=finite_motion_basis,
            dual_motion_basis=dual_motion_basis, identity_conditioned=identity_conditioned) for _ in states])
        for net, state in zip(self.models, states):
            net.load_state_dict(state)
        self.register_buffer('options', torch.as_tensor(options, dtype=torch.long))
        self.register_buffer('programs', (self.options[:, :1]//(10**torch.arange(defenders))) % 10)
        self.decision_period = decision_period
        self.inference_device = inference_device
        self.prior = None
        self.ordinary = True

    def eligible(self, packet):
        eligible = (packet['node'][:, 0, 15]*400.).round().long() % self.decision_period == 0
        if self.intervention_budget and '_conditional_count' in packet:
            eligible &= packet['_conditional_count'].flatten() < self.intervention_budget
        return eligible

    @torch.no_grad()
    def prepare_packet(self, actor, packet, action):
        from conditional_fusion_value import paired_effect_controls
        result = dict(packet)
        if bool(self.eligible(packet).any()):
            result['_ordinary_return_controls'] = paired_effect_controls(actor, packet, action)
        return result

    @torch.no_grad()
    def forward(self, packet, action):
        from conditional_fusion_value import intervention_task_context
        eligible = self.eligible(packet)
        if not bool(eligible.any()):
            return action
        return_device = action.device
        if self.inference_device is not None and self.options.device != torch.device(self.inference_device):
            if str(self.inference_device).startswith('cuda'):
                os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
                torch.use_deterministic_algorithms(True)
                torch.backends.cuda.matmul.allow_tf32 = False
            self.to(self.inference_device)
        context, order = intervention_task_context(packet, True, True)
        if getattr(self.models[0], 'identity_conditioned', False):
            from conditional_fusion_value import append_vessel_identity
            context = append_vessel_identity(context, order)
        if order.shape[1] != self.programs.shape[1]:
            raise ValueError('The return estimator and deployed fleet must have the same size.')
        controls = packet['_ordinary_return_controls'].gather(1, order[..., None].expand(-1, -1, 6))
        context = context.to(self.options.device); controls = controls.to(self.options.device)
        order = order.to(self.options.device); action = action.to(self.options.device)
        eligible = eligible.to(self.options.device)
        responses = response_basis(context, controls) if self.models[0].motion_basis else None
        values = torch.stack([net.forward_catalog(context, controls, self.programs,
            self.options[:, 1:], responses=responses).sigmoid() for net in self.models]).mean(0)
        choice = values.argmax(-1)
        accept = eligible & (choice != 0)
        if self.gain_margin:
            gain = values.gather(1, choice[:, None]).squeeze(1)-values[:, 0]
            accept &= gain > self.gain_margin
        if not bool(accept.any()):
            return action.to(return_device)
        preference = self.models[0].coefficients[self.programs[choice]]
        preference = preference.gather(1, order.argsort(-1)[..., None].expand(-1, -1, 2))
        delay, span = self.options[choice, 1:].unbind(-1)
        one = torch.ones_like(action[..., :1]); free = -one
        first = torch.where((delay == 0)[:, None, None], preference, free.expand(-1, -1, 2))
        proposed = torch.cat((action[..., :3], free, (delay+span)[:, None, None]*one, 2*free,
            torch.zeros_like(action[..., :2]), delay[:, None, None]*one, first, preference, one), -1)
        factual = torch.cat((action[..., :3], free, one, 2*free, torch.zeros_like(action[..., :2]),
            one, free.expand(-1, -1, 2), free.expand(-1, -1, 2), free), -1)
        return torch.where(accept[:, None, None], proposed, factual).to(return_device)


def fit_factual_probability_calibration(probability, target, weight=None, ridge=1e-3):
    """Fit a bounded monotone probability map on out-of-fold factual outcomes.

    The two endpoints define p -> (1-p)*low + p*high. A small identity
    penalty stabilizes nearly constant predictions. The constrained quadratic
    has only an interior solution and three edges, all solved directly.
    """
    probability = np.asarray(probability, np.float64)
    target = np.asarray(target, np.float64)
    weight = np.ones_like(target) if weight is None else np.asarray(weight, np.float64)
    if (probability.ndim != 1 or probability.shape != target.shape or weight.shape != target.shape
            or not len(target) or not np.isfinite(np.concatenate((probability, target, weight))).all()
            or np.any((probability < 0) | (probability > 1) | (target < 0) | (target > 1))
            or np.any(weight < 0) or weight.sum() <= 0 or not np.isfinite(ridge) or ridge <= 0):
        raise ValueError('Factual calibration requires finite probabilities, outcomes and positive mass.')
    weight = weight/weight.sum()
    design = np.column_stack((1-probability, probability))
    matrix = design.T@(weight[:, None]*design)+ridge*np.eye(2)
    rhs = design.T@(weight*target)+ridge*np.array([0., 1.])
    candidates = [np.linalg.solve(matrix, rhs),
        np.array([0., np.clip(rhs[1]/matrix[1, 1], 0., 1.)]),
        np.array([np.clip((rhs[0]-matrix[0, 1])/matrix[0, 0], 0., 1.), 1.])]
    constant = np.clip(rhs.sum()/matrix.sum(), 0., 1.)
    candidates.append(np.full(2, constant))
    candidates = [pair for pair in candidates if 0. <= pair[0] <= pair[1] <= 1.]
    best = min(candidates, key=lambda pair: pair@matrix@pair-2*rhs@pair)
    fitted = design@best
    return best.astype(np.float32), dict(observations=len(target),
        raw_brier=float(np.sum(weight*(probability-target)**2)),
        calibrated_brier=float(np.sum(weight*(fitted-target)**2)),
        mean_outcome=float(weight@target), mean_prediction=float(weight@probability),
        mean_calibrated=float(weight@fitted), ridge=ridge)


def fit_factual_logit_calibration(probability, target, weight=None, ridge=1e-3):
    """Fit a monotone logit map on the same held-out factual population.

    Unlike a line in probability space, this map does not impose learned
    nonzero and subunit endpoints. Its nonnegative slope preserves ordering.
    Weighted binomial likelihood and an identity ridge form a convex problem.
    """
    probability = np.asarray(probability, np.float64)
    target = np.asarray(target, np.float64)
    weight = np.ones_like(target) if weight is None else np.asarray(weight, np.float64)
    if (probability.ndim != 1 or probability.shape != target.shape or weight.shape != target.shape
            or not len(target) or not np.isfinite(np.concatenate((probability, target, weight))).all()
            or np.any((probability < 0) | (probability > 1) | (target < 0) | (target > 1))
            or np.any(weight < 0) or weight.sum() <= 0 or not np.isfinite(ridge) or ridge <= 0):
        raise ValueError('Factual calibration requires finite probabilities, outcomes and positive mass.')
    weight = weight/weight.sum()
    clipped = np.clip(probability, 1e-6, 1.-1e-6)
    design = np.column_stack((np.log(clipped)-np.log1p(-clipped), np.ones(len(clipped))))
    identity = np.array([1., 0.])
    def objective(parameters):
        logits = design@parameters
        return weight@(np.logaddexp(0., logits)-target*logits)+.5*ridge*np.square(parameters-identity).sum()
    parameters = identity.copy()
    for _ in range(80):
        logits = design@parameters
        fitted = np.exp(-np.logaddexp(0., -logits))
        gradient = design.T@(weight*(fitted-target))+ridge*(parameters-identity)
        curvature = design.T@((weight*fitted*(1-fitted))[:, None]*design)+ridge*np.eye(2)
        direction = (np.array([0., gradient[1]/curvature[1, 1]])
                     if parameters[0] <= 1e-12 and gradient[0] >= 0 else
                     np.linalg.solve(curvature, gradient))
        previous = objective(parameters); accepted = False
        for step in 2.**-np.arange(32):
            candidate = parameters-step*direction
            candidate[0] = max(0., candidate[0])
            if objective(candidate) <= previous+1e-14:
                accepted = True
                break
        if not accepted:
            break
        change = np.max(np.abs(candidate-parameters))
        parameters = candidate
        if change < 1e-10:
            break
    logits = design@parameters
    fitted = np.exp(-np.logaddexp(0., -logits))
    return parameters.astype(np.float32), dict(observations=len(target),
        raw_brier=float(np.sum(weight*(probability-target)**2)),
        calibrated_brier=float(np.sum(weight*(fitted-target)**2)),
        raw_log_loss=float(weight@(np.logaddexp(0., design[:, 0])-target*design[:, 0])),
        calibrated_log_loss=float(weight@(np.logaddexp(0., logits)-target*logits)),
        mean_outcome=float(weight@target), mean_prediction=float(weight@probability),
        mean_calibrated=float(weight@fitted), ridge=ridge, map='monotone logit')


def factual_crossfit_groups(roots, factual_bank, members, seed):
    """Keep every future and trajectory state of one layout in one fold."""
    def scene(record):
        value = record.get('scene')
        if value is None or len(value) != 3:
            raise ValueError('Factual cross-fitting requires the measured layout of every record.')
        return int(value[0]), int(value[1]), float(value[2])
    root_scenes = [scene(root) for root in roots]
    bank_scenes = [] if factual_bank is None else [scene(row) for row in factual_bank['records']]
    scenes = sorted(set(root_scenes+bank_scenes))
    if members < 2 or len(set(root_scenes)) < members:
        raise ValueError('Factual cross-fitting requires at least one paired layout per member.')
    # Assign paired layouts first so every held-out fold has paired outcomes.
    paired = sorted(set(root_scenes)); extra = sorted(set(scenes)-set(paired))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 578309]))
    ordered = [paired[index] for index in rng.permutation(len(paired))]
    ordered += [extra[index] for index in rng.permutation(len(extra))]
    lookup = {key:index % members for index, key in enumerate(ordered)}
    return np.asarray([lookup[key] for key in root_scenes]), np.asarray([lookup[key] for key in bank_scenes]), lookup


class LocalDifferenceDecision(nn.Module):
    def __init__(self, width=64, states=(), peer_conditioned=True, ordinary=False,
                 contexts=None, controls=None, options=(), corrections=None, uncertainty=None,
                 observed=None, bandwidth=.12, margin=.08, confidence=.5, global_penalty=.35,
                 decision_period=5, intervention_budget=1, context_time_scale=1., interpolation_power=0.,
                 member_corrections=None, conditional_strata=False, factual_values=None,
                 decision_mode='rolling', project_task_value=False, commit_factual=False,
                 continuation_contexts=None, continuation_controls=None, continuation_values=None,
                 continuation_remaining=None, temporal_value_bounds=False, context_conditioned=False,
                 local_contrast_head=False, projection_strength=2., projection_steps=24, motion_basis=False,
                 align_context_roles=False, joint_factual_calibration=False, comparison_weights_override=None,
                 inference_device=None, stratum_member_corrections=None, stratum_uncertainty=None,
                 stratum_observed=None, replan_delayed=False, intervention_locality=False,
                 residual_potential=False, factual_bound_pruning=False, candidate_bound_pruning=False,
                 stratum_value_bounds=False, stratum_localized_bounds=False,
                 factual_difference_transport=False, continuation_factual_only=False,
                 outcome_scaled_confidence=False, factual_risk_confidence=False,
                 neural_factual_calibration=None, neural_factual_calibration_kind='probability',
                 finite_motion_basis=False, dual_motion_basis=False, fleet_transfer=None,
                 identity_conditioned=False, factual_probability_calibration=True):
        super().__init__()
        self.conditional_strata = conditional_strata
        self.models = nn.ModuleList([JointTaskPotential(width, peer_conditioned,
            3 if conditional_strata else 1, context_conditioned, local_contrast_head, motion_basis,
            residual_potential, finite_motion_basis, dual_motion_basis, identity_conditioned) for _ in states])
        for net, state in zip(self.models, states):
            net.load_state_dict(state)
        calibration = (None if neural_factual_calibration is None else
                       torch.as_tensor(np.asarray(neural_factual_calibration), dtype=torch.float32))
        if neural_factual_calibration_kind not in ('probability', 'logit'):
            raise ValueError('Unknown factual probability calibration map.')
        if calibration is not None:
            if (not conditional_strata or ordinary or calibration.shape != (len(states), 2)
                    or not torch.isfinite(calibration).all()):
                raise ValueError('Factual probability maps require two finite parameters for every stratum model.')
            if (neural_factual_calibration_kind == 'probability' and
                    ((calibration < 0).any() or (calibration > 1).any()
                     or (calibration[:, 0] > calibration[:, 1]).any())):
                raise ValueError('Factual probability maps need bounded monotone endpoints for every stratum model.')
            if neural_factual_calibration_kind == 'logit' and (calibration[:, 0] < 0).any():
                raise ValueError('Factual logit calibration must preserve probability ordering.')
        self.neural_factual_calibration_kind = neural_factual_calibration_kind
        self.factual_probability_calibration = factual_probability_calibration
        self.register_buffer('neural_factual_calibration', calibration)
        self.ordinary = ordinary
        if identity_conditioned and align_context_roles:
            raise ValueError('Physical vessel identities cannot be reassigned by role matching.')
        self.bandwidth = bandwidth; self.margin = margin; self.confidence = confidence
        self.global_penalty = global_penalty; self.decision_period = decision_period
        self.intervention_budget = intervention_budget; self.prior = None
        self.record_decisions = False; self.last_decision = None
        self.record_catalog = False; self.last_catalog = None
        self.decision_mode = decision_mode
        self.project_task_value = project_task_value
        self.commit_factual = commit_factual
        self.temporal_value_bounds = temporal_value_bounds
        self.local_contrast_head = local_contrast_head
        self.projection_strength = projection_strength
        self.projection_steps = projection_steps
        self.align_context_roles = align_context_roles
        self.joint_factual_calibration = joint_factual_calibration
        self.inference_device = inference_device
        self.replan_delayed = replan_delayed
        self.intervention_locality = intervention_locality
        self.factual_bound_pruning = factual_bound_pruning
        self.candidate_bound_pruning = candidate_bound_pruning
        self.stratum_value_bounds = stratum_value_bounds
        self.stratum_localized_bounds = stratum_localized_bounds
        self.factual_difference_transport = factual_difference_transport
        self.continuation_factual_only = continuation_factual_only
        self.outcome_scaled_confidence = outcome_scaled_confidence
        self.factual_risk_confidence = factual_risk_confidence
        if fleet_transfer not in (None, 'shared_neural'):
            raise ValueError('Unknown cross-fleet conditional-value rule.')
        if fleet_transfer and (ordinary or local_contrast_head):
            raise ValueError('Shared cross-fleet inference requires shared conditional task outcomes.')
        if fleet_transfer and (confidence < 0 or global_penalty < 0):
            raise ValueError('Cross-fleet confidence penalties must be nonnegative.')
        self.fleet_transfer = fleet_transfer
        self._fleet_catalogs = {}
        if outcome_scaled_confidence and factual_risk_confidence:
            raise ValueError('Conditional confidence must use one factual-outcome rule.')
        if ordinary and (outcome_scaled_confidence or factual_risk_confidence):
            raise ValueError('Factual-outcome conditional confidence is not an ordinary-return objective.')
        if intervention_locality and align_context_roles:
            raise ValueError('Intervention locality uses the fitted physical boat correspondence.')
        if decision_mode not in ('rolling', 'complete_plan'):
            raise ValueError('Unknown conditional decision horizon.')
        self.context_time_scale = context_time_scale; self.interpolation_power = interpolation_power
        self.register_buffer('contexts', torch.as_tensor(contexts, dtype=torch.float32))
        self.register_buffer('controls', torch.as_tensor(controls, dtype=torch.float32))
        self.register_buffer('options', torch.as_tensor(options, dtype=torch.long))
        n = self.controls.shape[1]
        self.register_buffer('programs', (self.options[:, :1]//(10**torch.arange(n))) % 10)
        if local_contrast_head:
            pairs = torch.as_tensor(local_edges(options, n), dtype=torch.long)
            degrees = torch.bincount(pairs.flatten(), minlength=len(options)).to(torch.float32)
            weights = projection_strength/(1.+.5*(degrees[pairs[:, 0]]+degrees[pairs[:, 1]]))
            if comparison_weights_override is not None:
                weights = torch.as_tensor(comparison_weights_override, dtype=torch.float32)
                if len(weights) != len(pairs):
                    raise ValueError('The retained local-comparison weights do not match the candidate graph.')
            diagonal = torch.ones(len(options)).index_add(0, pairs[:, 0], weights).index_add(0, pairs[:, 1], weights)
            changed = (self.programs[pairs[:, 0]] != self.programs[pairs[:, 1]]).to(torch.long).argmax(-1)
            self.register_buffer('comparison_pairs', pairs)
            self.register_buffer('comparison_boats', changed)
            self.register_buffer('comparison_weights', weights)
            self.register_buffer('comparison_diagonal', diagonal)
            _, first, inverse = np.unique(self.options[pairs, 0].numpy(), axis=0, return_index=True, return_inverse=True)
            self.register_buffer('comparison_distinct_pairs', pairs[first])
            self.register_buffer('comparison_distinct_boats', changed[first])
            self.register_buffer('comparison_inverse', torch.as_tensor(inverse, dtype=torch.long))
            self.register_buffer('comparison_clock', self.options[pairs, 1:].amax(1).to(torch.float32).log1p()/6.)
        self.register_buffer('corrections', torch.as_tensor(corrections, dtype=torch.float32))
        self.register_buffer('member_corrections', None if member_corrections is None else
                             torch.as_tensor(member_corrections, dtype=torch.float32))
        self.register_buffer('uncertainty', torch.as_tensor(uncertainty, dtype=torch.float32))
        self.register_buffer('observed', torch.as_tensor(observed, dtype=torch.bool))
        for key, value, dtype in (('stratum_member_corrections', stratum_member_corrections, torch.float32),
                                  ('stratum_uncertainty', stratum_uncertainty, torch.float32),
                                  ('stratum_observed', stratum_observed, torch.bool)):
            self.register_buffer(key, None if value is None else torch.as_tensor(value, dtype=dtype))
        if stratum_member_corrections is not None and (ordinary or align_context_roles):
            raise ValueError('Stratified local calibration requires paired learning and the fitted boat correspondence.')
        if align_context_roles:
            positions = self.observed.nonzero()
            self.register_buffer('observation_roots', positions[:, 0])
            self.register_buffer('observation_options', positions[:, 1])
            keys = (self.options[:, 0]*512+self.options[:, 1])*512+self.options[:, 2]
            sorted_keys, lookup_order = keys.sort()
            self.register_buffer('option_keys', sorted_keys)
            self.register_buffer('option_lookup_order', lookup_order)
        self.register_buffer('root_features', state_features(self.contexts, self.controls, context_time_scale))
        self.register_buffer('factual_values', None if factual_values is None else
                             torch.as_tensor(factual_values, dtype=torch.float32))
        if continuation_contexts is None:
            self.register_buffer('continuation_features', None)
            self.register_buffer('continuation_values', None)
            self.register_buffer('continuation_failure_alive', None)
            self.register_buffer('continuation_success_alive', None)
            self.register_buffer('continuation_delay_index', None)
        else:
            features = state_features(torch.as_tensor(continuation_contexts, dtype=torch.float32),
                torch.as_tensor(continuation_controls, dtype=torch.float32), context_time_scale)
            outcomes = torch.as_tensor(continuation_values, dtype=torch.float32)
            remaining = torch.as_tensor(continuation_remaining, dtype=torch.float32)
            delays, index = self.options[:, 1].unique(return_inverse=True)
            alive = (remaining[:, None] > delays[None]).to(torch.float32)
            self.register_buffer('continuation_features', features)
            self.register_buffer('continuation_values', outcomes)
            self.register_buffer('continuation_failure_alive', alive*(1-outcomes[:, None]))
            self.register_buffer('continuation_success_alive', alive*outcomes[:, None])
            self.register_buffer('continuation_delay_index', index)

    def confidence_at(self, factual):
        if self.factual_risk_confidence:
            # Preserve rescue opportunities at low factual success while
            # increasing the cost of changing an already promising course.
            return self.confidence*(.5+factual.clamp(0., 1.))
        if not self.outcome_scaled_confidence:
            return self.confidence
        factual = factual.clamp(0., 1.)
        # Binary task ambiguity peaks at 1/2. Retain a nonzero penalty at
        # either endpoint and use this same bound during candidate pruning.
        return self.confidence*(.5+2*factual*(1-factual))

    def eligible(self, packet):
        tick = (packet['node'][:, 0, 15]*400.).round().long()
        # Evaluate complete finite interventions with a frozen continuation;
        # a rolling controller revisits the choice on the configured clock.
        if self.decision_mode == 'complete_plan':
            return tick == 0
        eligible = tick % self.decision_period == 0
        if not self.ordinary and self.intervention_budget and '_conditional_count' in packet:
            eligible &= packet['_conditional_count'].flatten() < self.intervention_budget
        return eligible

    @torch.no_grad()
    def prepare_packet(self, actor, packet, action):
        from conditional_fusion_value import paired_effect_controls
        result = dict(packet)
        if bool(self.eligible(packet).any()):
            result['_local_difference_controls'] = paired_effect_controls(actor, packet, action)
        return result

    def continuation_statistics(self, context, controls):
        if self.continuation_features is None:
            return None
        features = state_features(context, controls, self.context_time_scale)
        distance = (features[:, None]-self.continuation_features[None]).square().mean(-1)
        kernels = torch.exp(-distance/(2*self.bandwidth**2))
        weights = kernels/(distance+1e-6).pow(self.interpolation_power)
        total = weights.sum(-1)
        success_mass = weights@self.continuation_values
        failure_mass = weights@(1-self.continuation_values)
        factual = success_mass/total.clamp_min(1e-8)
        survival_failure = weights@self.continuation_failure_alive/failure_mass[:, None].clamp_min(1e-8)
        survival_success = weights@self.continuation_success_alive/success_mass[:, None].clamp_min(1e-8)
        # An unobserved factual stratum supplies no survival restriction.
        survival_failure = torch.where(failure_mass[:, None] > 1e-8, survival_failure,
                                       torch.ones_like(survival_failure))
        survival_success = torch.where(success_mass[:, None] > 1e-8, survival_success,
                                       torch.ones_like(survival_success))
        influence = kernels.amax(-1)
        result = (factual, influence, survival_failure[:, self.continuation_delay_index], survival_success[:, self.continuation_delay_index])
        return result+(total,) if self.joint_factual_calibration else result

    def context_relation(self, context, controls):
        features = state_features(context, controls, self.context_time_scale)
        if not self.align_context_roles:
            return (features[:, None]-self.root_features[None]).square().mean(-1), None
        from scipy.optimize import linear_sum_assignment
        n = controls.shape[1]
        query = features[:, :14*n].reshape(-1, n, 14).detach().cpu().numpy()
        reference = self.root_features[:, :14*n].reshape(-1, n, 14).detach().cpu().numpy()
        total = np.empty((len(query), len(reference)), np.float32)
        mapping = np.empty((len(query), len(reference), n), np.int64)
        for b, item in enumerate(query):
            costs = ((item[None, :, None]-reference[:, None])**2).sum(-1)
            for r, cost in enumerate(costs):
                own, source = linear_sum_assignment(cost)
                total[b, r] = cost[own, source].sum()
                mapping[b, r, source] = own
        global_difference = (features[:, None, -5:]-self.root_features[None, :, -5:]).square().sum(-1)
        distance = (torch.as_tensor(total, device=context.device)+global_difference)/features.shape[-1]
        return distance, torch.as_tensor(mapping, device=context.device)

    def aligned_local_statistics(self, weights, kernels, mapping):
        """Transfer complete plans through a physical boat correspondence."""
        roots, options = self.observation_roots, self.observation_options
        b, k = len(weights), len(self.options)
        destination = mapping[:, roots]
        code = (self.programs[options][None]*(10**destination)).sum(-1)
        key = (code*512+self.options[options, 1])*512+self.options[options, 2]
        position = torch.searchsorted(self.option_keys, key).clamp_max(k-1)
        valid = self.option_keys[position] == key
        query_option = self.option_lookup_order[position]
        selected = weights[:, roots]*valid
        mass = weights.new_zeros((b, k)).scatter_add(1, query_option, selected)
        error = weights.new_zeros((b, k)).scatter_add(1, query_option,
            selected*self.uncertainty[roots, options])/mass.clamp_min(1e-8)
        influence = weights.new_zeros((b, k)).scatter_reduce(1, query_option,
            kernels[:, roots]*valid, reduce='amax', include_self=True)
        if self.member_corrections is None:
            residual = weights.new_zeros((b, k)).scatter_add(1, query_option,
                selected*self.corrections[roots, options])/mass.clamp_min(1e-8)
            residual = residual[None]
        else:
            members = self.member_corrections[:, roots, options]
            residual = weights.new_zeros((len(members), b, k)).scatter_add(2,
                query_option[None].expand(len(members), -1, -1), selected[None]*members[:, None])
            residual = residual/mass[None].clamp_min(1e-8)
        return residual, error, influence

    def intervention_distances(self, context, controls, global_distance):
        """Condition transferred differences on the boats the plan changes.

        An unchanged majority must not hide a large change in the intervened
        boat's state or source commands. Whole-fleet and active-boat locality
        must both hold. The factual plan keeps the whole-context metric.
        """
        n = controls.shape[1]
        query = state_features(context, controls, self.context_time_scale)
        features_per_vessel = (query.shape[-1]-5)//n
        boat_difference = (query[:, None, :features_per_vessel*n].reshape(-1, 1, n, features_per_vessel)-
                           self.root_features[None, :, :features_per_vessel*n].reshape(1, -1, n, features_per_vessel)).square().sum(-1)
        global_difference = (query[:, None, -5:]-self.root_features[None, :, -5:]).square().sum(-1)
        active = (self.programs > 0).to(query.dtype)
        count = active.sum(-1)
        local = (torch.einsum('brn,kn->brk', boat_difference, active)+global_difference[:, :, None])/(features_per_vessel*count+5)
        local = torch.where(count[None, None] > 0, local, global_distance[:, :, None])
        return torch.maximum(global_distance[:, :, None], local)

    def calibrate_factual_probability(self, probability, member=None):
        if not self.factual_probability_calibration or self.neural_factual_calibration is None:
            return probability
        parameters = self.neural_factual_calibration
        if member is None:
            shape = (-1,)+(1,)*(probability.ndim-1)
            left, right = (parameters[:, index].reshape(shape) for index in (0, 1))
        else:
            left, right = parameters[member]
        if self.neural_factual_calibration_kind == 'logit':
            return (left*torch.logit(probability.clamp(1e-6, 1.-1e-6))+right).sigmoid()
        return left+(right-left)*probability

    def values(self, context, controls, continuation=None, relation=None, responses=None, factual_only=False,
               stratum=None, candidate_indices=None):
        all_values = []
        if candidate_indices is not None and self.local_contrast_head:
            raise ValueError('Coupled value reconstruction requires its complete comparison graph.')
        programs = self.programs if candidate_indices is None else self.programs[candidate_indices]
        timing = self.options[:, 1:] if candidate_indices is None else self.options[candidate_indices, 1:]
        if responses is None and self.models[0].motion_basis:
            responses = response_basis(context, controls)
        if continuation is None:
            continuation = self.continuation_statistics(context, controls)
        factual_reference = influence = factual_mass = None
        calibrate_factual = getattr(self, 'factual_probability_calibration', True)
        continuation_only = getattr(self, 'continuation_factual_only', False)
        if calibrate_factual and not self.ordinary and self.factual_values is not None and not continuation_only:
            distance, _ = self.context_relation(context, controls) if relation is None else relation
            kernels = torch.exp(-distance/(2*self.bandwidth**2))
            weights = kernels/(distance+1e-6).pow(self.interpolation_power)
            factual_reference = (weights@self.factual_values)/weights.sum(-1).clamp_min(1e-8)
            influence = kernels.amax(-1)
            factual_mass = weights.sum(-1)
        for member, net in enumerate(self.models):
            if factual_only:
                raw = net(context, controls,
                    torch.zeros((len(context), controls.shape[1]), dtype=torch.long, device=context.device),
                    torch.zeros((len(context), 2), dtype=torch.long, device=context.device), responses).sigmoid()[:, None]
            else:
                catalog = net.forward_catalog(context, controls, programs, timing,
                                              return_encodings=self.local_contrast_head, responses=responses)
                if self.local_contrast_head:
                    raw, encoded = catalog; raw = raw.sigmoid()
                else:
                    raw = catalog.sigmoid()
            if self.conditional_strata:
                baseline = raw[:, :1, 0]
                baseline = self.calibrate_factual_probability(baseline, member)
                if self.joint_factual_calibration and continuation is not None and factual_reference is not None:
                    reference, locality, _, _, mass = continuation
                    # Pool both sets of factual evidence once. In particular,
                    # an approximate branch root must not overwrite an exact
                    # factual trajectory state merely because it is applied last.
                    combined = (mass*reference+factual_mass*factual_reference)/(mass+factual_mass).clamp_min(1e-8)
                    locality = torch.maximum(locality, influence)
                    baseline = (1-locality[:, None])*baseline+locality[:, None]*combined[:, None]
                elif calibrate_factual:
                    if continuation is not None:
                        reference, locality, _, _ = continuation[:4]
                        baseline = (1-locality[:, None])*baseline+locality[:, None]*reference[:, None]
                    if factual_reference is not None:
                        baseline = (1-influence[:, None])*baseline+influence[:, None]*factual_reference[:, None]
                if stratum is not None:
                    baseline = torch.full_like(baseline, float(stratum))
                values = (1-baseline)*raw[:, :, 1]+baseline*(1-raw[:, :, 2])
                values[:, 0] = baseline[:, 0]
                if self.local_contrast_head and not factual_only:
                    values = self.reconstruct_values(net, encoded, context, values, baseline)
            else:
                values = raw
                if calibrate_factual and not self.ordinary and (factual_reference is not None
                        or continuation_only and continuation is not None):
                    if continuation_only:
                        reference, locality = continuation[:2]
                    else:
                        reference = factual_reference; locality = influence
                    if continuation is not None and not continuation_only:
                        continuation_reference, continuation_locality, _, _, mass = continuation
                        reference = (mass*continuation_reference+factual_mass*reference)/(mass+factual_mass).clamp_min(1e-8)
                        locality = torch.maximum(locality, continuation_locality)
                    baseline = (1-locality[:, None])*values[:, :1]+locality[:, None]*reference[:, None]
                    if self.factual_difference_transport:
                        # Transport the shared factual calibration to the
                        # complete potential. Preserve the learned paired
                        # differences except at binary-value boundaries.
                        values = (values+baseline-values[:, :1]).clamp(0., 1.)
                    else:
                        values = values.clone()
                    values[:, 0] = baseline[:, 0]
                    if self.local_contrast_head and not factual_only:
                        precision = (1+40*locality/(1-locality+1e-6)).clamp_max(1e4)
                        values = self.reconstruct_values(net, encoded, context, values, baseline, precision)
            all_values.append(values)
        return torch.stack(all_values)

    def reconstruct_values(self, net, encoded, context, prior, baseline, anchor_precision=None):
        """Jointly reconcile learned local differences and complete-plan values.

        Projected Jacobi iterations fit one bounded quadratic potential.
        A factual anchor can be fixed, or have finite precision when its
        uncertain estimate must be reconciled with local comparisons.
        """
        pairs = self.comparison_pairs
        delta = net.catalog_contrasts(context, encoded, self.comparison_distinct_pairs,
            self.comparison_distinct_boats, self.comparison_inverse, self.comparison_clock)
        difference = ((1-baseline)*delta[:, :, 0]+baseline*delta[:, :, 1]
                      if self.conditional_strata else delta[:, :, 0])*self.comparison_weights
        rhs = prior.clone().index_add(1, pairs[:, 1], difference).index_add(1, pairs[:, 0], -difference)
        diagonal = self.comparison_diagonal
        if anchor_precision is not None:
            diagonal = diagonal[None].expand(len(prior), -1).clone()
            diagonal[:, 0] += anchor_precision
            rhs[:, 0] += anchor_precision*baseline[:, 0]
        value = prior
        for _ in range(self.projection_steps):
            message = torch.zeros_like(value).index_add(1, pairs[:, 0], value[:, pairs[:, 1]]*self.comparison_weights)
            message.index_add_(1, pairs[:, 1], value[:, pairs[:, 0]]*self.comparison_weights)
            value = ((rhs+message)/diagonal).clamp(0., 1.)
            if anchor_precision is None:
                value[:, 0] = baseline[:, 0]
        return value

    def combine_stratum_gains(self, gains, baseline, locality, residual):
        if not self.stratum_value_bounds:
            return gains+(1-baseline)*locality[0]*residual[0]+baseline*locality[1]*residual[1]
        if self.stratum_localized_bounds:
            # The neural potential is a marginal task value, not a pair of
            # conditional rescue/harm heads. Bound the locally calibrated
            # consequences, then interpolate by their observed support.
            # With no conditional evidence, retain the neural gain exactly.
            failed = (gains+residual[0]).clamp(0., 1.)
            succeeded = (gains+residual[1]).clamp(-1., 0.)
            return (gains+(1-baseline)*locality[0]*(failed-gains)
                    +baseline*locality[1]*(succeeded-gains))
        failed = gains+locality[0]*residual[0]
        succeeded = gains+locality[1]*residual[1]
        # Conditional on factual failure, a binary task gain is in [0,1].
        # Conditional on factual success, it is in [-1,0]. Enforce these
        # identities before mixing; a bound on the final average alone
        # does not exclude a fictitious gain in the success stratum.
        failed = failed.clamp(0., 1.)
        succeeded = succeeded.clamp(-1., 0.)
        return (1-baseline)*failed+baseline*succeeded

    def cross_fleet_catalog(self, defenders):
        """Map canonical boat roles and retain every single-boat alternative.

        Missing roles disappear; additional boats keep their factual control
        in mixed plans. A uniform fleet command applies to every current boat.
        The shared network evaluates the resulting complete current-fleet
        plan, without transferring empirical six-boat outcome corrections.
        """
        if not 2 <= defenders <= 12:
            raise ValueError('The decimal intervention catalog supports 2 to 12 defenders.')
        if defenders not in self._fleet_catalogs:
            old_programs = self.programs.detach().cpu().numpy()
            old_options = self.options.detach().cpu().numpy()
            source_size = old_programs.shape[1]
            options = {(0, 0, 0)}
            for program, (_, delay, span) in zip(old_programs, old_options):
                changed = np.flatnonzero(program)
                if not len(changed):
                    continue
                mapped = np.zeros(defenders, np.int64)
                mapped[:min(defenders, source_size)] = program[:defenders]
                if len(changed) == source_size and np.all(program == program[0]):
                    mapped[:] = program[0]
                code = int(mapped @ (10**np.arange(defenders, dtype=np.int64)))
                if code:
                    options.add((code, int(delay), int(span)))
                if len(changed) == 1:
                    for boat in range(defenders):
                        options.add((int(program[changed[0]])*10**boat, int(delay), int(span)))
            catalog = torch.tensor(sorted(options), dtype=torch.long)
            programs = (catalog[:, :1]//(10**torch.arange(defenders))) % 10
            self._fleet_catalogs[defenders] = (programs, catalog[:, 1:])
        return tuple(item.to(self.options.device) for item in self._fleet_catalogs[defenders])

    @torch.no_grad()
    def cross_fleet_forward(self, context, controls, order, action, eligible):
        """Apply the shared paired potential with uncertainty at an unseen size."""
        programs, timing = self.cross_fleet_catalog(controls.shape[1])
        responses = response_basis(context, controls) if self.models[0].motion_basis else None
        factual = torch.stack([net(context, controls,
            torch.zeros_like(order), torch.zeros((len(context), 2), dtype=torch.long, device=context.device),
            responses).sigmoid() for net in self.models])
        if self.conditional_strata:
            factual = factual[..., 0]
            factual = self.calibrate_factual_probability(factual)
        penalty = self.confidence_at(factual.mean(0))*self.global_penalty
        possible = eligible & (1-factual.mean(0)-penalty > self.margin)
        if not bool(possible.any()):
            return action
        estimates = torch.stack([net.forward_catalog(context, controls, programs, timing,
            responses=responses).sigmoid() for net in self.models])
        if self.conditional_strata:
            baseline = factual[..., None]
            estimates = (1-baseline)*estimates[..., 1]+baseline*(1-estimates[..., 2])
            estimates[:, :, 0] = factual
        gains = estimates-estimates[:, :, :1]
        error = gains.std(0, unbiased=False)+self.global_penalty
        score = gains.mean(0)-self.confidence_at(estimates[:, :, :1].mean(0))*error
        score[:, 0] = 0.
        best, choice = score.max(-1)
        accept = possible & (choice != 0) & (best > self.margin)
        if self.replan_delayed:
            accept &= timing[choice, 0] == 0
        if not bool(accept.any()):
            return action
        preference = self.models[0].coefficients[programs[choice]]
        preference = preference.gather(1, order.argsort(-1)[..., None].expand(-1, -1, 2))
        delay, span = timing[choice].unbind(-1)
        one = torch.ones_like(action[..., :1]); free = -one
        first = torch.where((delay == 0)[:, None, None], preference, free.expand(-1, -1, 2))
        proposed = torch.cat((action[..., :3], free, (delay+span)[:, None, None]*one, 2*free,
            torch.zeros_like(action[..., :2]), delay[:, None, None]*one, first, preference, one), -1)
        factual_action = torch.cat((action[..., :3], free, one, 2*free, torch.zeros_like(action[..., :2]),
            one, free.expand(-1, -1, 2), free.expand(-1, -1, 2), free), -1)
        return torch.where(accept[:, None, None], proposed, factual_action)

    @torch.no_grad()
    def forward(self, packet, action):
        from conditional_fusion_value import intervention_task_context
        if self.record_decisions:
            self.last_decision = None
        if self.record_catalog:
            self.last_catalog = None
        eligible = self.eligible(packet)
        if not bool(eligible.any()):
            return action
        return_device = action.device
        if self.inference_device is not None and self.options.device != torch.device(self.inference_device):
            if torch.device(self.inference_device).type == 'cuda':
                os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
                torch.use_deterministic_algorithms(True)
                torch.backends.cuda.matmul.allow_tf32 = False
            self.to(self.inference_device)
        context, order = intervention_task_context(packet, True, True)
        if getattr(self.models[0], 'identity_conditioned', False):
            from conditional_fusion_value import append_vessel_identity
            context = append_vessel_identity(context, order)
        b, n = order.shape
        if n != self.programs.shape[1] and self.fleet_transfer is None:
            raise ValueError('Fit a fleet-compatible representation before claiming cross-fleet results.')
        controls = packet['_local_difference_controls'].gather(1, order[..., None].expand(-1, -1, 6))
        if context.device != self.options.device:
            context = context.to(self.options.device); controls = controls.to(self.options.device)
            order = order.to(self.options.device); eligible = eligible.to(self.options.device)
            action = action.to(self.options.device)
        if n != self.programs.shape[1]:
            return self.cross_fleet_forward(context, controls, order, action, eligible).to(return_device)
        continuation = self.continuation_statistics(context, controls)
        relation = None if self.ordinary else self.context_relation(context, controls)
        responses = response_basis(context, controls) if self.models[0].motion_basis else None
        factual_value = None
        if (not self.ordinary and (self.conditional_strata or self.models[0].residual_potential
                                  or self.factual_bound_pruning)
                and self.project_task_value and not self.commit_factual
                and self.confidence >= 0. and self.global_penalty >= 0.
                and not self.record_catalog):
            factual_value = self.values(context, controls, continuation, relation, responses, factual_only=True)[:, :, 0]
            # Every bounded task potential is at most one. If even this upper
            # bound cannot improve the factual value by the decision margin,
            # no candidate can be admitted; future states are still revisited.
            tolerance = 1e-4 if self.factual_bound_pruning else 1e-6
            eligible &= 1.-factual_value.mean(0) > self.margin-tolerance
            if not bool(eligible.any()):
                if self.record_decisions:
                    self.last_decision = dict(reason='factual_upper_bound',
                        factual_value=factual_value.mean(0).detach().cpu(),
                        option=torch.zeros((len(action), 3), dtype=torch.long),
                        accepted=torch.zeros(len(action), dtype=torch.bool),
                        predicted_gain=torch.zeros(len(action)), score=torch.zeros(len(action)))
                return action.to(return_device)
        candidate_indices = None
        if (self.candidate_bound_pruning and factual_value is not None and not self.record_catalog
                and not self.local_contrast_head and not self.intervention_locality
                and relation[1] is None):
            kernels = torch.exp(-relation[0]/(2*self.bandwidth**2))
            reference = factual_value.mean(0)[:, None]
            if self.stratum_member_corrections is not None:
                locality = (kernels[None, :, :, None] *
                            self.stratum_observed[:, None].to(kernels.dtype)).amax(2)
                influence = (1-reference)*locality[0]+reference*locality[1]
            else:
                influence = (kernels[:, :, None]*self.observed[None]).amax(1)
            # A corrected success value cannot exceed one. The global
            # uncertainty term is a lower bound on the total penalty; local
            # uncertainty and ensemble dispersion are both nonnegative.
            upper = 1-reference-self.confidence_at(reference)*(1-influence)*self.global_penalty
            possible = (upper > self.margin-1e-4).any(0)
            possible[0] = True
            candidate_indices = possible.nonzero().flatten()
        reduced = self.values(context, controls, continuation, relation, responses,
                              candidate_indices=candidate_indices)
        if candidate_indices is None:
            estimates = reduced
        else:
            estimates = reduced[:, :, :1].expand(-1, -1, len(self.options)).clone()
            estimates[:, :, candidate_indices] = reduced
        if self.ordinary:
            score = estimates.mean(0)
            choice = score.argmax(-1)
            accept = eligible & (choice != 0)
        else:
            gains = estimates-estimates[:, :, :1]
            distance, mapping = relation
            # Continuous confidence, not a training-root admission test.
            kernels = torch.exp(-distance/(2*self.bandwidth**2))
            weights = kernels/(distance+1e-6).pow(self.interpolation_power)
            if self.intervention_locality:
                distance = self.intervention_distances(context, controls, distance)
                kernels = torch.exp(-distance/(2*self.bandwidth**2))
                weights = kernels/(distance+1e-6).pow(self.interpolation_power)
                if self.stratum_member_corrections is not None:
                    mask = self.stratum_observed.to(kernels.dtype)
                    mass = torch.einsum('brk,srk->sbk', weights, mask).clamp_min(1e-8)
                    locality = (kernels[None]*mask[:, None]).amax(2)
                    residual = torch.einsum('brk,smrk->smbk', weights, self.stratum_member_corrections)/mass[:, None]
                    stratum_error = torch.einsum('brk,srk->sbk', weights, self.stratum_uncertainty)/mass
                    baseline = estimates[:, :, :1]
                    corrected_members = self.combine_stratum_gains(gains, baseline, locality, residual)
                    reference = baseline.mean(0)
                    influence = (1-reference)*locality[0]+reference*locality[1]
                    error = ((1-reference)*locality[0]*stratum_error[0]+
                             reference*locality[1]*stratum_error[1])/influence.clamp_min(1e-8)
                else:
                    mass = (weights*self.observed[None]).sum(1).clamp_min(1e-8)
                    influence = (kernels*self.observed[None]).amax(1)
                    members = self.corrections[None] if self.member_corrections is None else self.member_corrections
                    residual = torch.einsum('brk,mrk->mbk', weights, members)/mass[None]
                    error = (weights*self.uncertainty[None]).sum(1)/mass
                    corrected_members = gains+influence[None]*residual
            elif self.stratum_member_corrections is not None:
                # Transfer rescue and damage separately. A net gain measured
                # where the predecessor fails cannot be carried unchanged to
                # a state where that predecessor is likely to succeed.
                mask = self.stratum_observed.to(kernels.dtype)
                mass = torch.einsum('br,srk->sbk', weights, mask).clamp_min(1e-8)
                stratum_error = torch.einsum('br,srk->sbk', weights, self.stratum_uncertainty)/mass
                locality = (kernels[None, :, :, None]*mask[:, None]).amax(2)
                residual = torch.einsum('br,smrk->smbk', weights, self.stratum_member_corrections)/mass[:, None]
                baseline = estimates[:, :, :1]
                corrected_members = self.combine_stratum_gains(gains, baseline, locality, residual)
                reference = baseline.mean(0)
                influence = (1-reference)*locality[0]+reference*locality[1]
                # This is the already locality-weighted conditional error.
                error = ((1-reference)*locality[0]*stratum_error[0]+
                         reference*locality[1]*stratum_error[1])/influence.clamp_min(1e-8)
            elif mapping is not None:
                residual, error, influence = self.aligned_local_statistics(weights, kernels, mapping)
                corrected_members = gains+influence[None]*residual
            else:
                mass = weights @ self.observed.to(kernels.dtype)
                correction = weights @ self.corrections/mass.clamp_min(1e-8)
                error = weights @ self.uncertainty/mass.clamp_min(1e-8)
                # Repeating distant roots must not manufacture local confidence.
                influence = (kernels[:, :, None]*self.observed[None]).amax(1)
                if self.member_corrections is None:
                    corrected_members = gains+influence[None]*correction[None]
                else:
                    residual = torch.einsum('br,mrk->mbk', weights, self.member_corrections)/mass[None].clamp_min(1e-8)
                    corrected_members = gains+influence[None]*residual
            if self.project_task_value:
                # Transporting a local difference to a different context must
                # retain the range of the success-valued task potential. In
                # particular, a factual value of one permits no positive gain.
                reference = estimates[:, :, :1]
                corrected_members = (reference+corrected_members).clamp(0., 1.)-reference
            if self.temporal_value_bounds and continuation is not None:
                _, locality, survive_failure, survive_success = continuation[:4]
                survive_failure = 1-locality[:, None]*(1-survive_failure)
                survive_success = 1-locality[:, None]*(1-survive_success)
                reference = estimates[:, :, :1]
                lower = -reference*survive_success[None]
                upper = (1-reference)*survive_failure[None]
                # Until a delayed intervention starts, its continuation is
                # factual. It cannot repair or damage an already ended task.
                corrected_members = torch.maximum(lower, torch.minimum(upper, corrected_members))
            corrected = corrected_members.mean(0)
            error = influence*error+corrected_members.std(0, unbiased=False)+(1-influence)*self.global_penalty
            score = corrected-self.confidence_at(estimates[:, :, :1].mean(0))*error
            score[:, 0] = 0.
            best, choice = score.max(-1)
            accept = eligible & (choice != 0) & (best > self.margin)
        if self.replan_delayed:
            # A future alternative is a reason to continue observing now,
            # not a commitment to suppress every decision until that time.
            # Both estimators reconsider it at the next decision period.
            accept &= self.options[choice, 1] == 0
        if self.record_catalog:
            reference = estimates[:, :, 0].mean(0)
            predicted = (estimates.mean(0)-reference[:, None]) if self.ordinary else corrected
            self.last_catalog = dict(options=self.options.detach().cpu(),
                factual_value=reference.detach().cpu(), predicted_gain=predicted.detach().cpu(),
                score=score.detach().cpu())
        if self.record_decisions:
            reference = estimates[:, :, 0].mean(0)
            predicted = (estimates.mean(0)-reference[:, None]) if self.ordinary else corrected
            chosen_gain = predicted.gather(1, choice[:, None]).flatten()
            self.last_decision = dict(reason='catalog',
                factual_value=reference.detach().cpu(), option=self.options[choice].detach().cpu(),
                accepted=accept.detach().cpu(),
                predicted_gain=torch.where(accept, chosen_gain, torch.zeros_like(chosen_gain)).detach().cpu(),
                candidate_gain=chosen_gain.detach().cpu(),
                score=score.gather(1, choice[:, None]).flatten().detach().cpu())
        if self.ordinary:
            commit = eligible & (choice == 0) & self.commit_factual
        else:
            upper_gain = 1.-estimates[:, :, 0].mean(0)
            commit = eligible & ~accept & (upper_gain <= self.margin) & self.commit_factual
        if not bool((accept | commit).any()):
            return action.to(return_device)
        selected = self.programs[choice]
        preference = self.models[0].coefficients[selected]
        preference = preference.gather(1, order.argsort(-1)[..., None].expand(-1, -1, 2))
        delay, span = self.options[choice, 1:].unbind(-1)
        one = torch.ones_like(action[..., :1]); free = -one
        first = torch.where((delay == 0)[:, None, None], preference, free.expand(-1, -1, 2))
        proposed = torch.cat((action[..., :3], free, (delay+span)[:, None, None]*one, 2*free,
            torch.zeros_like(action[..., :2]), delay[:, None, None]*one, first, preference, one), -1)
        factual = torch.cat((action[..., :3], free, one, 2*free, torch.zeros_like(action[..., :2]),
            one, free.expand(-1, -1, 2), free.expand(-1, -1, 2), free), -1)
        # Retain the explicit terminal-commit option for historical recipes.
        # Current rolling recipes leave commit_factual false and reconsider
        # both factual and deferred alternatives at later observations.
        tick = packet['node'][:, 0, 15].to(action.device)*400.
        remaining = (400-tick.round()).clamp_min(1.)[:, None, None]*one
        committed = torch.cat((action[..., :3], free, remaining, 2*free,
            torch.zeros_like(action[..., :2]), remaining, free.expand(-1, -1, 2),
            free.expand(-1, -1, 2), one), -1)
        factual = torch.where(commit[:, None, None], committed, factual)
        return torch.where(accept[:, None, None], proposed, factual).to(return_device)


def fit_joint_potential(data, output, *, ordinary=False, updates=24000, ensemble_size=3, width=64,
                        edge_weight=4., anchor_weight=2., seed=42, device='cuda:0', bandwidth=.06,
                        margin=.02, confidence=.5, global_penalty=.2, peer_conditioned=True,
                        context_time_scale=20., interpolation_power=2., rank_weight=1., factual_weight=4.,
                        conditional_strata=False, factual_bank=None, factual_bank_weight=1., context_conditioned=True,
                        local_contrast_head=False, projection_strength=2., projection_steps=24, motion_basis=True,
                        joint_factual_calibration=True, balanced_local_sampling=True,
                        stratified_local_calibration=False, intervention_locality=False, residual_potential=False,
                        ordinary_architecture='mlp', initial_states=None, learning_rate=8e-4,
                        additional_candidate_options=(), factual_difference_transport=False,
                        effective_future_confidence=False, finite_motion_basis=False, dual_motion_basis=False,
                        response_supervision_weight=0., identity_conditioned=False,
                        snapshot_updates=(), root_sampling_fraction=0., factual_crossfit=False,
                        factual_calibration_kind='probability'):
    from plain_return import root_sampling_probabilities
    if not np.isfinite(response_supervision_weight) or response_supervision_weight < 0:
        raise ValueError('Response supervision must have a finite nonnegative weight.')
    if response_supervision_weight and ordinary:
        raise ValueError('The ordinary return fit must explicitly implement its own auxiliary objective.')
    if dual_motion_basis and not finite_motion_basis:
        raise ValueError('The dual motion basis requires finite motion features.')
    if initial_states is not None and ordinary:
        raise ValueError('Ordinary return learning cannot inherit conditional parameters.')
    if updates < 0 or (updates == 0 and initial_states is None):
        raise ValueError('Calibration without optimizer updates requires a fitted parameter prior.')
    if initial_states is not None and len(initial_states) != ensemble_size:
        raise ValueError('Warm initialization must provide every ensemble member.')
    if factual_crossfit and (ordinary or not conditional_strata or ensemble_size < 2
                            or initial_states is not None):
        raise ValueError('Factual cross-fitting requires fresh conditional-stratum ensemble members.')
    if factual_calibration_kind not in ('probability', 'logit'):
        raise ValueError('Unknown factual probability calibration map.')
    snapshot_updates = tuple(sorted(set(map(int, snapshot_updates))))
    if any(step <= 0 or step > updates or step % ensemble_size for step in snapshot_updates):
        raise ValueError('Learning snapshots need positive total budgets shared equally across ensemble members.')
    initialization_sha256 = None
    if initial_states is not None:
        initialization = hashlib.sha256()
        for state in initial_states:
            for key, value in sorted(state.items()):
                initialization.update(key.encode())
                initialization.update(value.detach().cpu().numpy().tobytes())
        initialization_sha256 = initialization.hexdigest()
    if ordinary_architecture not in ('mlp', 'interaction'):
        raise ValueError('Choose the plain MLP or the explicit interaction-return comparator.')
    if ordinary and ordinary_architecture == 'mlp':
        if finite_motion_basis:
            raise ValueError('The plain return MLP does not use motion-response features.')
        from plain_return import fit_plain_return
        capacity = ensemble_size*(10*width*width+(54+5*context_conditioned+60*motion_basis+identity_conditioned)*width+1)
        return fit_plain_return(data, output, factual_bank=factual_bank, updates=updates,
            parameter_budget=capacity, seed=seed, device=device, factual_bank_weight=factual_bank_weight,
            identity_conditioned=identity_conditioned, snapshot_updates=snapshot_updates,
            root_sampling_fraction=root_sampling_fraction,
            additional_candidate_options=additional_candidate_options)
    roots = [observed_programs(root) for root in data['roots']]
    n = len(roots[0]['effect_controls'])
    if any(len(root['context']) != (8+identity_conditioned)*n+5 for root in roots):
        raise ValueError('Training roots must match the requested public identity representation.')
    conditional_strata = bool(conditional_strata and not ordinary)
    local_contrast_head = local_contrast_head if not ordinary else False
    rows, edges, anchors, diagnostics = learning_records(dict(data, roots=roots))
    row_probability = root_sampling_probabilities(rows[:, 0], root_sampling_fraction)
    root_folds = bank_folds = layout_folds = None
    if factual_crossfit:
        root_folds, bank_folds, layout_folds = factual_crossfit_groups(roots, factual_bank, ensemble_size, seed)
    def population_probability(indices):
        return root_sampling_probabilities(rows[indices, 0], root_sampling_fraction) if len(indices) else None
    response_ids = response_targets = None
    response_label_sha256 = None
    if response_supervision_weight:
        response_ids, response_targets = measured_response_records(roots, rows)
        response_probability = population_probability(response_ids)
        response_label_sha256 = hashlib.sha256(response_ids.tobytes()+response_targets.tobytes()).hexdigest()
        response_ids = torch.as_tensor(response_ids, device=device)
        response_targets = torch.as_tensor(response_targets, device=device)
    digest = hashlib.sha256()
    for root in roots:
        digest.update(np.asarray(root['context'], np.float32).tobytes())
        digest.update(np.asarray(root['effect_controls'], np.float32).tobytes())
        digest.update(np.asarray(root['options'], np.int64).tobytes())
        digest.update(np.asarray(root.get('future_weights', np.ones(len(root['futures']))), np.float32).tobytes())
        digest.update(np.asarray([(q, f, item['success']) for (q, f), item in
                                  sorted(root['outcomes'].items())], np.float64).tobytes())
    continuation_context = continuation_controls = continuation_target = continuation_remaining = None
    if factual_bank is not None:
        if factual_bank['predecessor_sha256'] != data['predecessor_sha256']:
            raise ValueError('Continuation supervision requires the common frozen predecessor.')
        arrays = [np.concatenate([record[key] for record in factual_bank['records']]).astype(np.float32)
                  for key in ('contexts', 'controls', 'values', 'remaining')]
        if arrays[0].shape[-1] != (8+identity_conditioned)*n+5:
            raise ValueError('Factual trajectories must use the same measured identity representation.')
        for array in arrays: digest.update(array.tobytes())
        continuation_context, continuation_controls, continuation_target, continuation_remaining = [
            torch.as_tensor(array, device=device) for array in arrays]
    extra_options = {tuple(map(int, option)) for option in additional_candidate_options}
    if any(len(option) != 3 or not (0 <= option[0] < 10**n) or min(option[1:]) < 0
           or (option[0] == 0 and option != (0, 0, 0)) for option in extra_options):
        raise ValueError('The transferred catalog must contain fleet-compatible finite interventions.')
    options = [(0, 0, 0)]+sorted(({tuple(option) for root in roots for option in root['options']
                                  if option[0]} | extra_options)-{(0, 0, 0)})
    context = torch.as_tensor(np.asarray([r['context'] for r in roots]), dtype=torch.float32, device=device)
    controls = torch.as_tensor(np.asarray([r['effect_controls'] for r in roots]), dtype=torch.float32, device=device)
    root_ids = torch.as_tensor(rows[:, 0], dtype=torch.long, device=device)
    codes = torch.as_tensor(rows[:, 3], dtype=torch.long, device=device)
    digits = (codes[:, None]//(10**torch.arange(n, device=device))) % 10
    timing = torch.as_tensor(rows[:, 4:6], dtype=torch.long, device=device)
    target = torch.as_tensor(rows[:, 6], dtype=torch.float32, device=device)
    factual_labels = np.asarray([roots[int(row[0])]['outcomes'][0, int(row[2])]['success'] for row in rows])
    factual_target = torch.as_tensor(factual_labels, dtype=torch.float32, device=device)
    importance_array = np.asarray([roots[int(row[0])].get('future_weights',
        np.ones(len(roots[int(row[0])]['futures'])))[int(row[2])] for row in rows], np.float32)
    importance = torch.as_tensor(importance_array, device=device)
    standardization = np.ones(len(rows), np.float32)
    factual_population = np.asarray([np.average(
        [v['success'] for (q, f), v in root['outcomes'].items() if q == 0],
        weights=[root.get('future_weights', np.ones(len(root['futures'])))[f]
                 for (q, f), v in root['outcomes'].items() if q == 0]) for root in roots], np.float32)
    if conditional_strata:
        for root_id, root in enumerate(roots):
            baseline = factual_population[root_id]
            for q in range(1, len(root['options'])):
                subset = np.flatnonzero((rows[:, 0] == root_id) & (rows[:, 1] == q))
                for stratum, population in ((0, 1-baseline), (1, baseline)):
                    index = subset[factual_labels[subset] == stratum]
                    if len(index): standardization[index] = population*importance_array[subset].sum()/importance_array[index].sum()
    standardization = torch.as_tensor(standardization, device=device)
    factual_population = torch.as_tensor(factual_population, device=device)
    edge_indices = torch.as_tensor(edges, dtype=torch.long, device=device)
    anchor_indices = torch.as_tensor(anchors, dtype=torch.long, device=device)
    edge_target_probability = population_probability(edges[:, 0])
    edge_probability = edge_target_probability
    if balanced_local_sampling:
        discordance = rows[edges[:, 0], 6] != rows[edges[:, 1], 6]
        if discordance.any():
            # Consequential single-vessel changes can be rare among neutral
            # neighbours. Allocate half the draws to them, correcting the
            # sampling probabilities below to retain the original objective.
            if edge_target_probability is None:
                edge_probability = .5/len(edges)+.5*discordance/discordance.sum()
            else:
                consequential = edge_target_probability*discordance
                edge_probability = .5*edge_target_probability+.5*consequential/consequential.sum()
    candidate_edges = edges[(rows[edges[:, 0], 1] != 0) & (rows[edges[:, 1], 1] != 0)]
    discordant = candidate_edges[rows[candidate_edges[:, 0], 6] != rows[candidate_edges[:, 1], 6]]
    rank_indices = torch.as_tensor(candidate_edges, dtype=torch.long, device=device)
    factual = np.flatnonzero(rows[:, 1] == 0)
    anchor_probability = population_probability(anchors[:, 0])
    rank_probability = population_probability(candidate_edges[:, 0])
    factual_probability = population_probability(factual)
    responses = response_basis(context, controls) if motion_basis else None
    response_index = root_ids
    if finite_motion_basis:
        if not motion_basis or local_contrast_head:
            raise ValueError('Finite motion features require the motion basis and a direct task potential.')
        keys, response_index = torch.unique(torch.cat((root_ids[:, None], timing), -1),
                                            dim=0, return_inverse=True)
        chunks = [response_basis(context[key[:, 0]], controls[key[:, 0]], key[:, 1:], dual_motion_basis)
                  for key in keys.split(128)]
        responses = tuple(torch.cat([part[j] for part in chunks]) for j in range(3))
    continuation_responses = None
    if motion_basis and continuation_context is not None:
        chunks = [response_basis(continuation_context[i:i+128], continuation_controls[i:i+128],
                  torch.zeros((len(continuation_context[i:i+128]), 2), dtype=torch.long, device=device)
                  if finite_motion_basis else None, dual_motion_basis)
                  for i in range(0, len(continuation_context), 128)]
        continuation_responses = tuple(torch.cat([part[j] for part in chunks]) for j in range(3 if finite_motion_basis else 2))
    def select_responses(records, index):
        return None if records is None else tuple(item[index] for item in records)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    def draw(count, size, probability):
        return rng.integers(count, size=size) if probability is None else rng.choice(count, size=size, p=probability)
    states = []; losses = []
    snapshots = {budget:[] for budget in snapshot_updates}
    factual_calibrations = []; factual_calibration_diagnostics = []
    calibration_snapshots = {budget:[] for budget in snapshot_updates} if factual_crossfit else None
    calibration_diagnostic_snapshots = {budget:[] for budget in snapshot_updates} if factual_crossfit else None
    per_member = [updates//ensemble_size+int(i < updates % ensemble_size) for i in range(ensemble_size)]
    for member, iterations in enumerate(per_member):
        net = JointTaskPotential(width, peer_conditioned, 3 if conditional_strata else 1,
                                context_conditioned, local_contrast_head, motion_basis,
                                residual_potential, finite_motion_basis, dual_motion_basis,
                                identity_conditioned).to(device)
        if initial_states is not None:
            net.load_state_dict(initial_states[member])
        train_parameters = list(net.parameters())
        if response_supervision_weight:
            response_node = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 8)).to(device)
            response_global = nn.Sequential(nn.Linear(3*width+8, width), nn.SiLU(), nn.Linear(width, 6)).to(device)
            train_parameters += list(response_node.parameters())+list(response_global.parameters())
        training_parameter_count = sum(parameter.numel() for parameter in train_parameters)
        optimizer = torch.optim.AdamW(train_parameters, lr=learning_rate, weight_decay=.005)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations, eta_min=learning_rate/10.)
        root_weight = torch.as_tensor(np.ones(len(roots)) if member == 0 else rng.exponential(1., len(roots)),
                                      dtype=torch.float32, device=device)
        def training_probability(probability, row_indices):
            if not factual_crossfit:
                return probability
            if not len(row_indices):
                return None
            active = root_folds[rows[row_indices, 0].astype(int)] != member
            mass = active.astype(np.float64) if probability is None else active*probability
            if mass.sum() <= 0:
                raise ValueError('A factual cross-fit member has no observations in a required population.')
            return mass/mass.sum()
        member_row_probability = training_probability(row_probability, np.arange(len(rows)))
        member_factual_probability = training_probability(factual_probability, factual)
        member_anchor_probability = training_probability(anchor_probability, anchors[:, 0])
        member_rank_probability = training_probability(rank_probability, candidate_edges[:, 0])
        member_edge_probability = training_probability(edge_probability, edges[:, 0])
        member_edge_target = training_probability(edge_target_probability, edges[:, 0])
        if factual_crossfit and len(edges):
            # Rebuild the proposal within the training fold. Held-out outcome
            # frequencies must not change this member's gradient sampling.
            member_edge_probability = member_edge_target
            if balanced_local_sampling:
                discordance = rows[edges[:, 0], 6] != rows[edges[:, 1], 6]
                consequential = member_edge_target*discordance
                if consequential.sum() > 0:
                    member_edge_probability = .5*member_edge_target+.5*consequential/consequential.sum()
        member_response_probability = (training_probability(response_probability, response_ids.cpu().numpy())
                                       if response_supervision_weight else None)
        bank_state_folds = bank_state_weights = member_bank_probability = None
        if factual_crossfit and factual_bank is not None:
            lengths = [len(record['values']) for record in factual_bank['records']]
            bank_state_folds = np.repeat(bank_folds, lengths)
            bank_state_weights = np.concatenate([np.full(length, 1./length) for length in lengths if length])
            active = (bank_state_folds != member).astype(np.float64)
            if active.sum() <= 0:
                raise ValueError('A factual cross-fit member has no factual training trajectories.')
            member_bank_probability = active/active.sum()
        @torch.no_grad()
        def calibrate_factual_head():
            ids = np.flatnonzero(root_folds == member)
            predictions = []; labels = []; masses = []
            for begin in range(0, len(ids), 128):
                index = torch.as_tensor(ids[begin:begin+128], device=device)
                estimate = net(context[index], controls[index],
                    torch.zeros((len(index), n), dtype=torch.long, device=device),
                    torch.zeros((len(index), 2), dtype=torch.long, device=device))[:, 0].sigmoid()
                predictions.append(estimate.cpu().numpy())
                labels.append(factual_population[index].cpu().numpy())
                masses.append(np.ones(len(index), np.float64))
            if bank_state_folds is not None:
                held = np.flatnonzero(bank_state_folds == member)
                for begin in range(0, len(held), 128):
                    chosen = held[begin:begin+128]; index = torch.as_tensor(chosen, device=device)
                    estimate = net(continuation_context[index], continuation_controls[index],
                        torch.zeros((len(index), n), dtype=torch.long, device=device),
                        torch.zeros((len(index), 2), dtype=torch.long, device=device),
                        select_responses(continuation_responses, index))[:, 0].sigmoid()
                    predictions.append(estimate.cpu().numpy())
                    labels.append(continuation_target[index].cpu().numpy())
                    masses.append(bank_state_weights[chosen])
            calibrate = (fit_factual_logit_calibration if factual_calibration_kind == 'logit' else
                         fit_factual_probability_calibration)
            parameters, diagnostics = calibrate(
                np.concatenate(predictions), np.concatenate(labels), np.concatenate(masses))
            diagnostics.update(held_out_fold=member, paired_roots=len(ids),
                held_out_layouts=sum(fold == member for fold in layout_folds.values()),
                training_layouts=sum(fold != member for fold in layout_folds.values()),
                rule='layout-disjoint factual calibration; equal paired-root and trajectory mass')
            return parameters, diagnostics
        def predict(index, condition=False):
            raw = net(context[root_ids[index]], controls[root_ids[index]], digits[index], timing[index],
                      select_responses(responses, response_index[index]))
            if not conditional_strata:
                return raw
            if condition:
                logits = torch.where(factual_target[index] > .5, -raw[:, 2], raw[:, 1])
                return torch.where(codes[index] == 0, 24.*factual_target[index]-12., logits)
            # Candidate contrasts do not move the independently fitted factual
            # value; they learn repair and damage conditional on that reference.
            factual = factual_population[root_ids[index]]
            value = (1-factual)*raw[:, 1].sigmoid()+factual*(1-raw[:, 2].sigmoid())
            return torch.where(codes[index] == 0, raw[:, 0], torch.logit(value.clamp(1e-6, 1-1e-6)))
        with torch.enable_grad():
            for step in range(iterations):
                ids = torch.as_tensor(draw(len(rows), 128, member_row_probability), device=device)
                logits = predict(ids)
                loss = (F.binary_cross_entropy_with_logits(logits, target[ids], reduction='none')*
                        root_weight[root_ids[ids]]*standardization[ids]*importance[ids]).sum()/importance[ids].sum()
                if response_supervision_weight:
                    positions = torch.as_tensor(draw(len(response_ids), 64, member_response_probability), device=device)
                    index = response_ids[positions]
                    encoded, representation = net.encode(context[root_ids[index]], controls[root_ids[index]],
                        digits[index], timing[index], select_responses(responses, response_index[index]))
                    prediction = torch.cat((response_node(encoded).flatten(1), response_global(representation)), -1)
                    error = F.smooth_l1_loss(prediction, response_targets[positions], reduction='none').mean(-1)
                    mass = importance[index]*root_weight[root_ids[index]]
                    loss += response_supervision_weight*(error*mass).sum()/mass.sum().clamp_min(1e-8)
                if continuation_context is not None:
                    index = torch.as_tensor(draw(len(continuation_target), 64, member_bank_probability), device=device)
                    factual_logits = net(continuation_context[index], continuation_controls[index],
                        torch.zeros((len(index), n), dtype=torch.long, device=device),
                        torch.zeros((len(index), 2), dtype=torch.long, device=device),
                        select_responses(continuation_responses, index))
                    if conditional_strata: factual_logits = factual_logits[:, 0]
                    # Common factual outcome data also train the ordinary
                    # absolute-return estimator, with the identical weight.
                    loss += factual_bank_weight*F.binary_cross_entropy_with_logits(
                        factual_logits, continuation_target[index])
                if not ordinary:
                    if conditional_strata:
                        candidates = ids[codes[ids] != 0]
                        if len(candidates):
                            loss += .5*(F.binary_cross_entropy_with_logits(predict(candidates, condition=True),
                                target[candidates], reduction='none')*importance[candidates]).sum()/importance[candidates].sum()
                    ref = torch.as_tensor(rng.choice(factual, size=32, p=member_factual_probability), device=device)
                    loss += factual_weight*(F.binary_cross_entropy_with_logits(predict(ref), target[ref],
                        reduction='none')*importance[ref]).sum()/importance[ref].sum()
                    for pairs, weight in ((edge_indices, edge_weight), (anchor_indices, anchor_weight)):
                        if not weight:
                            continue
                        probability = member_edge_probability if pairs is edge_indices else member_anchor_probability
                        selected = draw(len(pairs), 64, probability)
                        pair = pairs[torch.as_tensor(selected, device=device)]
                        if local_contrast_head and pairs is edge_indices:
                            index = pair.flatten()
                            delta, raw = net.local_contrasts(context[root_ids[index]], controls[root_ids[index]],
                                digits[index], timing[index], return_distribution=net.contrast_categorical,
                                responses=select_responses(responses, root_ids[index]))
                            distribution = None
                            if net.contrast_categorical:
                                delta, distribution = delta
                            stratum = (factual_target[pair[:, 0]].to(torch.long) if conditional_strata else
                                       torch.zeros(len(pair), dtype=torch.long, device=device))
                            predicted_difference = delta.gather(1, stratum[:, None]).flatten()
                            if conditional_strata:
                                conditioned = torch.where(factual_target[index] > .5, -raw[:, 2], raw[:, 1]).sigmoid()
                                conditioned = torch.where(codes[index] == 0, factual_target[index], conditioned)
                            else:
                                conditioned = raw.squeeze(-1).sigmoid()
                            node_difference = conditioned.reshape(-1, 2).diff(dim=1).flatten()
                        else:
                            values = predict(pair.flatten(), condition=conditional_strata).sigmoid().reshape(-1, 2)
                            predicted_difference = values[:, 1]-values[:, 0]
                            node_difference = None
                        difference = target[pair[:, 1]]-target[pair[:, 0]]
                        mass = importance[pair[:, 0]]
                        if pairs is edge_indices and probability is not None:
                            if member_edge_target is None:
                                mass = mass/torch.as_tensor(len(pairs)*probability[selected], device=device, dtype=mass.dtype)
                            else:
                                mass = mass*torch.as_tensor(member_edge_target[selected]/probability[selected],
                                                           device=device, dtype=mass.dtype)
                        error = (predicted_difference-difference).square()
                        if node_difference is not None:
                            if distribution is not None:
                                selected_distribution = distribution[torch.arange(len(pair), device=device), stratum]
                                error = F.cross_entropy(selected_distribution, (difference+1).to(torch.long), reduction='none')
                            # The endpoint values and local consequence head
                            # jointly fit the same measured single-boat effect.
                            error = error+.5*(node_difference-difference).square()+.25*(node_difference-predicted_difference).square()
                        loss += weight*(error*
                            root_weight[root_ids[pair[:, 0]]]*mass).sum()/mass.sum()
                    if rank_weight and len(rank_indices):
                        pair = rank_indices[torch.as_tensor(draw(len(rank_indices), 32, member_rank_probability), device=device)]
                        pair_logits = predict(pair.flatten(), condition=conditional_strata).reshape(-1, 2)
                        preferred = (target[pair[:, 1]] > target[pair[:, 0]]).to(target.dtype)
                        informative = (target[pair[:, 1]] != target[pair[:, 0]]).to(target.dtype)
                        mass = importance[pair[:, 0]]
                        # Normalize over the entire local comparison population.
                        # Conditioning only on discordant outcomes removes their
                        # frequency and can overweight rare task changes relative
                        # to the value-fitting terms.
                        # Neutral futures have zero rank gradient but remain
                        # in its normalization and in the value/difference fit.
                        loss += rank_weight*(F.binary_cross_entropy_with_logits(pair_logits[:, 1]-pair_logits[:, 0],
                            preferred, reduction='none')*informative*mass).sum()/mass.sum()
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(train_parameters, 5.); optimizer.step()
                schedule.step()
                budget = (step+1)*ensemble_size
                if budget in snapshots:
                    snapshots[budget].append({key:value.detach().cpu().clone()
                                             for key,value in net.state_dict().items()})
                    if factual_crossfit:
                        parameters, diagnostics = calibrate_factual_head()
                        calibration_snapshots[budget].append(parameters)
                        calibration_diagnostic_snapshots[budget].append(diagnostics)
                if (step+1) % 1000 == 0 or step+1 == iterations:
                    print(dict(phase='joint_potential_fit', ordinary=ordinary, member=member,
                               update=step+1, loss=float(loss.detach())), flush=True)
        states.append({k: v.detach().cpu() for k, v in net.state_dict().items()})
        losses.append(float(loss.detach()) if iterations else None)
        if factual_crossfit:
            parameters, diagnostics = calibrate_factual_head()
            factual_calibrations.append(parameters); factual_calibration_diagnostics.append(diagnostics)
    recipe = dict(width=width, states=states, peer_conditioned=peer_conditioned, ordinary=ordinary,
        context_conditioned=context_conditioned,
        identity_conditioned=identity_conditioned,
        residual_potential=residual_potential,
        motion_basis=motion_basis,
        finite_motion_basis=finite_motion_basis,
        dual_motion_basis=dual_motion_basis,
        joint_factual_calibration=joint_factual_calibration and not ordinary,
        intervention_locality=intervention_locality and not ordinary,
        local_contrast_head=local_contrast_head, projection_strength=projection_strength, projection_steps=projection_steps,
        contexts=context.cpu(), controls=controls.cpu(), options=options,
        corrections=np.zeros((len(roots), len(options)), np.float32),
        member_corrections=np.zeros((ensemble_size, len(roots), len(options)), np.float32),
        uncertainty=np.zeros((len(roots), len(options)), np.float32),
        observed=np.zeros((len(roots), len(options)), bool), bandwidth=bandwidth, margin=margin,
        confidence=confidence, global_penalty=global_penalty, decision_period=5, intervention_budget=0,
        context_time_scale=context_time_scale, interpolation_power=interpolation_power,
        conditional_strata=conditional_strata,
        neural_factual_calibration=np.asarray(factual_calibrations) if factual_crossfit else None,
        neural_factual_calibration_kind=factual_calibration_kind,
        factual_difference_transport=factual_difference_transport and not ordinary,
        factual_values=factual_population.cpu() if not ordinary else None,
        project_task_value=True, commit_factual=False,
        replan_delayed=False,
        continuation_contexts=None if ordinary or continuation_context is None else continuation_context.cpu(),
        continuation_controls=None if ordinary or continuation_controls is None else continuation_controls.cpu(),
        continuation_values=None if ordinary or continuation_target is None else continuation_target.cpu(),
        continuation_remaining=None if ordinary or continuation_remaining is None else continuation_remaining.cpu(),
        temporal_value_bounds=not ordinary and factual_bank is not None)
    lookup = {option: i for i, option in enumerate(options)}; local_fits = []
    if not ordinary:
        learner = LocalDifferenceDecision(**recipe).to(device).eval()
        mappings = [[lookup[tuple(option)] for option in root['options']] for root in roots]
        with torch.no_grad():
            # Independent scalar potentials need only the options actually
            # observed at this root. The graph-based contrast head still
            # requires its complete catalog for coupled reconstruction.
            if not local_contrast_head:
                member_predictions = [learner.values(context[i:i+1], controls[i:i+1],
                    candidate_indices=torch.as_tensor(mapping, dtype=torch.long, device=device))[:, 0].cpu().numpy()
                    for i, mapping in enumerate(mappings)]
            else:
                member_predictions = [learner.values(context[i:i+1], controls[i:i+1])[:, 0, mapping].cpu().numpy()
                                      for i, mapping in enumerate(mappings)]
        for i, root in enumerate(roots):
            mapping = mappings[i]
            members = member_predictions[i]
            prediction = members.mean(0)
            values, errors, counts, detail = reconcile_root(root, prediction, edge_weight=edge_weight,
                effective_future_confidence=effective_future_confidence)
            residual = (values-values[0])-(prediction-prediction[0])
            observed = counts > 0
            recipe['corrections'][i, np.asarray(mapping)[observed]] = residual[observed]
            for member in range(ensemble_size):
                difference = members[member]-members[member, 0]
                recipe['member_corrections'][member, i, np.asarray(mapping)[observed]] = ((values-values[0])-difference)[observed]
            recipe['uncertainty'][i, np.asarray(mapping)[observed]] = errors[observed]
            recipe['observed'][i, np.asarray(mapping)[observed]] = True
            local_fits.append(detail)
    model = copy.deepcopy(data['model'])
    # Replace the value module and its calibration. Only explicitly supplied
    # parameter initialization can be retained from an earlier learner.
    model['successor_task_value'] = (dict(kind='plain_joint_return', width=width, states=states,
        peer_conditioned=peer_conditioned, context_conditioned=context_conditioned,
        motion_basis=motion_basis, residual_potential=residual_potential, finite_motion_basis=finite_motion_basis,
        dual_motion_basis=dual_motion_basis,
        identity_conditioned=identity_conditioned,
        options=options, defenders=n, decision_period=5) if ordinary else
        dict(kind='local_difference_joint', **recipe))
    removal = tuple(model.get('mechanism_ablation', (False, False, False)))+(False,)*3
    model['mechanism_ablation'] = (removal[0], False, removal[2])
    model.update(gate_period=1, physical_interventions=True, fixed_intervention_clock=True, success_objective=True)
    model['successor_task_fit'] = dict(ordinary=ordinary, roots=len(roots),
        root_sampling_fraction=root_sampling_fraction,
        factual_crossfit=factual_crossfit,
        factual_calibration_kind=factual_calibration_kind,
        factual_calibration_diagnostics=factual_calibration_diagnostics,
        factual_crossfit_root_folds=None if root_folds is None else root_folds.tolist(),
        factual_crossfit_trajectory_folds=None if bank_folds is None else bank_folds.tolist(),
        factual_calibration_parameters=2*ensemble_size if factual_crossfit else 0,
        unique_terminal_observations=data.get('measured_terminal_observations', len(rows)),
        supervised_node_labels=len(rows)+(0 if continuation_target is None else len(continuation_target)),
        reused_delayed_labels=data.get('reused_delayed_labels', 0),
        reused_terminal_prefix_labels=data.get('reused_terminal_prefix_labels', 0),
        paired_local_observations=len(edges), factual_comparisons=len(anchors), updates=updates,
        response_supervision_weight=response_supervision_weight,
        measured_response_labels=0 if response_ids is None else len(response_ids),
        response_label_sha256=response_label_sha256,
        response_auxiliary_heads_at_inference=False,
        training_parameters_per_member=training_parameter_count,
        member_updates=per_member, width=width, seed=seed, objective='absolute return' if ordinary else
        'whole-plan outcome likelihood plus paired single-vessel value differences and factual contrasts',
        local_reconciliation=not ordinary, inherited_stages=False, root_diagnostics=diagnostics,
        return_representation='candidate-interaction network',
        local_fits=local_fits, last_loss=losses, training_data=data.get('training_data'),
        training_observation_sha256=digest.hexdigest(), execution_revision=33,
        difference_weight=edge_weight, factual_contrast_weight=anchor_weight,
        local_rank_weight=0. if ordinary else rank_weight, discordant_local_observations=len(discordant),
        local_rank_sampling='all nonfactual single-vessel comparisons with neutral outcomes in the normalization',
        factual_weight=0. if ordinary else factual_weight, factual_value_anchored=not ordinary,
        context_conditioned_nodes=context_conditioned,
        public_vessel_identity=identity_conditioned,
        duration_centered_potential=residual_potential,
        common_motion_response_features=motion_basis,
        finite_window_motion_features=finite_motion_basis,
        dual_motion_features=dual_motion_basis,
        motion_feature_control_seconds=.2 if finite_motion_basis else None,
        motion_feature_integration_seconds=.05 if finite_motion_basis else None,
        learned_single_vessel_contrasts=local_contrast_head,
        probability_corrected_local_sampling=balanced_local_sampling and not ordinary,
        learning_rate_schedule=f'cosine {learning_rate:g} to {learning_rate/10.:g} over each member budget',
        initialization_parameter_sha256=initialization_sha256,
        frozen_parameter_calibration=updates == 0,
        effective_future_confidence=effective_future_confidence,
        additional_candidate_catalog_size=len(extra_options-{(0, 0, 0)}),
        inference_joint_value_reconstruction=local_contrast_head,
        receding_option_improvement=not ordinary,
        joint_factual_calibration=joint_factual_calibration and not ordinary,
        intervention_conditioned_locality=intervention_locality and not ordinary,
        factual_choice_keeps_future_decisions=True,
        delayed_choice_keeps_future_decisions=False,
        conditional_factual_strata=conditional_strata,
        empirical_factual_reference=not ordinary,
        development_future_weight=data.get('development_future_weight', 1.),
        candidate_catalog='all sampled complete programs, including negative outcomes',
        uncertainty_estimator='separate paired rescue lower and harm upper Beta bounds' if not ordinary else None,
        predecessor_sha256=data.get('predecessor_sha256'),
        factual_trajectory_replays=0 if factual_bank is None else factual_bank['trajectories'],
        paired_continuation_state_labels=0 if factual_bank is None else factual_bank.get('paired_continuation_state_labels', 0),
        factual_bank_training_data=None if factual_bank is None else factual_bank.get('training_data'),
        additional_factual_state_labels=0 if continuation_target is None else len(continuation_target),
        factual_bank_weight=0. if factual_bank is None else factual_bank_weight,
        simulation_steps=data.get('simulation_steps', 0)+(0 if factual_bank is None else factual_bank['simulation_steps']))
    if stratified_local_calibration and not ordinary:
        del learner
        model = calibrate_conditional_differences(model, data, device=device)
    if snapshots:
        if any(len(states) != ensemble_size for states in snapshots.values()):
            raise ValueError('A learning snapshot must contain every independently fitted member.')
        model['learning_snapshots'] = dict(kind='actual optimizer states at equal per-member budgets',
            final_updates=updates, states=snapshots,
            calibration='recompute paired residuals against each saved potential before evaluation')
        if factual_crossfit:
            model['learning_snapshots']['neural_factual_calibration'] = calibration_snapshots
            model['learning_snapshots']['factual_calibration_diagnostics'] = calibration_diagnostic_snapshots
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); torch.save(model, output)
    return model


@torch.no_grad()
def calibrate_conditional_differences(model, data, *, device='cuda:0'):
    """Joint local node/edge fits within each observed factual stratum.

    The two fitted conditional consequences are combined only at the query
    state's factual probability. No unobserved rescue or harm is treated as
    empirical evidence. Neural weights, data and optimizer budget are unchanged.
    """
    result = copy.deepcopy(model); recipe = result['successor_task_value']
    if recipe.get('ordinary') or recipe.get('kind') != 'local_difference_joint':
        raise ValueError('Conditional calibration requires the paired-difference learner.')
    conditional_heads = recipe.get('conditional_strata', False)
    roots = [observed_programs(root) for root in data['roots']]
    options = list(map(tuple, np.asarray(recipe['options']).tolist()))
    if len(roots) != len(recipe['contexts']):
        raise ValueError('Calibration roots must match the fitted data.')
    clean = {k: v for k, v in recipe.items() if k not in
             ('kind', 'stratum_member_corrections', 'stratum_uncertainty', 'stratum_observed')}
    learner = LocalDifferenceDecision(**clean).to(device).eval()
    members = len(learner.models); shape = (2, len(roots), len(options))
    corrections = np.zeros((2, members, len(roots), len(options)), np.float32)
    uncertainty = np.zeros(shape, np.float32); observed = np.zeros(shape, bool)
    lookup = {q: i for i, q in enumerate(options)}; supported = [0, 0]
    weight = result['successor_task_fit']['difference_weight']
    for i, root in enumerate(roots):
        mapping = np.asarray([lookup[tuple(q)] for q in root['options']])
        scalar_predictions = None
        if not conditional_heads:
            scalar_predictions = learner.values(learner.contexts[i:i+1],
                learner.controls[i:i+1])[:, 0].cpu().numpy()
        for stratum in (0, 1):
            outcomes = {key: value for key, value in root['outcomes'].items()
                if (0, key[1]) in root['outcomes'] and
                int(root['outcomes'][0, key[1]]['success']) == stratum}
            if not outcomes or not any(q for q, _ in outcomes):
                continue
            subset = dict(root, outcomes=outcomes)
            if conditional_heads:
                predictions = learner.values(learner.contexts[i:i+1], learner.controls[i:i+1],
                                              stratum=stratum)[:, 0].cpu().numpy()
                differences = predictions[:, mapping]-stratum
                prior = predictions[:, mapping]
            else:
                differences = scalar_predictions[:, mapping]-scalar_predictions[:, :1]
                prior = np.clip(stratum+differences, 0., 1.)
            values, errors, counts, _ = reconcile_root(subset, prior.mean(0), edge_weight=weight,
                effective_future_confidence=result['successor_task_fit'].get('effective_future_confidence', False))
            available = counts > 0; selected = mapping[available]
            for member in range(members):
                residual = (values-stratum)-differences[member]
                corrections[stratum, member, i, selected] = residual[available]
            uncertainty[stratum, i, selected] = errors[available]
            observed[stratum, i, selected] = True
            supported[stratum] += 1
    recipe.update(stratum_member_corrections=corrections, stratum_uncertainty=uncertainty,
                  stratum_observed=observed)
    result['successor_task_fit'].update(execution_revision=26 if conditional_heads else 36,
        conditional_local_difference_calibration=True, calibrated_factual_stratum_roots=supported,
        conditional_local_calibration='separate anchored joint node/edge objectives, query-state factual mixture',
        calibration_value_model='conditional heads' if conditional_heads else 'shared scalar task potential')
    return result


def limit_decision_windows(model, *, maximum_delay=0, maximum_span=None, decision_period=1):
    """Replan from current observations instead of freezing a long prefix.

    Both estimators can use this common action menu. Training labels and
    parameters stay intact. The retained graph weights preserve every local
    value equation because windows only share the fixed factual node.
    """
    result = copy.deepcopy(model); recipe = result['successor_task_value']
    if recipe['kind'] != 'local_difference_joint':
        raise ValueError('Window selection applies to the local-intervention estimator.')
    options = np.asarray(recipe['options'], dtype=np.int64)
    keep = options[:, 1] <= maximum_delay
    if maximum_span is not None:
        keep &= options[:, 2] <= maximum_span
    keep[0] = True
    selected = np.flatnonzero(keep)
    if recipe.get('local_contrast_head'):
        n = np.asarray(recipe['controls']).shape[1]
        pairs = torch.as_tensor(local_edges(options.tolist(), n), dtype=torch.long)
        if recipe.get('comparison_weights_override') is None:
            degrees = torch.bincount(pairs.flatten(), minlength=len(options)).to(torch.float32)
            weights = recipe.get('projection_strength', 2.)/(1.+.5*(degrees[pairs[:, 0]]+degrees[pairs[:, 1]]))
        else:
            weights = torch.as_tensor(recipe['comparison_weights_override'])
        surviving = keep[pairs.numpy()].all(1)
        index = np.full(len(options), -1, np.int64); index[selected] = np.arange(len(selected))
        expected = np.asarray(local_edges(options[selected].tolist(), n))
        if not np.array_equal(index[pairs.numpy()[surviving]], expected):
            raise ValueError('Window restriction changed the retained single-vessel comparisons.')
        recipe['comparison_weights_override'] = weights[surviving]
    recipe['options'] = [tuple(row) for row in options[selected].tolist()]
    for key in ('corrections', 'uncertainty', 'observed'):
        recipe[key] = np.asarray(recipe[key])[:, selected].copy()
    if recipe.get('member_corrections') is not None:
        recipe['member_corrections'] = np.asarray(recipe['member_corrections'])[:, :, selected].copy()
    for key in ('stratum_member_corrections', 'stratum_uncertainty', 'stratum_observed'):
        if recipe.get(key) is not None:
            recipe[key] = np.asarray(recipe[key])[..., selected].copy()
    recipe.update(decision_period=decision_period, intervention_budget=0, commit_factual=False)
    result['successor_task_fit'].update(execution_revision=24,
        current_observation_replanning=True, decision_period=decision_period,
        maximum_intervention_delay=maximum_delay, maximum_intervention_span=maximum_span,
        trained_candidate_options=len(options), executed_candidate_options=len(selected))
    return result


def refresh_local_calibration(model, data, *, device='cuda:0', effective_future_confidence=None,
                              confidence_only=False, **changes):
    """Re-estimate the paired potential after changing factual calibration.

    Neural parameters, labels and optimizer updates are held fixed. The local
    residuals must be recomputed against the modified factual-anchored value,
    rather than carrying a residual fitted to a different reference value.
    """
    result = copy.deepcopy(model)
    recipe = result['successor_task_value']
    if recipe['ordinary']:
        raise ValueError('An ordinary-return replacement has no local calibration.')
    if confidence_only and (changes or recipe.get('stratum_member_corrections') is not None):
        raise ValueError('Confidence-only calibration must preserve the common scalar potential.')
    if result['successor_task_fit']['predecessor_sha256'] != data['predecessor_sha256']:
        raise ValueError('Recalibration requires the original paired predecessor.')
    recipe.update(changes)
    if effective_future_confidence is None:
        effective_future_confidence = result['successor_task_fit'].get('effective_future_confidence', False)
    roots = [observed_programs(root) for root in data['roots']]
    context = torch.as_tensor(np.stack([r['context'] for r in roots]), dtype=torch.float32, device=device)
    controls = torch.as_tensor(np.stack([r['effect_controls'] for r in roots]), dtype=torch.float32, device=device)
    torch.testing.assert_close(context, torch.as_tensor(recipe['contexts'], device=device))
    torch.testing.assert_close(controls, torch.as_tensor(recipe['controls'], device=device))
    learner = LocalDifferenceDecision(**{k: v for k, v in recipe.items() if k != 'kind'}).to(device).eval()
    options = [tuple(option) for option in np.asarray(recipe['options']).tolist()]
    lookup = {option: q for q, option in enumerate(options)}
    mappings = [np.asarray([lookup[tuple(option)] for option in root['options']]) for root in roots]
    reduced_calibration = effective_future_confidence and not recipe.get('local_contrast_head')
    with torch.no_grad():
        if reduced_calibration:
            predictions = [learner.values(context[i:i+1], controls[i:i+1],
                candidate_indices=torch.as_tensor(mapping, dtype=torch.long, device=device))[:, 0].cpu().numpy()
                for i, mapping in enumerate(mappings)]
        else:
            members = torch.cat([learner.values(context[i:i+1], controls[i:i+1])
                                 for i in range(len(roots))], 1).cpu().numpy()
            predictions = [members[:, i, mapping] for i, mapping in enumerate(mappings)]
    member_count = len(learner.models)
    if not confidence_only:
        recipe['corrections'] = np.zeros((len(roots), len(options)), np.float32)
        recipe['member_corrections'] = np.zeros((member_count, len(roots), len(options)), np.float32)
    recipe['uncertainty'] = np.zeros((len(roots), len(options)), np.float32)
    if not confidence_only:
        recipe['observed'] = np.zeros((len(roots), len(options)), bool)
    details = []
    weight = result['successor_task_fit']['difference_weight']
    for i, root in enumerate(roots):
        mapping = mappings[i]; members = predictions[i]; prediction = members.mean(0)
        value, error, count, detail = reconcile_root(root, prediction, edge_weight=weight,
            effective_future_confidence=effective_future_confidence)
        observed = count > 0; selected = mapping[observed]
        difference = value-value[0]
        if confidence_only:
            # Recover the already fitted joint gain, rather than changing it
            # by resolving the graph while investigating uncertainty alone.
            prior = model['successor_task_value']
            residual = (np.asarray(prior['corrections'])[i, mapping][None]
                        if prior.get('member_corrections') is None else
                        np.asarray(prior['member_corrections'])[:, i, mapping])
            difference = (members-members[:, :1]+residual).mean(0)
            lower = ((1-detail['factual_value'])*np.asarray(detail['repair_lower'])
                     -detail['factual_value']*np.asarray(detail['damage_upper']))
            error = np.maximum(0., difference-lower); error[0] = 0.
            if not np.array_equal(np.asarray(recipe['observed'])[i, mapping], observed):
                raise ValueError('Confidence calibration must retain the same observed comparisons.')
        else:
            recipe['corrections'][i, selected] = (difference-(prediction-prediction[0]))[observed]
            for member in range(member_count):
                recipe['member_corrections'][member, i, selected] = (
                    difference-(members[member]-members[member, 0]))[observed]
        recipe['uncertainty'][i, selected] = error[observed]
        recipe['observed'][i, selected] = True
        details.append(detail)
    result['successor_task_fit'].update(local_fits=details, execution_revision=22,
        joint_factual_calibration=recipe.get('joint_factual_calibration', False),
        receding_option_improvement=not recipe['intervention_budget'],
        effective_future_confidence=effective_future_confidence,
        calibration_uses_observed_catalog_only=reduced_calibration,
        confidence_only_calibration=confidence_only,
        recalibrated_against_current_factual_value=True)
    return result


def development_roots(references, failures=20, successes=4, late=False):
    """Count distinct layouts, rather than repeated noise evaluations."""
    groups = []
    for reference in references:
        selected = []
        seen = set()
        for success, quota in ((False, failures), (True, successes)):
            count = 0
            for (scene, future), row in zip(reference['tasks'], reference['rows']):
                if scene[0] in seen or bool(row['success']) != success:
                    continue
                seen.add(scene[0]); count += 1
                lookback = (40 if row['outcome'] == 2 else 120) if late and not success else None
                selected.append((scene, future, lookback))
                if count >= quota:
                    break
        groups.append(selected)
    return [group[i] for i in range(max(map(len, groups), default=0)) for group in groups if i < len(group)]


def collect_local_tasks(policy, root_tasks, output, *, spans=(10, 40, 200), repetitions=2,
                        confirmation=6, workers=12, seed=42, existing_data=None, adaptive_screen=True,
                        interaction_order=2, screen_limit=0, screen_strategy='fleet_first',
                        neighborhood_centers=0, record_continuations=False, context_policy=None,
                        neighborhood_width=2, inference_batch=0, inference_device='cpu',
                        physical_response_horizon=0, opponent_checkpoint=None):
    """Adaptive paired sampling, confirming selected plans on fresh futures.

    The complete outcome table is shared by conditional and ordinary fits.
    Candidate deletion neighbours receive the same confirmation futures.
    """
    import conditional_fusion_value as c
    import develop_source_fusion as d
    import run_remote_evidence as r
    from functools import partial
    if physical_response_horizon not in (0, 20):
        raise ValueError('Physical response supervision uses the common 20-tick horizon.')
    r.initialize_source_branches(str(d.SOURCE), opponent_checkpoint)
    opponent_sha256 = r._SOURCE_ATTACKER_SHA256
    policy = Path(policy); output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    model = torch.load(policy, map_location='cpu', weights_only=False)
    if model.get('successor_task_value') is not None:
        raise ValueError('Collect a fresh whole-mechanism comparison from F00 or F10.')
    actor = d.joint_load(str(policy), policy.stat().st_mtime_ns)
    behavior = actor if context_policy is None else d.joint_load(
        str(context_policy), Path(context_policy).stat().st_mtime_ns)
    data = existing_data
    if data is not None and data['predecessor_sha256'] != hashlib.sha256(policy.read_bytes()).hexdigest():
        raise ValueError('Resumed local observations require the same frozen predecessor.')
    if data is not None and data.get('opponent_sha256') != opponent_sha256:
        raise ValueError('Resumed local observations require the same frozen opponent.')
    if data is not None and (data.get('recipe', {}).get('inference_batch', 0) != inference_batch or
                             data.get('recipe', {}).get('inference_device', 'cpu') != inference_device or
                             data.get('recipe', {}).get('physical_response_horizon', 0) != physical_response_horizon):
        raise ValueError('Resumed local observations require the same inference protocol.')
    roots = [] if data is None else list(data['roots'])
    known = {(tuple(root['scene']), root['prefix_future'], root['lookback']) for root in roots}
    requested = {(tuple(scene), future, lookback) for scene, future, lookback in root_tasks}
    steps = 0 if existing_data is None else existing_data['simulation_steps']
    coefficients = np.asarray([[-1., -1.]]+[[a, b] for a in (0., .5, 1.) for b in (0., .5, 1.)])
    rng = np.random.default_rng(seed)
    with (ProcessPoolExecutor(max_workers=workers, initializer=r.initialize_source_branches,
          initargs=(str(d.SOURCE), opponent_checkpoint)) if workers > 1 else nullcontext(None)) as pool:
        for scene, prefix_future, lookback in root_tasks:
            key = (tuple(scene), prefix_future, lookback)
            if key in known:
                continue
            np.random.seed(scene[0]); torch.manual_seed(scene[0])
            env = d.numerical_source().TADEnv(scene[1]); obs, _ = env.reset(agility=scene[2], noisy_agility=False)
            if prefix_future is not None: np.random.seed(prefix_future)
            elapsed = 0
            selected_action = None
            if lookback is not None:
                history = []; done = 0
                while not done:
                    before = (elapsed, copy.deepcopy(env), obs.copy(), np.random.get_state())
                    with torch.no_grad(): action = behavior(d.mechanism_tensor(d.joint_packet(env, obs)), deterministic=True)[0][0].numpy()
                    history.append((*before, action.copy()))
                    obs, _, done, _, count = d.joint_advance(env, obs, behavior, action); elapsed += count; steps += count
                if lookback in ('first_intervention', 'last_intervention'):
                    decisions = [h for h in history if h[4].shape[-1] >= 13 and (h[4][:, 11:13] >= 0.).any()]
                    chosen = (decisions[0] if lookback == 'first_intervention' else decisions[-1]) if decisions else history[0]
                elif lookback == 'uniform_eligible':
                    eligible = [h for h in history if h[0] % 5 == 0]
                    selection_rng = np.random.default_rng(np.random.SeedSequence(
                        [seed, scene[0], 0 if prefix_future is None else prefix_future, 39797]))
                    chosen = eligible[int(selection_rng.integers(len(eligible)))]
                else:
                    cutoff = max(0, (elapsed-lookback)//5*5)
                    chosen = max((h for h in history if h[0] <= cutoff), key=lambda h: h[0])
                elapsed, env, obs, state, selected_action = chosen
                np.random.set_state(state)
            tensor = d.mechanism_tensor(d.joint_packet(env, obs))
            with torch.no_grad():
                control = actor(tensor, deterministic=True, gate_only=True)[0][0].numpy()[:, :3]
                context, order = c.intervention_task_context(tensor, True, True)
                effect = c.paired_effect_controls(actor, tensor, torch.as_tensor(control[None]))
            n = scene[1]; inverse = order[0].argsort().numpy()
            codes = connected_programs(tensor, order, interaction_order)
            options = [(0, 0, 0)]+[(code, 0, span) for code in codes for span in spans]
            focus_option = None
            focus_codes = set()
            if selected_action is not None and selected_action.shape[-1] >= 13 and (selected_action[:, 11:13] >= 0.).any():
                # Re-measure the actual decision that reached this context.
                # Only the sampling prefix follows the current learned policy;
                # every paired branch still has the same frozen predecessor.
                preference = selected_action[order[0].numpy(), 11:13]
                digits = ((preference[:, None]-coefficients[None])**2).sum(-1).argmin(-1)
                code = int((digits*(10**np.arange(n))).sum())
                delay = int(selected_action[0, 8]); span = int(selected_action[0, 4])-delay
                focus_option = (code, delay, span)
                neighbors = single_vessel_neighborhood(code, n)
                focus_codes = {code, *neighbors}
                if neighborhood_width and context_policy is not None:
                    # Spend new rollouts on each boat's promising or uncertain
                    # local alternatives. Retain the decision actually made,
                    # its deletion path and every sampled negative outcome.
                    value = behavior.successor_task_value
                    device = next(value.models[0].parameters()).device
                    cc = context.to(device)
                    uu = effect[:, order[0]].to(device)
                    programs = torch.as_tensor((np.asarray(neighbors)[:, None]//(10**np.arange(n))) % 10,
                        dtype=torch.long, device=device)
                    timing = torch.tensor([[delay, span]], device=device).expand(len(neighbors), -1)
                    predictions = []
                    with torch.no_grad():
                        for net in value.models:
                            raw = net(cc.expand(len(neighbors), -1), uu.expand(len(neighbors), -1, -1), programs, timing).sigmoid()
                            if value.conditional_strata:
                                factual = net(cc, uu, torch.zeros((1, n), dtype=torch.long, device=device),
                                    torch.zeros((1, 2), dtype=torch.long, device=device)).sigmoid()[0, 0]
                                raw = (1-factual)*raw[:, 1]+factual*(1-raw[:, 2])
                            predictions.append(raw)
                    predictions = torch.stack(predictions)
                    score = (predictions.mean(0)+predictions.std(0, unbiased=False)).cpu().numpy()
                    center = (code//(10**np.arange(n))) % 10
                    changed = (programs.cpu().numpy() != center).argmax(-1)
                    focus_codes = {code}
                    for boat in range(n):
                        ids = np.flatnonzero(changed == boat)
                        focus_codes.update(neighbors[j] for j in ids[np.argsort(-score[ids], kind='stable')[:neighborhood_width]])
                        if center[boat]:focus_codes.add(code-int(center[boat])*10**boat)
                    parent = code
                    while parent:
                        boat = int(np.flatnonzero((parent//(10**np.arange(n))) % 10)[-1])
                        parent -= (parent//10**boat % 10)*10**boat
                        focus_codes.add(parent)
                options = list(dict.fromkeys(options+
                    [(q, delay, span) if q else (0, 0, 0) for q in sorted(focus_codes)]))
            one = np.ones((n, 1), np.float32)
            def intervention_action(option):
                code, delay, span = option
                if not code:
                    return control
                preference = coefficients[(code//(10**np.arange(n))) % 10][inverse].astype(np.float32)
                first = preference if delay == 0 else np.full_like(preference, -1.)
                return np.concatenate((control, -one, (delay+span)*one, -2*one, np.zeros((n, 2), np.float32),
                    delay*one, first, preference, np.zeros_like(one)), -1)
            actions = [intervention_action(option) for option in options]
            random_state = np.random.get_state()
            futures = [None]+[int(np.random.SeedSequence([scene[0], elapsed, seed, i, 66719]).generate_state(1)[0])
                              for i in range(max(repetitions, confirmation)+1)]
            outcomes = {}
            def sample(keys):
                nonlocal steps
                missing = sorted(set(keys)-outcomes.keys())
                jobs = [(str(policy), dict(environment=env, observation=obs, random_state=random_state, action=actions[q]),
                         scene, futures[f], max(physical_response_horizon,
                             max(1, options[q][1]+options[q][2]) if record_continuations else 1))
                        for q, f in missing]
                if inference_batch and jobs:
                    # Keep batch membership independent of the worker count.
                    # Short screening rounds still expose up to eight jobs;
                    # larger confirmation rounds share their neural inference.
                    batch = min(inference_batch, max(1, (len(jobs)+7)//8))
                    groups = [jobs[start:start+batch] for start in range(0, len(jobs), batch)]
                    measure = partial(source_successor_batches, batch_size=batch, device=inference_device)
                    results = pool.map(measure, groups, chunksize=1) if pool else map(measure, groups)
                    iterator = (row for group in results for row in group)
                else:
                    iterator = pool.map(r.source_successor_branch, jobs, chunksize=1) if pool else map(r.source_successor_branch, jobs)
                for entry, row in zip(missing, iterator):
                    outcomes[entry] = dict(success=float(row['success']), outcome=row['outcome'], duration=row['duration'])
                    response = row.get('response')
                    if physical_response_horizon and response is not None and response['steps'] <= physical_response_horizon:
                        successor = d.mechanism_tensor(response['packet'])
                        future_context, future_order = c.intervention_task_context(successor, True, True)
                        # Canonical distance order may change during the branch.
                        # Restore physical boat identities before making a delta.
                        nodes = future_context[0, :8*n].reshape(n, 8)[future_order[0].argsort()][order[0]]
                        following = torch.cat((nodes.flatten(), future_context[0, 8*n:])).numpy()
                        outcomes[entry].update(following=following, response_steps=response['steps'],
                            response_sampling='physical_tick')
                    if record_continuations and response is not None and not response['terminal']:
                        # At the end of the finite option, the remaining
                        # trajectory follows the frozen predecessor. Its
                        # already measured terminal result is therefore a
                        # factual-value label at this reached public state.
                        successor = d.mechanism_tensor(response['packet'])
                        with torch.no_grad():
                            native = actor(successor, deterministic=True, gate_only=True)[0]
                            future_context, future_order = c.intervention_task_context(successor, True, True)
                            future_control = c.paired_effect_controls(actor, successor, native)
                        outcomes[entry].update(continuation_context=future_context[0].numpy(),
                            continuation_controls=future_control[0, future_order[0]].numpy(),
                            continuation_order=future_order[0].numpy(),
                            continuation_remaining=row['physical_steps']-response['steps'])
                    steps += row['physical_steps']
            if adaptive_screen:
                sample((0, f) for f in range(max(repetitions, confirmation)))
                # Search complete fleet plans early, then acquire the local
                # path connecting each successful plan to its predecessor.
                # A limited screen must not always query the smallest code
                # in each group. New randomized recipes cover other boats
                # and mixtures while retaining the same size/window balance.
                root_seed = np.random.SeedSequence([seed, scene[0],
                    0 if prefix_future is None else prefix_future, elapsed, 47267])
                pending = candidate_screen_order(options, n, interaction_order,
                    screen_strategy, root_seed)
                if screen_limit:
                    limit = max(screen_limit, 48) if context_policy is not None and focus_option is None else screen_limit
                    pending = pending[:limit]
                if focus_option is not None:
                    focused = [q for q in range(1, len(options)) if options[q][1:] == focus_option[1:] and options[q][0] in focus_codes]
                    pending = list(dict.fromkeys(focused+pending))
                screened = []
                for start in range(0, len(pending), 12):
                    batch = pending[start:start+12]
                    sample((q, 0) for q in batch); screened.extend(batch)
                    effective = [q for q in screened if outcomes[q, 0]['success'] > outcomes[0, 0]['success']]
                    if len(effective) >= 3:
                        break
                lookup = {option: i for i, option in enumerate(options)}
                paired = set(screened)
                for q in screened:
                    code, delay, span = options[q]
                    while code:
                        boat = int(np.flatnonzero((code//(10**np.arange(n))) % 10)[-1])
                        code -= (code//10**boat % 10)*10**boat
                        parent = 0 if not code else lookup.get((code, delay, span))
                        if parent is not None: paired.add(parent)
                sample((q, f) for q in paired for f in range(repetitions))
            else:
                sample((q, f) for q in range(len(options)) for f in range(repetitions))
            # Confirm promising full plans and local deletion edges using
            # additional common noise, retaining every negative observation.
            sampled = [q for q in range(1, len(options)) if all((q, f) in outcomes for f in range(repetitions))]
            ranked = sorted(sampled, key=lambda q: (
                -np.mean([outcomes[q, f]['success']-outcomes[0, f]['success'] for f in range(repetitions)]),
                options[q][2], options[q][0]))
            promising = [q for q in ranked if any(outcomes[q, f]['success'] > outcomes[0, f]['success']
                                                 for f in range(repetitions))][:8]
            chosen = set(promising); lookup = {option: i for i, option in enumerate(options)}
            for q in promising:
                code, delay, span = options[q]
                for boat in range(n):
                    digit = code//10**boat % 10
                    if digit:
                        parent = code-digit*10**boat
                        j = 0 if parent == 0 else lookup.get((parent, delay, span))
                        if j is not None: chosen.add(j)
                # Prefix chains make the whole-team alternative identifiable
                # through genuine single-vessel comparisons at every link.
                while code:
                    boat = int(np.flatnonzero((code//(10**np.arange(n))) % 10)[-1])
                    code -= (code//10**boat % 10)*10**boat
                    parent = 0 if not code else lookup.get((code, delay, span))
                    if parent is not None: chosen.add(parent)
            if neighborhood_centers:
                # Complete a genuine one-vessel neighbourhood around strong
                # rivals, not only a deletion path to the factual plan. This
                # includes harmful alternatives and changes between two active
                # coefficients. Both learners receive every measured outcome.
                centers = (promising or ranked)[:neighborhood_centers]
                for q in centers:
                    code, delay, span = options[q]
                    for alternate in single_vessel_neighborhood(code, n):
                        option = (alternate, delay, span) if alternate else (0, 0, 0)
                        if option not in lookup:
                            lookup[option] = len(options); options.append(option)
                            actions.append(intervention_action(option))
                        chosen.add(lookup[option])
                    chosen.add(q)
            if focus_option is not None:
                # Keep the actual decision and selected one-boat comparisons
                # even when another early screen already succeeds.
                chosen.update(q for q, option in enumerate(options) if option[1:] == focus_option[1:] and option[0] in focus_codes)
            chosen.add(0)
            sample((q, f) for q in chosen for f in range(max(repetitions, confirmation)))
            roots.append(dict(context=context[0].numpy(), effect_controls=effect[0, order[0]].numpy(),
                options=options, outcomes=outcomes, futures=futures, scene=scene, prefix_future=prefix_future,
                root_steps=elapsed, lookback=lookback, order=order[0].numpy(),
                sampled_behavior_option=focus_option))
            known.add(key)
            data = dict(model=model, roots=roots, completed=requested <= known, simulation_steps=steps,
                predecessor_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(), training_data=str(output),
                opponent_sha256=opponent_sha256,
                recipe=dict(spans=spans, repetitions=repetitions, confirmation=confirmation, seed=seed,
                            candidate_selection='full outcomes and connected single-vessel local contrasts',
                            adaptive_screen=adaptive_screen, observed_future=True, interaction_order=interaction_order,
                            screen_limit=screen_limit, screen_strategy=screen_strategy,
                            neighborhood_centers=neighborhood_centers, record_continuations=record_continuations,
                            neighborhood_width=neighborhood_width,
                            inference_batch=inference_batch, inference_device=inference_device,
                            physical_response_horizon=physical_response_horizon,
                            context_policy=None if context_policy is None else str(context_policy),
                            context_policy_sha256=None if context_policy is None else hashlib.sha256(Path(context_policy).read_bytes()).hexdigest(),
                            branch_continuation='common frozen predecessor'))
            r.atomic_save(data, output)
            print(dict(phase='local_difference_root', roots=len(roots), scene=scene[0], step=elapsed,
                observations=len(outcomes), promising=len(promising), simulation_steps=steps), flush=True)
    return data


def paired_continuation_bank(data):
    """Factual terminal supervision after observed interventions have ended."""
    records = []
    for root in data['roots']:
        for (q, f), value in root['outcomes'].items():
            if 'continuation_context' not in value or value.get('reused_prefix') or value.get('reused_terminal_prefix'):
                continue
            record = dict(contexts=value['continuation_context'][None],
                controls=value['continuation_controls'][None], values=np.asarray([value['success']], np.float32),
                remaining=np.asarray([value['continuation_remaining']], np.float32), scene=root['scene'],
                future=root['futures'][f], option=root['options'][q], physical_steps=0)
            if 'continuation_order' in value:
                record['orders'] = np.asarray(value['continuation_order'])[None].copy()
            records.append(record)
    return dict(records=records, predecessor_sha256=data['predecessor_sha256'], simulation_steps=0,
        trajectories=len(records), state_labels=len(records), completed=data['completed'],
        supervision='terminal result at observed post-option predecessor states; no extra environment interaction')


def complete_local_pairs(data, references, output, *, candidates=12, workers=6):
    """Complete promising local comparison paths on the development futures.

    Selection uses the already observed conditional consequences. Each selected
    complete plan and its one-vessel deletion path receives the same future
    population, including futures where the factual controller succeeds. All
    measured outcomes are retained for both conditional and ordinary training.
    """
    import conditional_fusion_value as c
    import develop_source_fusion as d
    import run_remote_evidence as r
    policy = Path(references[0]['policy'])
    if hashlib.sha256(policy.read_bytes()).hexdigest() != data['predecessor_sha256']:
        raise ValueError('Complete local comparisons against the measured predecessor.')
    population = {}
    for reference in references:
        if reference['policy_sha256'] != data['predecessor_sha256']:
            raise ValueError('Development futures must use the common predecessor.')
        for (scene, future), outcome in zip(reference['tasks'], reference['rows']):
            population.setdefault(tuple(scene), {})[future] = outcome
    actor = d.joint_load(str(policy), policy.stat().st_mtime_ns)
    r.initialize_source_branches(str(d.SOURCE))
    coefficients = np.asarray([[-1., -1.]]+[[a, b] for a in (0., .5, 1.) for b in (0., .5, 1.)])
    output = Path(output); roots = []; steps = 0
    pending = [root for root in data['roots'] if root['root_steps'] == 0
               and any(not row['success'] for row in population.get(tuple(root['scene']), {}).values())]
    result = dict(model=data['model'], roots=roots, simulation_steps=steps,
        predecessor_sha256=data['predecessor_sha256'], completed=not pending,
        recipe=dict(candidates=candidates, shared_futures=True, single_vessel_paths=True))
    if not pending:
        torch.save(result, output)
        return result
    with (ProcessPoolExecutor(max_workers=workers, initializer=r.initialize_source_branches,
          initargs=(str(d.SOURCE),)) if workers > 1 else nullcontext(None)) as pool:
        for original in pending:
            root = copy.deepcopy(original); scene = tuple(root['scene']); n = scene[1]
            totals = np.zeros(len(root['options'])); counts = totals.copy()
            for (q, future), row in root['outcomes'].items():
                totals[q] += row['success']; counts[q] += 1
            prior = (totals+.5)/(counts+1.)
            values, errors, _, _ = reconcile_root(root, prior)
            gains = values-values[0]
            # Cover both the best estimated gain and the best conservative gain.
            selected = {0}
            for confidence in (0., .5, 1.):
                scores = gains-confidence*errors
                ranking = sorted(range(1, len(scores)), key=lambda q: (-scores[q], root['options'][q]))
                selected.update(q for q in ranking[:candidates] if scores[q] > 0.)
            lookup = {tuple(option): q for q, option in enumerate(root['options'])}
            powers = 10**np.arange(n)
            for q in list(selected):
                code, delay, span = root['options'][q]
                for boat in np.flatnonzero((code//powers) % 10):
                    parent = code-(code//10**boat % 10)*10**boat
                    option = (parent, delay, span) if parent else (0, 0, 0)
                    if option not in lookup:
                        lookup[option] = len(root['options']); root['options'].append(option)
                    selected.add(lookup[option])
                while code:
                    boat = int(np.flatnonzero((code//powers) % 10)[-1])
                    code -= (code//10**boat % 10)*10**boat
                    option = (code, delay, span) if code else (0, 0, 0)
                    if option not in lookup:
                        lookup[option] = len(root['options']); root['options'].append(option)
                    selected.add(lookup[option])
            for future in population[scene]:
                if future not in root['futures']: root['futures'].append(future)
            future_ids = [root['futures'].index(future) for future in population[scene]]
            missing = [(q, f) for q in sorted(selected) for f in future_ids if (q, f) not in root['outcomes']]
            if missing:
                np.random.seed(scene[0]); torch.manual_seed(scene[0])
                env = d.numerical_source().TADEnv(n); obs, _ = env.reset(agility=scene[2], noisy_agility=False)
                packet = d.mechanism_tensor(d.joint_packet(env, obs))
                with torch.no_grad():
                    control = actor(packet, deterministic=True, gate_only=True)[0][0].numpy()[:, :3]
                    context, order = c.intervention_task_context(packet, True, True)
                if not np.allclose(context[0].numpy(), root['context'], atol=1e-6):
                    raise ValueError('Development comparison changed its physical initial state.')
                inverse = order[0].argsort().numpy(); one = np.ones((n, 1), np.float32)
                actions = {0: control}; random_state = np.random.get_state()
                for q in selected-{0}:
                    code, delay, span = root['options'][q]
                    preference = coefficients[(code//powers) % 10][inverse].astype(np.float32)
                    first = preference if delay == 0 else -np.ones_like(preference)
                    actions[q] = np.concatenate((control, -one, (delay+span)*one, -2*one,
                        np.zeros((n, 2), np.float32), delay*one, first, preference, np.zeros_like(one)), -1)
                jobs = [(str(policy), dict(environment=env, observation=obs, random_state=random_state,
                         action=actions[q]), scene, root['futures'][f], 1) for q, f in missing]
                iterator = pool.map(r.source_successor_branch, jobs, chunksize=1) if pool else map(r.source_successor_branch, jobs)
                for index, row in zip(missing, iterator):
                    root['outcomes'][index] = dict(success=float(row['success']), outcome=row['outcome'], duration=row['duration'])
                    steps += row['physical_steps']
            # This artifact records only incremental measured branches. The
            # source snapshot is merged once, preserving its existing labels.
            root['outcomes'] = {key: root['outcomes'][key] for key in missing}
            root.pop('future_weights', None)
            if root['outcomes']: roots.append(root)
            result = dict(model=data['model'], roots=roots, simulation_steps=steps,
                predecessor_sha256=data['predecessor_sha256'], completed=original is pending[-1],
                recipe=dict(candidates=candidates, shared_futures=True, single_vessel_paths=True))
            torch.save(result, output)
            print(dict(phase='complete_development_local_pairs', scene=scene[0], candidates=len(selected),
                measured=len(missing), roots=len(roots), simulation_steps=steps), flush=True)
    return result


def factual_trajectory_states(task):
    """Observe the frozen factual continuation at successive public states."""
    import conditional_fusion_value as c
    import develop_source_fusion as d
    policy, scene, future, expected, period = task
    policy = Path(policy)
    actor = d.joint_load(str(policy), policy.stat().st_mtime_ns, (False, False))
    if actor.successor_task_value is not None:
        raise ValueError('Factual references require the frozen predecessor.')
    np.random.seed(scene[0]); torch.manual_seed(scene[0])
    env = d.numerical_source().TADEnv(scene[1]); obs, _ = env.reset(agility=scene[2], noisy_agility=False)
    if future is not None: np.random.seed(future)
    contexts = []; controls = []; orders = []; ticks = []; elapsed = done = 0
    while not done:
        packet = d.mechanism_tensor(d.joint_packet(env, obs))
        with torch.no_grad():
            action = actor(packet, deterministic=True)[0][0].numpy()
            if elapsed % period == 0:
                context, order = c.intervention_task_context(packet, True, True)
                effect = c.paired_effect_controls(actor, packet, torch.as_tensor(action[None]))
                contexts.append(context[0].numpy()); controls.append(effect[0, order[0]].numpy()); ticks.append(elapsed)
                orders.append(order[0].numpy())
        obs, _, done, _, count = d.joint_advance(env, obs, actor, action)
        elapsed += count
    outcome = d.outcome(env, done)
    if expected is not None and outcome['success'] != expected:
        raise ValueError('A factual trajectory changed its measured terminal result.')
    return dict(contexts=np.asarray(contexts), controls=np.asarray(controls), orders=np.asarray(orders),
        values=np.full(len(contexts), outcome['success'], np.float32),
        remaining=np.asarray([elapsed-tick for tick in ticks], np.float32),
        scene=scene, future=future, physical_steps=elapsed)


def factual_trajectory_batch(tasks, *, device='cpu'):
    """Measure factual trajectories with batched copies of the same controller."""
    import conditional_fusion_value as c
    import develop_source_fusion as d
    import run_remote_evidence as r
    policy = Path(tasks[0][0])
    actor = d.joint_load(str(policy), policy.stat().st_mtime_ns)
    if actor.successor_task_value is not None:
        raise ValueError('Factual sampling requires the frozen predecessor.')
    jobs = []
    for _, scene, future, _, _ in tasks:
        np.random.seed(scene[0]); torch.manual_seed(scene[0])
        env = d.numerical_source().TADEnv(scene[1])
        obs, _ = env.reset(agility=scene[2], noisy_agility=False)
        if future is not None: np.random.seed(future)
        with torch.no_grad():
            action = actor(d.mechanism_tensor(d.joint_packet(env, obs)), deterministic=True)[0][0].numpy()
        jobs.append(dict(environment=env, observation=obs, action=action,
                         random_state=np.random.get_state()))
    r.SOURCE = d.SOURCE
    measured = r.batched_credit_rows(policy, [(t[1], t[2]) for t in tasks],
        batch_size=len(tasks), device=device, keep_steps=True, initial_jobs=jobs)
    records = []
    actor = d.joint_load(str(policy), policy.stat().st_mtime_ns).to(device)
    try:
        for task, trajectory in zip(tasks, measured):
            _, scene, future, expected, period = task
            row = trajectory['row']; selected = trajectory['steps'][::period]
            if expected is not None and row['success'] != expected:
                # Resolve a discrete decision near a floating-point tie with
                # the canonical single-episode evaluator used by the reference.
                actor.cpu()
                canonical = factual_trajectory_states(task)
                canonical['sampling_steps'] = row['physical_steps']+canonical['physical_steps']
                records.append(canonical)
                actor.to(device)
                continue
            contexts = []; controls = []; orders = []
            with torch.no_grad():
                for start in range(0, len(selected), 64):
                    group = selected[start:start+64]
                    packet = {key: torch.as_tensor(np.stack([item['packet'][key] for item in group]),
                              device=device) for key in group[0]['packet']}
                    native = actor(packet, deterministic=True, gate_only=True)[0]
                    context, order = c.intervention_task_context(packet, True, True)
                    effect = c.paired_effect_controls(actor, packet, native)
                    contexts.append(context.cpu().numpy())
                    orders.append(order.cpu().numpy())
                    controls.append(effect.gather(1, order[..., None].expand(-1, -1, 6)).cpu().numpy())
            elapsed = row['physical_steps']
            records.append(dict(contexts=np.concatenate(contexts), controls=np.concatenate(controls), orders=np.concatenate(orders),
                values=np.full(len(selected), row['success'], np.float32),
                remaining=np.asarray([elapsed-tick for tick in range(0, elapsed, period)], np.float32),
                scene=scene, future=future, physical_steps=elapsed))
    finally:
        actor.cpu()
    return records


def collect_factual_trajectories(policy, references, output, *, workers=10, period=5,
                                inference_batch=0, inference_device='cpu'):
    """Retain terminal supervision throughout the same factual trajectories.

    These are additional state labels from the same closed-loop episodes,
    not additional independent episodes. The ordinary learner receives the
    identical observations and interaction budget.
    """
    import develop_source_fusion as d
    import run_remote_evidence as r
    from functools import partial
    policy = Path(policy); digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    tasks = []
    for reference in references:
        if reference['policy_sha256'] != digest:
            raise ValueError('Factual trajectory labels require their exact predecessor.')
        tasks.extend((str(policy), scene, future, row['success'], period)
            for (scene, future), row in zip(reference['tasks'], reference['rows']))
    output = Path(output)
    saved = torch.load(output, map_location='cpu', weights_only=False) if output.exists() else None
    if saved is not None and (saved['predecessor_sha256'] != digest or saved['period'] != period):
        raise ValueError('The saved factual trajectories use a different predecessor or recording period.')
    records = [] if saved is None else saved['records']
    steps = 0 if saved is None else saved['simulation_steps']
    known = {(tuple(record['scene']), record['future']) for record in records}
    expected = {(tuple(scene), future) for _, scene, future, _, _ in tasks}
    if not known <= expected:
        raise ValueError('The saved factual trajectories use a different task set.')
    if known == expected:
        return saved
    pending = [task for task in tasks if (tuple(task[1]), task[2]) not in known]
    with (ProcessPoolExecutor(max_workers=workers, initializer=r.initialize_source_branches,
          initargs=(str(d.SOURCE),)) if workers > 1 else nullcontext(None)) as pool:
        if inference_batch:
            groups = [pending[start:start+inference_batch] for start in range(0, len(pending), inference_batch)]
            measure = partial(factual_trajectory_batch, device=inference_device)
            measured = pool.map(measure, groups, chunksize=1) if pool else map(measure, groups)
            iterator = (record for batch in measured for record in batch)
        else:
            iterator = (pool.map(factual_trajectory_states, pending, chunksize=1)
                        if pool else map(factual_trajectory_states, pending))
        for record in iterator:
            records.append(record); steps += record.get('sampling_steps', record['physical_steps'])
            if len(records) % 32 == 0 or len(records) == len(tasks):
                bank = dict(records=records, predecessor_sha256=digest, simulation_steps=steps,
                    trajectories=len(records), state_labels=sum(len(row['values']) for row in records),
                    completed=len(records) == len(tasks), period=period,
                    inference_batch=inference_batch, inference_device=inference_device)
                torch.save(bank, output)
                print(dict(phase='factual_continuation_states', trajectories=len(records),
                    state_labels=bank['state_labels'], simulation_steps=steps), flush=True)
    return bank


def factual_bank_from_model(model, data):
    """Recover the exact fitted factual observations retained in a checkpoint."""
    value = model['successor_task_value']
    fit = model['successor_task_fit']
    record = {}
    for key, field in (('contexts', 'continuation_contexts'), ('controls', 'continuation_controls'),
                       ('values', 'continuation_values'), ('remaining', 'continuation_remaining')):
        if value.get(field) is None:
            raise ValueError('The reference checkpoint does not retain its factual observations.')
        record[key] = torch.as_tensor(value[field]).cpu().numpy().copy()
    return dict(records=[record], predecessor_sha256=data['predecessor_sha256'],
        simulation_steps=fit['simulation_steps']-data['simulation_steps'],
        trajectories=fit['factual_trajectory_replays'], state_labels=len(record['values']),
        paired_continuation_state_labels=fit.get('paired_continuation_state_labels', 0),
        training_data=fit.get('factual_bank_training_data'), completed=True)


def source_successor_batches(tasks, *, batch_size=128, device='cpu'):
    """Batch finite interventions on a frozen predecessor without value stages."""
    import develop_source_fusion as d
    import run_remote_evidence as r
    groups = {}
    for index, (policy, job, scene, future, tick) in enumerate(tasks):
        action = job['action']
        if action.shape[-1] == 14:
            if np.any(action[:, 13] != 0):
                raise ValueError('Batched predecessor sampling cannot traverse a value-stage chain.')
            # With no value module, the depth-zero delayed prefix is the same
            # per-tick native predecessor as the free phase of this program.
            job = dict(job, action=action[:, :13].copy())
        groups.setdefault((policy, job['action'].shape[-1]), []).append((index, job, scene, future, tick))
    result = [None]*len(tasks)
    r.SOURCE = d.SOURCE
    for (policy, _), group in groups.items():
        actor = d.joint_load(str(policy), Path(policy).stat().st_mtime_ns)
        if actor.successor_task_value is not None:
            raise ValueError('Batched local comparisons require a predecessor without M2.')
        rows = r.batched_credit_rows(policy, [(scene, future) for _, _, scene, future, _ in group],
            batch_size=batch_size, device=device, initial_jobs=[job for _, job, _, _, _ in group],
            response_tick=[tick for _, _, _, _, tick in group])
        for (index, _, _, _, _), row in zip(group, rows):
            result[index] = row
    return result


def breadth_local_design(blueprint, candidate_options, *, pairs_per_root=6, repetitions=2, seed=0):
    """Allocate fresh paired queries to every root without inspecting labels.

    Each queried edge changes one vessel at a common intervention window.
    Stratifying edges by plan size and window spreads the bounded root budget
    over the deployment catalog, including roots previously sampled only
    factually. The returned outcomes contain scheduling bounds, never labels.
    """
    if pairs_per_root < 1 or repetitions < 1:
        raise ValueError('Breadth sampling requires positive pair and future counts.')
    options = [(0, 0, 0)]+sorted(set(map(tuple, candidate_options))-{(0, 0, 0)})
    if any(code < 0 or delay < 0 or span < 0 or (code and span == 0)
           for code, delay, span in options):
        raise ValueError('Use valid finite intervention programs.')
    rng = np.random.default_rng(seed)
    by_fleet = {}
    roots = []
    signatures = []
    for root_id, original in enumerate(blueprint['roots']):
        scene = tuple(original['scene']); n = int(scene[1])
        if n not in by_fleet:
            if any(code >= 10**n for code, _, _ in options):
                raise ValueError('The candidate catalog must match every root fleet.')
            groups = {}
            for a, b in local_edges(options, n):
                code, delay, span = options[b]
                held = max(sum(code//10**boat % 10 != 0 for boat in range(n))
                           for code in (options[a][0], options[b][0]))
                groups.setdefault((held, delay, span), []).append((a, b))
            if not groups:
                raise ValueError('The catalog has no single-vessel comparison edges.')
            by_fleet[n] = [groups[key] for key in sorted(groups)]
        groups = by_fleet[n]
        group_order = rng.permutation(len(groups))
        selected = {0}
        for index in range(pairs_per_root):
            group = groups[group_order[index % len(groups)]]
            selected.update(group[int(rng.integers(len(group)))])
        chosen = [options[index] for index in sorted(selected)]
        futures = []
        while len(futures) < repetitions:
            future = int(rng.integers(1, 2**32-1))
            if future not in futures:
                futures.append(future)
        root = dict(scene=scene, prefix_future=original['prefix_future'],
            root_steps=int(original['root_steps']), lookback=original.get('lookback'),
            options=chosen, futures=futures, future_weights=np.ones(repetitions, np.float32),
            outcomes={(q, f): dict(duration=400) for q in range(len(chosen))
                      for f in range(repetitions)}, blueprint_root=root_id)
        roots.append(root)
        signatures.append((scene, root['prefix_future'], root['root_steps'], root['lookback'],
                           tuple(chosen), tuple(futures)))
    design = dict(kind='breadth_local_pairs', seed=int(seed), pairs_per_root=int(pairs_per_root),
        repetitions=int(repetitions), roots=len(roots), maximum_observations_per_root=
        (1+2*pairs_per_root)*repetitions, planned_observations=sum(len(x['outcomes']) for x in roots),
        catalog_sha256=hashlib.sha256(repr(options).encode()).hexdigest(),
        design_sha256=hashlib.sha256(repr(signatures).encode()).hexdigest(),
        inherited_labels=False, allocation='uniform plan-size/window strata; fresh paired futures')
    return dict(roots=roots, acquisition_design=design)


def _recollect_breadth_root(task):
    import tempfile
    policy, root_id, root, inference_batch, inference_device = task
    # Keep only the combined acquisition on disk. The single-root collector
    # uses its existing bounded replay path and removes its temporary state.
    with tempfile.TemporaryDirectory(prefix='arboids-local-root-') as temporary:
        value = recollect_local_design(policy, dict(roots=[root]),
            Path(temporary)/'root.pth', simulation_budget=800+400*len(root['outcomes']),
            workers=1, inference_batch=inference_batch, inference_device=inference_device)
    if not value['all_design_roots_measured'] or len(value['roots']) != 1:
        raise ValueError('A breadth root did not finish all its paired queries.')
    measured = value['roots'][0]
    measured['design_root'] = root_id
    return measured, value['simulation_steps']


def recollect_breadth_design(policy, blueprint, output, *, workers=4,
                            inference_batch=32, inference_device='cpu'):
    """Parallelize bounded roots while keeping every branch batch unchanged."""
    import develop_source_fusion as d
    import run_remote_evidence as r
    policy = Path(policy); output = Path(output)
    design = blueprint.get('acquisition_design')
    if design is None or design.get('kind') != 'breadth_local_pairs':
        raise ValueError('Root-parallel acquisition requires a bounded breadth design.')
    if workers < 1 or inference_batch < 0:
        raise ValueError('Use a positive worker count and nonnegative branch batch size.')
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    result = (torch.load(output, map_location='cpu', weights_only=False) if output.exists() else
        dict(model=torch.load(policy, map_location='cpu', weights_only=False), roots=[],
            predecessor_sha256=digest, acquisition_design=copy.deepcopy(design),
            simulation_steps=0, next_design_root=0, completed=False, inherited_labels=False,
            training_data=str(output), inference_batch=inference_batch,
            inference_device=inference_device, root_dispatch_kind='independent_bounded_roots'))
    if (result['predecessor_sha256'] != digest or result['acquisition_design'] != design
            or result['inference_batch'] != inference_batch or result['inference_device'] != inference_device):
        raise ValueError('The saved breadth data uses a different predecessor, design or branch batch.')
    if result['model'].get('successor_task_value') is not None:
        raise ValueError('Breadth sampling requires its frozen controller without M2.')
    if result['completed']:
        return result
    if (not 0 <= result['next_design_root'] <= len(blueprint['roots']) or
            [root['design_root'] for root in result['roots']] != list(range(result['next_design_root']))):
        raise ValueError('The saved breadth roots must retain their original design order.')
    result.update(workers=workers, design_root_count=len(blueprint['roots']),
        planned_step_ceiling=sum(800+400*len(root['outcomes']) for root in blueprint['roots']))
    pending = [(str(policy), index, blueprint['roots'][index], inference_batch, inference_device)
               for index in range(result['next_design_root'], len(blueprint['roots']))]
    with (ProcessPoolExecutor(max_workers=workers, initializer=r.initialize_source_branches,
          initargs=(str(d.SOURCE),)) if workers > 1 else nullcontext(None)) as pool:
        measured = pool.map(_recollect_breadth_root, pending, chunksize=1) if pool else map(_recollect_breadth_root, pending)
        for root, steps in measured:
            result['roots'].append(root)
            result['simulation_steps'] += steps
            result['next_design_root'] += 1
            if len(result['roots']) % 4 == 0 or len(result['roots']) == len(blueprint['roots']):
                output.parent.mkdir(parents=True, exist_ok=True)
                r.atomic_save(result, output)
                print(dict(phase='breadth_root_collection', roots=len(result['roots']),
                    total=len(blueprint['roots']), physical_steps=result['simulation_steps']), flush=True)
    result.update(completed=True, all_design_roots_measured=True,
        measured_terminal_observations=sum(len(root['outcomes']) for root in result['roots']))
    r.atomic_save(result, output)
    return result


def recollect_local_design(policy, blueprint, output, *, simulation_budget, workers=4,
                           inference_batch=0, inference_device='cpu'):
    """Re-measure a local comparison design under a replacement predecessor.

    Layouts, futures and complete intervention programs come from the common
    design. States, source controls and every terminal label are regenerated.
    Prefixes follow the replacement itself; no reference value or teacher
    action is imported. Behaviour-triggered reference roots retain their
    recorded physical clock under the replacement. The physical-step cap
    includes prefix simulation.
    """
    import conditional_fusion_value as c
    import develop_source_fusion as d
    import run_remote_evidence as r
    from functools import partial
    policy = Path(policy); output = Path(output)
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    model = torch.load(policy, map_location='cpu', weights_only=False)
    if model.get('successor_task_value') is not None:
        raise ValueError('Replacement sampling requires its frozen controller without M2.')
    r.initialize_source_branches(str(d.SOURCE))
    actor = d.joint_load(str(policy), policy.stat().st_mtime_ns)
    result = (torch.load(output, map_location='cpu', weights_only=False) if output.exists() else
              dict(model=model, roots=[], predecessor_sha256=digest, simulation_steps=0,
                   next_design_root=0, completed=False, training_data=str(output),
                   simulation_budget=int(simulation_budget), inherited_labels=False))
    if result['predecessor_sha256'] != digest or result['simulation_budget'] != simulation_budget:
        raise ValueError('The saved replacement data belongs to a different predecessor or budget.')
    design = blueprint.get('acquisition_design')
    if result.get('acquisition_design') != design and (result['roots'] or result.get('partial_root')):
        raise ValueError('The saved replacement data uses a different acquisition design.')
    if design is not None:
        result['acquisition_design'] = copy.deepcopy(design)
    if result['completed']:
        return result
    result.update(inference_batch=inference_batch, inference_device=inference_device,
                  dispatch_batches_per_worker=4)
    coefficients = np.asarray([[-1., -1.]]+[[a, b] for a in (0., .5, 1.) for b in (0., .5, 1.)])
    with (ProcessPoolExecutor(max_workers=workers, initializer=r.initialize_source_branches,
          initargs=(str(d.SOURCE),)) if workers > 1 else nullcontext(None)) as pool:
        for design_id in range(result['next_design_root'], len(blueprint['roots'])):
            if simulation_budget-result['simulation_steps'] < 800:
                break
            original = observed_programs(blueprint['roots'][design_id])
            scene = tuple(original['scene']); future = original['prefix_future']
            np.random.seed(scene[0]); torch.manual_seed(scene[0])
            env = d.numerical_source().TADEnv(scene[1])
            obs, _ = env.reset(agility=scene[2], noisy_agility=False)
            if future is not None: np.random.seed(future)
            elapsed = done = 0
            history = []
            if original['root_steps']:
                lookback = original.get('lookback')
                if isinstance(lookback, str) and lookback not in ('first_intervention', 'last_intervention', 'uniform_eligible'):
                    raise ValueError(f'Unknown reference root timing: {lookback}')
                clock_target = (original['root_steps'] if lookback is None or
                                isinstance(lookback, str) else None)
                while not done:
                    history.append((elapsed, copy.deepcopy(env), obs.copy(), np.random.get_state()))
                    with torch.no_grad():
                        action = actor(d.mechanism_tensor(d.joint_packet(env, obs)), deterministic=True)[0][0].numpy()
                    obs, _, done, _, count = d.joint_advance(env, obs, actor, action)
                    elapsed += count; result['simulation_steps'] += count
                    if clock_target is not None and elapsed >= clock_target:
                        break
                if not done:
                    history.append((elapsed, copy.deepcopy(env), obs.copy(), np.random.get_state()))
                cutoff = (max(0, elapsed-int(lookback)) if clock_target is None else clock_target)
                elapsed, env, obs, random_state = max(
                    (item for item in history if item[0] <= cutoff), key=lambda item: item[0])
                np.random.set_state(random_state)
            packet = d.mechanism_tensor(d.joint_packet(env, obs))
            with torch.no_grad():
                control = actor(packet, deterministic=True, gate_only=True)[0][0].numpy()[:, :3]
                context, order = c.intervention_task_context(packet, True, True)
                effect = c.paired_effect_controls(actor, packet, torch.as_tensor(control[None]))
            inverse = order[0].argsort().numpy(); one = np.ones((scene[1], 1), np.float32)
            options = list(map(tuple, original['options']))
            actions = []
            for code, delay, span in options:
                if not code:
                    actions.append(control)
                    continue
                preference = coefficients[(code//(10**np.arange(scene[1]))) % 10][inverse].astype(np.float32)
                first = preference if delay == 0 else np.full_like(preference, -1.)
                actions.append(np.concatenate((control, -one, (delay+span)*one, -2*one,
                    np.zeros((scene[1], 2), np.float32), delay*one, first, preference,
                    np.zeros_like(one)), -1))
            random_state = np.random.get_state()
            pending_root = result.get('partial_root')
            if pending_root is not None and (pending_root['design_root'] != design_id or
                    not np.array_equal(pending_root['context'], context[0].numpy())):
                raise ValueError('The saved comparison batch has a different predecessor state.')
            outcomes = {} if pending_root is None else dict(pending_root['outcomes'])
            keys = sorted(original['outcomes'])
            # Complete factual observations before candidate comparisons.
            # Preserve negative as well as positive labels at every root.
            position = len(outcomes)
            if set(outcomes) != set(keys[:position]):
                raise ValueError('The saved comparison batch changed its observation design.')
            remaining_horizon = max(1, 400-elapsed)
            while position < len(keys):
                allowance = (simulation_budget-result['simulation_steps'])//remaining_horizon
                # Queue several unchanged inference batches per worker so a
                # long continuation does not leave the shorter workers idle.
                size = min(inference_batch*max(4, workers*4) if inference_batch else max(1, workers*2),
                           len(keys)-position, allowance)
                if not size:
                    break
                selected = keys[position:position+size]
                jobs = [(str(policy), dict(environment=env, observation=obs,
                    random_state=random_state, action=actions[q]), scene, original['futures'][f],
                    max(1, options[q][1]+options[q][2])) for q, f in selected]
                if inference_batch:
                    # Similar reference horizons keep inference batches occupied;
                    # all selected comparisons are still measured and restored
                    # to their original order before entering the learning data.
                    # Keep the established four-batch sorting blocks intact
                    # while dispatching enough blocks to occupy every worker.
                    # This preserves each inference batch when concurrency changes.
                    schedule = [i for start in range(0, len(jobs), inference_batch*4)
                        for i in sorted(range(start, min(start+inference_batch*4, len(jobs))),
                            key=lambda i: original['outcomes'][selected[i]]['duration'])]
                    ordered_jobs = [jobs[i] for i in schedule]
                    groups = [ordered_jobs[start:start+inference_batch] for start in range(0, len(jobs), inference_batch)]
                    measure = partial(source_successor_batches, batch_size=inference_batch, device=inference_device)
                    batches = pool.map(measure, groups, chunksize=1) if pool else map(measure, groups)
                    ordered_rows = [row for batch in batches for row in batch]
                    measured = [None]*len(jobs)
                    for index, row in zip(schedule, ordered_rows):
                        measured[index] = row
                else:
                    measured = pool.map(r.source_successor_branch, jobs, chunksize=1) if pool else map(r.source_successor_branch, jobs)
                for key, row in zip(selected, measured):
                    item = dict(success=float(row['success']), outcome=row['outcome'], duration=row['duration'])
                    response = row.get('response')
                    if response is not None and not response['terminal']:
                        successor = d.mechanism_tensor(response['packet'])
                        with torch.no_grad():
                            native = actor(successor, deterministic=True, gate_only=True)[0]
                            later_context, later_order = c.intervention_task_context(successor, True, True)
                            later_control = c.paired_effect_controls(actor, successor, native)
                        item.update(continuation_context=later_context[0].numpy(),
                            continuation_controls=later_control[0, later_order[0]].numpy(),
                            continuation_order=later_order[0].numpy(),
                            continuation_remaining=row['physical_steps']-response['steps'])
                    outcomes[key] = item
                    result['simulation_steps'] += row['physical_steps']
                position += size
                result['partial_root'] = dict(design_root=design_id, context=context[0].numpy(),
                                               outcomes=outcomes)
                output.parent.mkdir(parents=True, exist_ok=True)
                r.atomic_save(result, output)
                if position % 128 < size or position == len(keys):
                    print(dict(phase='replacement_local_branches', root=design_id+1,
                        observations=position, planned=len(keys), steps=result['simulation_steps'],
                        budget=simulation_budget), flush=True)
            if outcomes:
                result['roots'].append(dict(context=context[0].numpy(),
                    effect_controls=effect[0, order[0]].numpy(), options=options, outcomes=outcomes,
                    futures=list(original['futures']), scene=scene, prefix_future=future,
                    root_steps=elapsed, lookback=original.get('lookback'), order=order[0].numpy(),
                    future_weights=np.asarray(original.get('future_weights',
                        np.ones(len(original['futures']))), np.float32), design_root=design_id))
            result['next_design_root'] = design_id+1
            result.pop('partial_root', None)
            output.parent.mkdir(parents=True, exist_ok=True)
            r.atomic_save(result, output)
            print(dict(phase='replacement_local_comparisons', root=design_id+1,
                roots=len(blueprint['roots']), observations=len(outcomes),
                steps=result['simulation_steps'], budget=simulation_budget), flush=True)
            if position < len(keys):
                break
    result.update(completed=True, measured_terminal_observations=sum(
        len(root['outcomes']) for root in result['roots']), design_root_count=len(blueprint['roots']),
        all_design_roots_measured=result['next_design_root'] == len(blueprint['roots']) and
            all(len(root['outcomes']) == len(observed_programs(blueprint['roots'][root['design_root']])['outcomes'])
                for root in result['roots']))
    r.atomic_save(result, output)
    return result
