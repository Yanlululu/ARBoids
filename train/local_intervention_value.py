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


@torch.no_grad()
def public_physical_response_prior(context, paths):
    """Express public four-second dynamics in the successor's observed frame.

    The target keeps its currently observed velocity and every boat keeps
    its finite-window public command. Learned residuals account for target
    response, subsequent fleet replanning and early termination.
    """
    b, n = paths.shape[:2]
    before = context[:, :8*n].reshape(b, n, 8)
    # In the target-centred inward frame, ||p_i||^2-||p_i-p_a||^2
    # equals R^2-2 R x_i. Multiple measured boats recover the target's
    # range without a hidden simulator state or a future observation.
    x = before[..., 0]
    difference = before[..., 7].square()-before[..., :2].square().sum(-1)
    centered = x-x.mean(-1, keepdim=True)
    spread = centered.square().sum(-1)
    radius = -.5*(centered*(difference-difference.mean(-1, keepdim=True))).sum(-1)/spread.clamp_min(1e-8)
    fallback = (x+(before[..., 7].square()-before[..., 1].square()).clamp_min(0).sqrt()).sort(
        dim=-1, stable=True).values[:, (n-1)//2]
    radius = 30.*torch.where(spread > 1e-8, radius, fallback).clamp_min(1e-4)
    target_velocity = 5.*context[:, -3:-1]
    target_delta = 4.*target_velocity
    target_end = target_delta-torch.stack((radius, torch.zeros_like(radius)), -1)
    inward = -target_end/target_end.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def rotate(vector):
        direction = inward.reshape(b, *([1]*(vector.ndim-2)), 2)
        return torch.stack(((vector*direction).sum(-1),
            direction[..., 0]*vector[..., 1]-direction[..., 1]*vector[..., 0]), -1)

    states = paths[..., -1, :6]
    position = rotate(states[..., :2]-target_delta[:, None, None])/30.
    velocity = rotate(states[..., 3:5])/3.
    heading = rotate(torch.stack((states[..., 2].cos(), states[..., 2].sin()), -1))
    origin = states[..., :2]-torch.stack((radius, torch.zeros_like(radius)), -1)[:, None, None]
    following = torch.cat((position, velocity, heading, states[..., 5:6],
                            origin.norm(dim=-1, keepdim=True)/30.), -1)
    # Convert the new target bearing back to the original world axes.
    old_inward = -context[:, -5:-3]
    world_end = torch.stack((old_inward[:, 0]*target_end[:, 0]-old_inward[:, 1]*target_end[:, 1],
                            old_inward[:, 1]*target_end[:, 0]+old_inward[:, 0]*target_end[:, 1]), -1)
    bearing = world_end/world_end.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    global_after = torch.cat((bearing, rotate(target_velocity)/5., context[:, -1:]+.05), -1)
    return following-before[:, :, None], torch.cat((global_after-context[:, -5:],
                                                    context.new_ones((b, 1))), -1)


class JointTaskPotential(nn.Module):
    def __init__(self, width=64, peer_conditioned=True, output_width=1, context_conditioned=False,
                 local_contrast_head=False, motion_basis=False, residual_potential=False,
                 finite_motion_basis=False, dual_motion_basis=False, identity_conditioned=False,
                 paired_response_encoding=False, aligned_response_contrast=False, paired_event_difference=False,
                 response_conditioned_value=False, physics_residual_response=False, shared_factual_response=False,
                 shared_successor_value=False):
        super().__init__()
        self.width = width
        self.peer_conditioned = peer_conditioned
        self.context_conditioned = context_conditioned
        self.motion_basis = motion_basis
        self.finite_motion_basis = finite_motion_basis
        self.dual_motion_basis = dual_motion_basis
        self.identity_conditioned = identity_conditioned
        self.paired_response_encoding = paired_response_encoding
        self.aligned_response_contrast = aligned_response_contrast
        self.paired_event_difference = paired_event_difference
        self.response_conditioned_value = response_conditioned_value
        self.physics_residual_response = physics_residual_response
        self.shared_factual_response = shared_factual_response
        self.shared_successor_value = shared_successor_value
        if shared_successor_value and (not response_conditioned_value or output_width != 1
                                      or paired_response_encoding):
            raise ValueError('Shared successor values require one response-conditioned scalar potential.')
        if shared_factual_response and not physics_residual_response:
            raise ValueError('Shared factual responses require a public physical response prior.')
        if physics_residual_response and not response_conditioned_value:
            raise ValueError('A physical response prior requires the response-conditioned value representation.')
        if response_conditioned_value and (not finite_motion_basis or residual_potential or local_contrast_head):
            raise ValueError('Response-conditioned value requires finite motion features and a direct value head.')
        if paired_event_difference and not paired_response_encoding:
            raise ValueError('Paired event differences require factual and intervention response encodings.')
        if aligned_response_contrast and not paired_response_encoding:
            raise ValueError('Aligned vessel response contrasts require paired factual and intervention encodings.')
        if paired_response_encoding and (output_width != 3 or residual_potential or local_contrast_head
                                         or not motion_basis):
            raise ValueError('Paired response encoding requires factual, rescue and harm outputs with public motion features.')
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
        self.head = nn.Sequential(nn.Linear(3*width+8, width), nn.SiLU(),
                                  nn.Linear(width, 1 if paired_response_encoding else output_width))
        self.paired_head = (nn.Sequential(nn.Linear(2*(3*width+8), width), nn.SiLU(),
            nn.Linear(width, 2)) if paired_response_encoding else None)
        self.effect_head = (nn.Sequential(nn.Linear(2*(3*width+8), width), nn.SiLU(),
            nn.Linear(width, 1)) if residual_potential else None)
        self.contrast = (nn.Sequential(nn.Linear(3*width+8, width), nn.SiLU(),
            nn.Linear(width, 6*width)) if local_contrast_head else None)
        self.contrast_categorical = local_contrast_head == 'categorical'
        self.contrast_agreement = (nn.Sequential(nn.Linear(3*width+8, width), nn.SiLU(),
            nn.Linear(width, 2)) if self.contrast_categorical else None)
        if response_conditioned_value:
            self.response_node = nn.Linear(width, 8)
            self.response_global = nn.Linear(3*width+8, 6)
            self.future_node = nn.Sequential(nn.Linear(22, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
            self.future_message = nn.Sequential(nn.Linear(2*width+10, width), nn.SiLU(),
                                                nn.Linear(width, width), nn.SiLU())
            self.future_update = nn.Sequential(nn.Linear(3*width, width), nn.SiLU())
            if physics_residual_response:
                for layer in (self.response_node, self.response_global):
                    nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
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

    def physical_prior(self, context, controls, programs, timing, responses=None):
        if not self.physics_residual_response:
            return None
        if responses is None or len(responses) == 2:
            responses = response_basis(context, controls, timing, self.dual_motion_basis)
        nodes, global_response = public_physical_response_prior(context, responses[1])
        batch = torch.arange(len(context), device=context.device)[:, None]
        boats = torch.arange(controls.shape[1], device=context.device)[None]
        return nodes[batch, boats, programs], global_response

    def factual_response_residual(self, context, controls, responses=None):
        """Share learned factual motion across alternatives so common errors cancel."""
        n = controls.shape[1]
        programs = torch.zeros((len(context), n), dtype=torch.long, device=context.device)
        timing = torch.zeros((len(context), 2), dtype=torch.long, device=context.device)
        encoded, representation = self.encode(context, controls, programs, timing, responses)
        return self.response_node(encoded), self.response_global(representation)

    def response_prediction(self, encoded, representation, programs, timing, physical_prior=None,
                            factual_residual=None):
        """Predict measured state changes within the common twenty-tick horizon."""
        if self.shared_factual_response:
            if physical_prior is None or factual_residual is None:
                raise ValueError('Shared factual responses require public effects and a common factual residual.')
            return physical_prior[0]+factual_residual[0], physical_prior[1]+factual_residual[1]
        delay = timing[..., 0].clamp(0, 20)
        span = torch.minimum(timing[..., 1].clamp_min(0), 20-delay)
        clock = torch.stack((delay, span), -1).to(encoded.dtype).log1p()/6.
        clock = clock*(programs > 0).any(-1, keepdim=True)
        physical = torch.cat((representation[..., :-3], clock.expand(*representation.shape[:-1], 2),
                              representation[..., -1:]), -1)
        node_response, global_response = self.response_node(encoded), self.response_global(physical)
        if self.physics_residual_response:
            if physical_prior is None:
                raise ValueError('Residual response prediction requires the public physical prior.')
            node_response = node_response+physical_prior[0]
            global_response = global_response+physical_prior[1]
        return node_response, global_response

    def consequence_encode(self, context, encoded, representation, programs, timing, physical_prior=None,
                           factual_residual=None):
        """Make value depend on predicted physical successors, never true future inputs."""
        response, global_response = self.response_prediction(encoded, representation, programs, timing,
            physical_prior, factual_residual)
        return self.encode_response_value(context, representation, programs, response, global_response)

    def encode_response_value(self, context, representation, programs, response, global_response):
        """Share the consequence evaluator between predictions and training observations."""
        n = response.shape[-2]
        rows = context[:, :8*n].reshape(len(context), *([1]*(response.ndim-3)), n, 8)
        following = rows+response
        global_before = context[:, -5:].reshape(len(context), *([1]*(response.ndim-3)), 5)
        future_global = torch.cat((global_before+global_response[..., :5], global_response[..., 5:]), -1)
        own = self.future_node(torch.cat((following, response,
            future_global[..., None, :].expand(*following.shape[:-1], 6)), -1))
        difference = following[..., :, None, :]-following[..., None, :, :]
        distance = difference[..., :2].square().sum(-1).clamp_min(1e-8).sqrt()
        closing = (difference[..., :2]*difference[..., 2:4]).sum(-1)/distance
        geometry = torch.cat((difference, distance[..., None], closing[..., None]), -1)
        pair = self.future_message(torch.cat((own[..., :, None, :].expand(*geometry.shape[:-1], self.width),
            own[..., None, :, :].expand(*geometry.shape[:-1], self.width), geometry), -1))
        mask = (~torch.eye(n, dtype=torch.bool, device=response.device))[..., None]
        future = self.future_update(torch.cat((own, (pair*mask).sum(-2)/max(n-1, 1),
            pair.masked_fill(~mask, -1e4).amax(-2)), -1))
        if self.shared_successor_value:
            # Factual trajectories and interventions enter the same physical
            # value representation. Tagging the changed vessel would leave a
            # feature block identically zero on all factual training examples.
            third = future.amin(-2)
        else:
            active = (programs > 0)[..., None].to(response.dtype)
            third = (future*active).sum(-2)/active.sum(-2).clamp_min(1)
        result = torch.cat((future.mean(-2), future.amax(-2), third, representation[..., -8:]), -1)
        return future, result

    def observed_response_logits(self, context, controls, programs, timing, observed, factual=None):
        """Train the shared value head on measured consequences; never used by a decision.

        Paired effects require both physical observations from the same future.
        These teacher inputs are training labels, separate from forward's public
        current-state inputs and its predicted physical response.
        """
        n = controls.shape[1]
        if not self.response_conditioned_value or observed.shape != (len(context), 8*n+6):
            raise ValueError('Observed response values require aligned physical labels and a response-conditioned head.')
        clock = timing.to(context.dtype).log1p()/6.*(programs > 0).any(-1, keepdim=True)
        fleet_size = context.new_full((len(context), 1), n/6.)
        representation = torch.cat((context[:, -5:], clock, fleet_size), -1)
        encoded, representation = self.encode_response_value(context, representation, programs,
            observed[:, :8*n].reshape(-1, n, 8), observed[:, 8*n:])
        if not self.paired_response_encoding:
            return self.head(representation).squeeze(-1)
        if factual is None or factual.shape != observed.shape:
            raise ValueError('A measured conditional value requires its same-future factual response.')
        reference = torch.cat((context[:, -5:], torch.zeros_like(clock), fleet_size), -1)
        factual_nodes, reference = self.encode_response_value(context, reference, torch.zeros_like(programs),
            factual[:, :8*n].reshape(-1, n, 8), factual[:, 8*n:])
        features = (self.aligned_response_features(encoded, factual_nodes, reference, programs > 0,
            representation[..., -8:]-reference[..., -8:]) if self.aligned_response_contrast else
            torch.cat((reference, representation-reference), -1))
        effects = self.event_logits(self.paired_head(features), self.intervention_exposure(controls, programs, timing),
            self.paired_event_difference)
        return torch.cat((self.head(reference), effects), -1)

    def forward(self, context, controls, programs, timing, responses=None):
        if self.physics_residual_response and (responses is None or len(responses) == 2):
            responses = response_basis(context, controls, timing, self.dual_motion_basis)
        factual_residual = (self.factual_response_residual(context, controls, responses)
            if self.shared_factual_response else None)
        encoded, representation = self.encode(context, controls, programs, timing, responses)
        if self.response_conditioned_value:
            encoded, representation = self.consequence_encode(context, encoded, representation, programs, timing,
                self.physical_prior(context, controls, programs, timing, responses), factual_residual)
        if self.paired_response_encoding:
            factual_nodes, reference = self.encode(context, controls, torch.zeros_like(programs),
                torch.zeros_like(timing), responses)
            if self.response_conditioned_value:
                factual_nodes, reference = self.consequence_encode(context, factual_nodes, reference,
                    torch.zeros_like(programs), torch.zeros_like(timing), self.physical_prior(context, controls,
                        torch.zeros_like(programs), torch.zeros_like(timing), responses), factual_residual)
            features = (self.aligned_response_features(encoded, factual_nodes, reference, programs > 0,
                representation[..., -8:]-reference[..., -8:]) if self.aligned_response_contrast else
                torch.cat((reference, representation-reference), -1))
            raw = self.paired_head(features)
            exposure = self.intervention_exposure(controls, programs, timing)
            effects = self.event_logits(raw, exposure, self.paired_event_difference)
            return torch.cat((self.head(reference), effects), -1)
        if not self.residual_potential:
            return self.head(representation).squeeze(-1)
        reference = self.encode(context, controls, torch.zeros_like(programs), torch.zeros_like(timing), responses)[1]
        # A complete plan is evaluated relative to its own factual state.
        # The effect vanishes continuously with its physical duration, so a
        # one-tick choice does not inherit a full persistent-maneuver effect.
        effect = self.effect_head(torch.cat((reference, representation-reference), -1)).squeeze(-1)
        exposure = -torch.expm1(-.2*timing[:, 1].to(context.dtype))*(programs > 0).any(-1)
        return self.head(reference).squeeze(-1)+exposure*effect

    @staticmethod
    def aligned_response_features(encoded, factual_nodes, reference, active, global_difference):
        """Contrast each vessel with itself before pooling the fleet effect.

        The affected vessels contribute their own factual context, rather
        than an absolute candidate activation masquerading as a change.
        Peer responses retain their signs in the largest-response pool.
        """
        delta=encoded-factual_nodes
        held=active[..., None].to(encoded.dtype)
        count=held.sum(-2).clamp_min(1)
        factual_subject=(factual_nodes*held).sum(-2)/count
        factual=torch.cat((factual_nodes.mean(-2).expand_as(factual_subject),
            factual_nodes.amax(-2).expand_as(factual_subject),factual_subject,
            reference[..., -8:].expand(*factual_subject.shape[:-1],8)), -1)
        largest=delta.gather(-2,delta.abs().argmax(-2,keepdim=True)).squeeze(-2)
        contrast=torch.cat((delta.mean(-2),largest,(delta*held).sum(-2)/count,
                            global_difference), -1)
        return torch.cat((factual,contrast), -1)

    def intervention_exposure(self, controls, programs, timing):
        """The unchanged program and a zero-length intervention have no exposure."""
        return (-torch.expm1(-.2*timing[..., 1].to(controls.dtype))*(programs > 0).any(-1))

    @staticmethod
    def event_logits(raw, exposure, competing=False):
        if competing:
            # Rescue and harm are mutually exclusive outcomes of the same
            # paired future. Learn their joint event mass without conditioning
            # on an unobserved future factual outcome. Exposure anchors the
            # unchanged continuation while retaining unrestricted event rates.
            total = F.softplus(torch.logsumexp(raw, -1))*exposure
            probability = -torch.expm1(-total)[..., None]*raw.softmax(-1)
            return torch.logit(probability.clamp(1e-7, 1-1e-7))
        # Two event intensities predict rescue and harm directly in their
        # factual strata. A common intervention exposure anchors both events
        # to the unchanged continuation. Identical commands at the root alone
        # do not imply identical feedback commands throughout the window.
        intensity = F.softplus(raw)*exposure[..., None]
        probability = -torch.expm1(-intensity)
        return torch.logit(probability.clamp(1e-7, 1-1e-7))

    @staticmethod
    def event_gain(logits):
        """Expected paired effect, independent of the factual-value estimate."""
        return logits[..., 1].sigmoid()-logits[..., 2].sigmoid()

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
        if self.physics_residual_response:
            prior_nodes, prior_global = public_physical_response_prior(context, responses[1])
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
            if self.physics_residual_response:
                prior_nodes = prior_nodes.reshape(b, window_count, n, 10, 8)
                prior_global = prior_global.reshape(b, window_count, 6)
        boats = torch.arange(n, device=context.device)
        mask = (~torch.eye(n, dtype=torch.bool, device=context.device))[None, None, :, :, None]
        design = programs if window_inverse is None else torch.cat((window_inverse[:, None], programs), -1)
        distinct, inverse = torch.unique(design, dim=0, return_inverse=True)
        if self.response_conditioned_value:
            representatives = torch.full((len(distinct),), len(programs), dtype=torch.long, device=context.device)
            representatives.scatter_reduce_(0, inverse, torch.arange(len(programs), device=context.device), reduce='amin')
        parts = []; encodings = []
        reference = None
        factual_residual = None
        if self.shared_factual_response:
            factual_responses = (responses if window_inverse is None else
                tuple(item.reshape(b, window_count, *item.shape[1:])[:, 0] for item in responses))
            factual_residual = self.factual_response_residual(context, controls, factual_responses)
        if self.residual_potential or self.paired_response_encoding:
            factual_responses = (responses if window_inverse is None else
                tuple(item.reshape(b, window_count, *item.shape[1:])[:, 0] for item in responses))
            factual_nodes, reference = self.encode(context, controls,
                torch.zeros((b, n), dtype=torch.long, device=context.device),
                torch.zeros((b, 2), dtype=torch.long, device=context.device), factual_responses)
            if self.response_conditioned_value:
                factual_nodes, reference = self.consequence_encode(context, factual_nodes, reference,
                    torch.zeros((b, n), dtype=torch.long, device=context.device),
                    torch.zeros((b, 2), dtype=torch.long, device=context.device),
                    self.physical_prior(context, controls,
                        torch.zeros((b, n), dtype=torch.long, device=context.device),
                        torch.zeros((b, 2), dtype=torch.long, device=context.device), factual_responses), factual_residual)
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
            changed = (encoded*held[None, :, :, None]).sum(2)/held.sum(1)[None, :, None].clamp_min(1)
            global_state = torch.cat((context[:, None, -5:].expand(-1, size, -1),
                rows.new_zeros((b, size, 2)), rows.new_full((b, size, 1), n/6.)), -1)
            representation = torch.cat((encoded.mean(2), encoded.amax(2), changed, global_state), -1)
            if self.response_conditioned_value:
                prior = None
                if self.physics_residual_response:
                    prior = ((prior_nodes[:, boats[None], q], prior_global[:, None].expand(-1, size, -1))
                        if window_inverse is None else
                        (prior_nodes[:, window[:, None], boats[None], q], prior_global[:, window]))
                encoded, representation = self.consequence_encode(context, encoded, representation, q,
                    timing[representatives[start:start+size]], prior,
                    None if factual_residual is None else tuple(item[:, None] for item in factual_residual))
            if return_encodings:
                encodings.append(encoded)
            if self.residual_potential or self.paired_response_encoding:
                factual = reference[:, None].expand(-1, size, -1)
                contrast_head = self.paired_head if self.paired_response_encoding else self.effect_head
                contrast = (self.aligned_response_features(encoded,factual_nodes[:,None],
                    reference[:,None],held[None],representation[..., -8:]-factual[..., -8:])
                    if self.aligned_response_contrast else torch.cat((factual,representation-factual),-1))
                parts.append(contrast_head[0](contrast))
            else:
                parts.append(self.head[0](representation))
        # Timing enters the final conditional-value head. Reuse the exact
        # same joint-plan representation across immediate and delayed choices.
        clock = timing.to(rows.dtype).log1p()/6.*(programs > 0).any(-1, keepdim=True)
        head = (self.paired_head if self.paired_response_encoding else
                self.effect_head if self.residual_potential else self.head)
        hidden = torch.cat(parts, 1)[:, inverse]+F.linear(clock, head[0].weight[:, -3:-1])[None]
        values = head[2](head[1](hidden)).squeeze(-1)
        if self.paired_response_encoding:
            exposure = self.intervention_exposure(controls[:, None], programs[None], timing[None])
            effects = self.event_logits(values, exposure, self.paired_event_difference)
            values = torch.cat((self.head(reference)[:, None].expand(-1, len(programs), -1), effects), -1)
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


def discounted_task_values(success, remaining):
    """Normalize the measured discounted task return, excluding the root potential.

    The existing feedback objective uses gamma=.995, a .001 tick cost and
    terminal rewards +/-1. Potential shaping telescopes to a common root
    constant, which cancels from every paired intervention difference.
    """
    success, remaining = np.broadcast_arrays(np.asarray(success, np.float64),
                                             np.asarray(remaining, np.float64))
    if (not np.isfinite(success).all() or not np.isin(success, (0., 1.)).all()
            or not np.isfinite(remaining).all() or not np.allclose(remaining, np.rint(remaining), atol=1e-3, rtol=0.)
            or np.any((remaining < 1-1e-3) | (remaining > 400+1e-3))):
        raise ValueError('Discounted task values require binary outcomes and measured remaining ticks from 1 to 400.')
    ticks = np.rint(remaining)
    discount = np.power(.995, ticks-1)
    task_return = discount*(2*success-1)-.001*(1-np.power(.995, ticks))/.005
    # One-tick failure and success give -1.001 and .999 respectively.
    return np.clip((task_return+1.001)/2., 0., 1.)


def measured_root_values(root, value_target='success'):
    if value_target == 'success':
        return {key: float(item['success']) for key, item in root['outcomes'].items()}
    if value_target != 'discounted_return':
        raise ValueError('Unknown measured value target.')
    keys = list(root['outcomes'])
    values = discounted_task_values([root['outcomes'][key]['success'] for key in keys],
        [root['outcomes'][key]['duration']/.2-root.get('root_steps', 0) for key in keys])
    return dict(zip(keys, map(float, values)))


def paired_difference_loss(probabilities, difference, value_target='success'):
    """Fit signed improvement/loss mass; retain exact binary-event likelihoods."""
    if value_target == 'success':
        labels = torch.where(difference > 0, 1, torch.where(difference < 0, 2, 0))
        return F.nll_loss(probabilities.clamp_min(1e-7).log(), labels, reduction='none')
    # For bounded continuous returns these are fractional categorical
    # targets. Their positive-minus-negative expectation is the exact
    # measured return difference, including same-success timing effects.
    labels = torch.stack((1-difference.abs(), difference.clamp_min(0.),
                          (-difference).clamp_min(0.)), -1)
    return -(labels*probabilities.clamp_min(1e-7).log()).sum(-1)


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
        difference = after-before
        if 'effect_controls' in root:
            n = len(root['effect_controls'])
            if len(before) == 9*n+5:
                if not np.array_equal(after[8*n:9*n], before[8*n:9*n]):
                    raise ValueError('A measured successor must retain the root physical vessel identities.')
                difference = np.concatenate((difference[:8*n], difference[-5:]))
        indices.append(index)
        targets.append(np.concatenate((difference, np.asarray([steps/20.], np.float32))))
    if not indices:
        raise ValueError('Response supervision requires measured physical successors.')
    return np.asarray(indices, np.int64), np.asarray(targets, np.float32)


def measured_response_contrasts(rows, anchors, response_ids, response_targets):
    """Subtract physical consequences only within the same measured future."""
    lookup = np.full(len(rows), -1, np.int64)
    lookup[response_ids] = np.arange(len(response_ids))
    anchors = np.asarray(anchors, np.int64).reshape(-1, 2)
    pairs = anchors[(lookup[anchors] >= 0).all(-1)]
    if not len(pairs):
        raise ValueError('Physical contrast learning needs paired measured successors under a common future.')
    if (not np.array_equal(rows[pairs[:, 0], :1], rows[pairs[:, 1], :1])
            or not np.array_equal(rows[pairs[:, 0], 2], rows[pairs[:, 1], 2])):
        raise ValueError('Physical contrasts must match their root and common future.')
    targets = response_targets[lookup[pairs[:, 1]]]-response_targets[lookup[pairs[:, 0]]]
    return pairs, targets


def measured_factual_response_records(factual_bank):
    """Align observed 20-tick trajectory successors to each starting vessel order."""
    indices, targets = [], []
    offset = 0
    for record in factual_bank['records']:
        contexts = np.asarray(record['contexts'], np.float32)
        controls = np.asarray(record['controls'])
        remaining = np.asarray(record['remaining'], np.float64)
        orders = np.asarray(record.get('orders'))
        n = controls.shape[1]
        if (contexts.ndim != 2 or contexts.shape[1] not in (8*n+5, 9*n+5)
                or remaining.shape != (len(contexts),) or orders.shape != (len(contexts), n)
                or not np.isfinite(contexts).all() or not np.isfinite(remaining).all()
                or not np.allclose(remaining, np.rint(remaining), rtol=0., atol=1e-6)
                or np.any(np.diff(remaining) >= 0) or np.any(remaining <= 0)
                or not np.array_equal(np.sort(orders, axis=-1), np.broadcast_to(np.arange(n), orders.shape))):
            raise ValueError('Factual response supervision requires measured ordered states and integer trajectory clocks.')
        if contexts.shape[1] == 9*n+5 and not np.allclose(
                contexts[:, 8*n:9*n], orders/max(n-1, 1), rtol=0., atol=1e-6):
            raise ValueError('Factual response identities must agree with their measured vessel orders.')
        positions = {int(round(tick)): index for index, tick in enumerate(remaining)}
        for index, tick in enumerate(remaining):
            following = positions.get(int(round(tick))-20)
            if following is None:
                continue
            correspondence = np.argsort(orders[following])[orders[index].astype(np.int64)]
            nodes = contexts[following, :8*n].reshape(n, 8)[correspondence].reshape(-1)
            before = np.concatenate((contexts[index, :8*n], contexts[index, -5:]))
            after = np.concatenate((nodes, contexts[following, -5:]))
            indices.append(offset+index)
            targets.append(np.concatenate((after-before, np.ones(1, np.float32))))
        offset += len(contexts)
    if not indices:
        raise ValueError('Factual response supervision requires observed successors exactly 20 ticks later.')
    return np.asarray(indices, np.int64), np.asarray(targets, np.float32)


def merge_paired_data(datasets, *, allow_distinct_contexts=False, allow_opponent_mixture=False):
    """Merge observed branches without duplicating a root/program/future."""
    if not datasets:
        raise ValueError('Supply paired terminal observations.')
    relative_modes = {bool(data.get('feedback_relative_hold', False)) for data in datasets}
    if len(relative_modes) != 1:
        raise ValueError('Absolute and feedback-relative interventions are different physical experiments.')
    isolated = all(data.get('isolate_factual_start', False) for data in datasets)
    if True in relative_modes and not isolated:
        raise ValueError('Relative paired branches require isolated activation in every source.')
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
        feedback_relative_hold=next(iter(relative_modes)),
        isolate_factual_start=isolated,
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
        for outcome in root['outcomes'].values():
            if 'following' in outcome:
                # Measured successor rows are already aligned to root order.
                outcome['following'] = append_vessel_identity(torch.as_tensor(outcome['following']),
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


def reconcile_return_root(root, prior, edge_weight=4., ridge=.5):
    """Reconcile continuous task values and their common-future differences.

    Binary rescue strata are inappropriate for discounted returns. Weighted
    effective sample counts keep repeated or rescaled observations from
    manufacturing precision; a neutral unit prior retains uncertainty when
    a small paired sample happens to have zero variance.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    options = root['options']; size = len(options)
    n = len(root['effect_controls']) if 'effect_controls' in root else (len(root['context'])-5)//8
    values = measured_root_values(root, 'discounted_return')
    weights = np.asarray(root.get('future_weights', np.ones(len(root['futures']))), np.float64)
    if weights.shape != (len(root['futures']),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Return reconciliation requires finite nonnegative future weights.')
    futures = {q: sorted(f for option, f in values if option == q and weights[f] > 0) for q in range(size)}
    if not futures[0]:
        raise ValueError('Return reconciliation requires measured factual futures.')
    def moments(samples, future_ids):
        mass = weights[future_ids]; total = mass.sum(); normalized = mass/total
        mean = float(normalized@np.asarray(samples))
        effective = float(1./(normalized@normalized))
        variance = float(normalized@np.square(np.asarray(samples)-mean))
        if effective > 1.+1e-8: variance *= effective/(effective-1.)
        return mean, variance, effective, total
    means = np.asarray(prior, np.float64).copy(); counts = np.zeros(size); effective_counts = np.zeros(size)
    for q, ids in futures.items():
        if ids:
            means[q], _, effective_counts[q], counts[q] = moments([values[q, f] for f in ids], ids)
    baseline = means[0]
    rhs = effective_counts*means+ridge*np.asarray(prior)
    row_indices = []; column_indices = []; entries = []; edge_count = 0
    for a, b in local_edges(options, n):
        ids = sorted(set(futures[a]) & set(futures[b]))
        if not ids: continue
        difference, _, effective, _ = moments([values[b, f]-values[a, f] for f in ids], ids)
        mass = edge_weight*effective
        row_indices.extend((a, b, a, b)); column_indices.extend((a, b, b, a))
        entries.extend((mass, mass, -mass, -mass))
        rhs[b] += mass*difference; rhs[a] -= mass*difference; edge_count += len(ids)
    matrix = sparse.diags(effective_counts+ridge, format='csr')+sparse.coo_matrix(
        (entries, (row_indices, column_indices)), shape=(size, size)).tocsr()
    active = np.flatnonzero(counts[1:] > 0)+1
    reduced = matrix[active][:, active].tocsc()
    reduced_rhs = rhs[active]-matrix[active, 0].toarray().ravel()*baseline
    fitted = spsolve(reduced, reduced_rhs) if len(active) else np.empty(0)
    if np.any((fitted < 0.) | (fitted > 1.)):
        from scipy.optimize import minimize
        solution = minimize(lambda x: (.5*x@reduced@x-reduced_rhs@x, reduced@x-reduced_rhs),
            np.clip(fitted, 0., 1.), jac=True, method='L-BFGS-B', bounds=[(0., 1.)]*len(active),
            options=dict(maxiter=200, ftol=1e-10, gtol=1e-7))
        fitted = solution.x
    result = np.asarray(prior).copy(); result[0] = baseline; result[active] = fitted
    lower = np.full(size, -1.); paired_counts = np.zeros(size)
    for q in active:
        ids = sorted(set(futures[0]) & set(futures[q]))
        if not ids: continue
        mean, variance, effective, _ = moments([values[q, f]-values[0, f] for f in ids], ids)
        # Unit-mass neutral prior on a bounded signed difference; this is a
        # regularized standard error, not a formal coverage guarantee.
        standard_error = np.sqrt((effective*variance+1./3.)/(effective+1.)**2)
        lower[q] = mean-.8416212335729143*standard_error
        paired_counts[q] = effective
    lower[0] = 0.
    uncertainty = np.maximum(0., result-baseline-lower); uncertainty[0] = 0.
    return result, uncertainty, counts, dict(local_difference_observations=edge_count,
        factual_value=float(baseline), factual_observations=len(futures[0]),
        factual_objective_mass=float(counts[0]), value_target='discounted_return',
        confidence_counting='effective weighted futures with a neutral variance prior',
        effective_observation_count=effective_counts.tolist(), paired_effective_count=paired_counts.tolist(),
        gain_lower=lower.tolist())


def reconcile_root(root, prior, edge_weight=4., ridge=.5, effective_future_confidence=False,
                   value_target='success'):
    """Fit one integrable potential using absolute and paired local outcomes."""
    if value_target == 'discounted_return':
        return reconcile_return_root(root, prior, edge_weight, ridge)
    if value_target != 'success':
        raise ValueError('Unknown measured value target.')
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


class FactualWaitHead(nn.Module):
    def __init__(self, input_width, width=64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, -4.6)

    def forward(self, features):
        return self.network(features).squeeze(-1).sigmoid()


class ObservedContinuationHead(nn.Module):
    """Two terminal returns under factual and deferred-policy continuations."""
    def __init__(self, input_width, width=64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        return self.network(features).sigmoid()


class FactualWaitValue(nn.Module):
    """Value of preserving the intervention opportunity under factual feedback.

    Deployment uses only the current public state and current source, rule and
    factual commands. Recorded later states enter the training transitions only.
    """
    def __init__(self, input_width, width, states, mean, scale, observed_returns=False):
        super().__init__()
        if not states:
            raise ValueError('A waiting value requires fitted model states.')
        mean = torch.as_tensor(mean, dtype=torch.float32)
        scale = torch.as_tensor(scale, dtype=torch.float32)
        if (mean.shape != (input_width,) or scale.shape != mean.shape
                or not torch.isfinite(mean).all() or not torch.isfinite(scale).all()
                or (scale <= 0).any()):
            raise ValueError('Waiting-state normalization must be finite and positive.')
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        self.observed_returns = bool(observed_returns)
        # A measured task-return difference contains no artificial margin cost.
        # The decision margin remains the separate current-intervention gate.
        self.margin_adjusted = not self.observed_returns
        head_type = ObservedContinuationHead if self.observed_returns else FactualWaitHead
        self.heads = nn.ModuleList([head_type(input_width, width) for _ in states])
        for head, state in zip(self.heads, states):
            head.load_state_dict(state)

    def forward(self, context, controls):
        features = (state_features(context, controls)-self.mean)/self.scale
        values = torch.stack([head(features) for head in self.heads]).mean(0)
        return values[:, 1]-values[:, 0] if self.observed_returns else values


def factual_wait_targets(next_opportunity, next_wait, elapsed_ticks, live, gamma=.995):
    """One factual continuation step, with both exercise values net of cost."""
    if not 0. < gamma <= 1.:
        raise ValueError('The waiting discount must lie in (0, 1].')
    return gamma**elapsed_ticks*torch.maximum(next_opportunity, next_wait)*live


@torch.no_grad()
def factual_opportunity_transitions(learner, factual_bank, batch_size=16):
    """Score every recorded factual state without reading its outcome labels."""
    from conditional_fusion_value import append_vessel_identity
    if not factual_bank.get('completed') or not factual_bank.get('records'):
        raise ValueError('Waiting needs complete measured factual trajectories.')
    if learner.wait_value is not None:
        raise ValueError('Fit waiting against a frozen intervention value without a waiting module.')
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('Opportunity scoring needs a positive batch size.')
    contexts = []; controls = []; orders = []; following = []; ticks = []; episodes = []
    for episode, record in enumerate(factual_bank['records']):
        x = np.asarray(record['contexts'], np.float32)
        u = np.asarray(record['controls'], np.float32)
        order = np.asarray(record['orders'], np.int64)
        n = u.shape[1] if u.ndim == 3 else 0
        if (not len(x) or x.shape != (len(x), 8*n+5) or u.shape != (len(x), n, 6)
                or order.shape != (len(x), n) or not np.isfinite(x).all() or not np.isfinite(u).all()
                or not np.array_equal(np.sort(order, axis=-1), np.broadcast_to(np.arange(n), order.shape))):
            raise ValueError('Waiting transitions require ordered public states and their actual commands.')
        clock = np.rint(400*x[:, -1]).astype(np.int64)
        if (not np.allclose(400*x[:, -1], clock, atol=1e-4, rtol=0.)
                or np.any(clock % learner.decision_period)
                or np.any(np.diff(clock) != learner.decision_period)):
            raise ValueError('Waiting transitions must retain consecutive factual decision times.')
        start = len(following)
        following.extend(list(range(start+1, start+len(x)))+[-1])
        ticks.extend(list(np.diff(clock))+[learner.decision_period])
        episodes.extend([episode]*len(x))
        contexts.append(x); controls.append(u); orders.append(order)
    context = torch.from_numpy(np.concatenate(contexts))
    control = torch.from_numpy(np.concatenate(controls))
    order = torch.from_numpy(np.concatenate(orders))
    if getattr(learner.models[0], 'identity_conditioned', False):
        context = append_vessel_identity(context, order)
    record_catalog = learner.record_catalog
    learner.record_catalog = True
    scores = []
    try:
        for start in range(0, len(context), batch_size):
            stop = min(start+batch_size, len(context)); size = stop-start
            learner.forward_public(context[start:stop], control[start:stop], order[start:stop],
                torch.zeros((size, control.shape[1], 3)), torch.ones(size, dtype=torch.bool))
            catalog = learner.last_catalog
            if catalog is None or not torch.isfinite(catalog['score']).all():
                raise ValueError('A factual state has no finite deployed catalog scores.')
            scores.append(catalog['score'].amax(-1))
            if stop % (64*batch_size) == 0 or stop == len(context):
                print(dict(phase='factual_opportunity_scoring', ordinary=learner.ordinary,
                    states=stop, total=len(context)), flush=True)
    finally:
        learner.record_catalog = record_catalog
        learner.last_catalog = None
    margin = learner.gain_margin if isinstance(learner, PlainJointReturnDecision) else learner.margin
    opportunity = (torch.cat(scores)-margin).clamp(0., 1.)
    features = state_features(context, control)
    following = torch.tensor(following, dtype=torch.long)
    ticks = torch.tensor(ticks, dtype=torch.float32)
    episodes = torch.tensor(episodes, dtype=torch.long)
    digest = hashlib.sha256()
    for value in (features, following, ticks, episodes):
        digest.update(value.contiguous().numpy().tobytes())
    return dict(features=features, following=following, elapsed_ticks=ticks, episodes=episodes,
        opportunity=opportunity, observation_sha256=digest.hexdigest(), margin=float(margin),
        opportunities_are_model_estimates=True, future_outcome_labels_used=False)


@torch.enable_grad()
def fit_factual_wait_opportunity(model, factual_bank, output, *, updates=3000, width=64,
        ensemble_size=3, seed=42, device='cpu', score_device='cpu', score_batch_size=16, gamma=.995):
    """Fit matched stopping heads; the underlying intervention values stay frozen."""
    if model.get('successor_task_fit', {}).get('value_target') == 'success':
        raise ValueError('Success-valued waiting requires measured continuation pairs, not predicted-opportunity targets.')
    if updates < 1 or ensemble_size < 1 or width < 1:
        raise ValueError('Waiting fitting requires positive capacity and update counts.')
    if not 0. < gamma <= 1.:
        raise ValueError('The waiting discount must lie in (0, 1].')
    recipe = copy.deepcopy(model['successor_task_value'])
    kind = recipe.pop('kind'); recipe['inference_device'] = score_device
    if recipe.get('wait_value') is not None:
        raise ValueError('Use an intervention-only predecessor for waiting fitting.')
    if recipe.get('intervention_budget') != 1 or recipe.get('commit_factual', False):
        raise ValueError('Finite waiting fitting requires one unspent intervention opportunity.')
    if kind == 'plain_joint_return':
        learner = PlainJointReturnDecision(**recipe).eval()
    elif kind == 'local_difference_joint' and not recipe.get('ordinary', False):
        learner = LocalDifferenceDecision(**recipe).eval()
    else:
        raise ValueError('Unknown matched intervention estimator for waiting fitting.')
    transitions = factual_opportunity_transitions(learner, factual_bank, score_batch_size)
    del learner
    features = transitions['features'].to(device)
    mean = features.mean(0); scale = features.std(0, unbiased=False).clamp_min(.05)
    features = (features-mean)/scale
    opportunity = transitions['opportunity'].to(device)
    following = transitions['following'].to(device)
    live = following >= 0; following = following.clamp_min(0)
    elapsed = transitions['elapsed_ticks'].to(device)
    rng = np.random.default_rng(seed+1731)
    torch.manual_seed(seed+1731)
    heads = nn.ModuleList([FactualWaitHead(features.shape[1], width).to(device) for _ in range(ensemble_size)])
    targets = copy.deepcopy(heads).requires_grad_(False)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=3e-4, weight_decay=.01)
    for update in range(updates):
        indices = torch.as_tensor(rng.integers(len(features), size=128), device=device)
        next_ids = following[indices]
        with torch.no_grad():
            future = torch.stack([head(features[next_ids]) for head in targets]).mean(0)
            expected = factual_wait_targets(opportunity[next_ids], future, elapsed[indices], live[indices], gamma)
        loss = sum(F.mse_loss(head(features[indices]), expected) for head in heads)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(heads.parameters(), 2.); optimizer.step()
        with torch.no_grad():
            for destination, source in zip(targets.parameters(), heads.parameters()):
                destination.lerp_(source, .01)
        if (update+1) % 1000 == 0 or update+1 == updates:
            print(dict(phase='factual_wait_fit', ordinary=kind == 'plain_joint_return',
                updates_per_member=update+1, members=ensemble_size, loss=float(loss.detach())), flush=True)
    learned = dict(input_width=features.shape[1], width=width,
        states=[{key: value.detach().cpu() for key, value in head.state_dict().items()} for head in heads],
        mean=mean.cpu(), scale=scale.cpu())
    with torch.no_grad():
        wait = torch.stack([head(features) for head in heads]).mean(0)
        target = factual_wait_targets(opportunity[following], wait[following], elapsed, live, gamma)
        diagnostics = dict(training_bellman_rmse=float((wait-target).square().mean().sqrt()),
            predicted_wait_mean=float(wait.mean()), immediate_opportunity_mean=float(opportunity.mean()),
            predicted_exercise_fraction=float((opportunity>wait).float().mean()))
    result = copy.deepcopy(model)
    result['successor_task_value']['wait_value'] = learned
    result['factual_wait_fit'] = dict(seed=seed, width=width, ensemble_members=ensemble_size,
        updates_per_member=updates, updates=updates*ensemble_size,
        parameters_per_member=sum(parameter.numel() for parameter in heads[0].parameters()),
        factual_trajectories=len(factual_bank['records']), factual_states=len(features),
        training_observation_sha256=transitions['observation_sha256'],
        training_opportunity_sha256=hashlib.sha256(transitions['opportunity'].numpy().tobytes()).hexdigest(),
        intervention_values_updated=False, future_outcomes_used=False,
        future_public_states_used_only_for_training=True, gamma_per_control_tick=gamma,
        decision_margin=transitions['margin'], score_device=score_device, score_batch_size=score_batch_size,
        target='discount by actual elapsed control ticks; maximum of next net intervention score and learned wait',
        terminal_continuation=0., **diagnostics)
    if output is not None:
        torch.save(result, output)
    return result


def intervention_policy_identity(model):
    """Fingerprint the frozen decision rule, including its catalog and thresholds."""
    digest = hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            digest.update(str((str(value.dtype), value.shape)).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                visit(key); visit(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(str(len(value)).encode())
            for item in value: visit(item)
        else:
            digest.update(repr(value.item() if isinstance(value, np.generic) else value).encode())
        digest.update(b'\x00')
    fit = model['successor_task_fit']
    visit((fit['predecessor_sha256'], fit['value_target'], model['successor_task_value']))
    return digest.hexdigest()


def collect_observed_wait_pairs(predecessor, continuation, factual_bank, output=None, *,
        repetitions=4, seed=43, batch_size=16, device='cpu', collection_context=None):
    """Execute factual and five-tick-deferred policies on the same real futures."""
    import conditional_fusion_value as c
    import develop_source_fusion as d
    import run_remote_evidence as r
    predecessor = Path(predecessor); continuation = Path(continuation)
    if output is not None and Path(output).exists():
        raise ValueError('Keep existing measured continuations; choose an unused collection output.')
    source_digest = hashlib.sha256(predecessor.read_bytes()).hexdigest()
    policy_digest = hashlib.sha256(continuation.read_bytes()).hexdigest()
    model = torch.load(continuation, map_location='cpu', weights_only=False)
    recipe = model['successor_task_value']; fit = model['successor_task_fit']
    if (repetitions < 1 or batch_size < 1 or fit['predecessor_sha256'] != source_digest
            or factual_bank['predecessor_sha256'] != source_digest
            or recipe.get('wait_value') is not None or recipe.get('intervention_budget') != 1
            or recipe.get('decision_period', 5) != 5 or fit.get('value_target') != 'success'):
        raise ValueError('Measured success waiting needs a frozen success policy and its actual factual snapshots.')
    records = factual_bank['records']
    if not records or not all(record.get('root_snapshot') is not None for record in records):
        raise ValueError('Each deferred-policy measurement requires a saved factual environment snapshot.')
    r.SOURCE = d.SOURCE
    r.initialize_source_branches(str(d.SOURCE))
    actor = d.joint_load(str(predecessor), predecessor.stat().st_mtime_ns).cpu()
    if actor.successor_task_value is not None:
        raise ValueError('The measured factual continuation cannot contain an intervention value stage.')
    result = dict(kind='actual_wait_continuation_pairs', predecessor_sha256=source_digest,
        continuation_policy_sha256=policy_digest, continuation_policy_identity=intervention_policy_identity(model),
        roots=[], measured_terminal_observations=0, simulation_steps=0, completed=False)
    result['collection_context'] = dict(collection_context or {}, strict_single_vessel_per_intervention=True,
        maximum_interventions=1, defer_ticks=5, value_target='success', source_m1_sha256=source_digest,
        continuation_policy_sha256=policy_digest, root_states=len(records), seed=seed,
        common_futures_per_root=repetitions, future_outcomes_used_to_select_states_or_continuations=False,
        current_policy_measured=True, inference_device=device, inference_batch=batch_size)
    roots_per_batch = max(1, batch_size//repetitions)
    for begin in range(0, len(records), roots_per_batch):
        jobs = []; tasks = []; batch_roots = []
        for record in records[begin:begin+roots_per_batch]:
            actor.cpu()
            snapshot = record['root_snapshot']; scene = tuple(record['scene'])
            tick = int(snapshot['root_steps'])
            if snapshot['predecessor_sha256'] != source_digest or tick % 5:
                raise ValueError('Deferred measurements require this predecessor at an eligible decision clock.')
            env = copy.deepcopy(snapshot['environment']); obs = snapshot['observation'].copy()
            packet = d.mechanism_tensor(d.joint_packet(env, obs))
            with torch.no_grad():
                action = actor(packet, deterministic=True, gate_only=True)[0][0, :, :3].cpu()
                context, order = c.intervention_task_context(packet, True, True)
                control = c.paired_effect_controls(actor, packet, action[None])
            futures = [int(np.random.SeedSequence([seed, int(scene[0]), tick, repeat, 923617]).generate_state(1)[0])
                       for repeat in range(repetitions)]
            root = dict(scene=scene, prefix_future=record['future'], root_steps=tick,
                context=context[0].cpu().numpy(), order=order[0].cpu().numpy(),
                effect_controls=control[0, order[0]].cpu().numpy(), futures=futures,
                future_weights=np.ones(repetitions, np.float32), outcomes={}, maximum_wait_prefix_state_error=0.)
            batch_roots.append(root)
            for future in futures:
                jobs.append(dict(environment=env, observation=obs, random_state=snapshot['random_state'],
                    action=action.numpy().copy(), isolate_factual_start=True))
                tasks.append((scene, future))
        # The explicit factual first action suppresses the root decision. The
        # same public five-tick eligibility clock next permits the frozen M2.
        factual = r.batched_credit_rows(predecessor, tasks, batch_size=batch_size, device=device,
            keep_steps=True, initial_jobs=jobs)
        deferred = r.batched_credit_rows(continuation, tasks, batch_size=batch_size, device=device,
            keep_steps=True, initial_jobs=jobs)
        for index, (left, right) in enumerate(zip(factual, deferred)):
            root = batch_roots[index//repetitions]; future = index % repetitions
            if (len(left['steps'][:5]) != len(right['steps'][:5])
                    or any(a['done'] != b['done'] for a, b in zip(left['steps'][:5], right['steps'][:5]))):
                raise ValueError('A deferred policy changed the factual waiting prefix terminal.')
            for a, b in zip(left['steps'][:5], right['steps'][:5]):
                error = max(float(np.max(np.abs(a['action']-b['action']))),
                            float(np.max(np.abs(a['packet']['node']-b['packet']['node']))))
                root['maximum_wait_prefix_state_error'] = max(root['maximum_wait_prefix_state_error'], error)
            for q, measured in enumerate((left, right)):
                row = measured['row']; result['simulation_steps'] += row['physical_steps']
                root['outcomes'][q, future] = dict(success=float(row['success']),
                    remaining_ticks=int(row['physical_steps']), duration=float(row['duration']),
                    normalized_return=float(row['success']),
                    intervention_count=row['conditional_interventions'], interventions=row['interventions'])
        result['roots'].extend(batch_roots)
        result['measured_terminal_observations'] = 2*repetitions*len(result['roots'])
        if output is not None:
            r.atomic_save(result, output)
        print(dict(phase='measured_success_wait', root_states=len(result['roots']),
            total_roots=len(records), terminal_observations=result['measured_terminal_observations']), flush=True)
    if (hashlib.sha256(predecessor.read_bytes()).hexdigest() != source_digest
            or hashlib.sha256(continuation.read_bytes()).hexdigest() != policy_digest):
        raise ValueError('A frozen continuation changed during its outcome measurement.')
    result['completed'] = True
    observed_wait_examples(result, value_target='success')
    if output is not None:
        torch.save(result, output)
    return result


def observed_wait_examples(data, factual_bank=None, *, identity_conditioned=False,
                           value_target='discounted_return'):
    """Current-state inputs with measured, common-future continuation returns."""
    from conditional_fusion_value import append_vessel_identity
    if value_target not in ('success', 'discounted_return'):
        raise ValueError('Observed waiting needs a measured success or discounted_return target.')
    provenance = data.get('collection_context', {})
    if (data.get('kind') != 'actual_wait_continuation_pairs' or not data.get('completed')
            or not data.get('roots') or not data.get('predecessor_sha256')
            or not data.get('continuation_policy_sha256')):
        raise ValueError('Observed waiting needs complete measured continuation pairs.')
    if (not provenance.get('strict_single_vessel_per_intervention')
            or provenance.get('maximum_interventions') != 1 or provenance.get('defer_ticks') != 5
            or provenance.get('value_target') not in ('success', 'discounted_return')
            or provenance.get('source_m1_sha256') != data['predecessor_sha256']
            or provenance.get('continuation_policy_sha256') != data['continuation_policy_sha256']):
        raise ValueError('Observed waiting requires one deferred intervention and the same factual source.')
    features = []; pair_roots = []; returns = []; probabilities = []
    future_ids = []; factual_features = []; factual_returns = []
    def public_features(context, control, order):
        context = torch.as_tensor(np.asarray(context), dtype=torch.float32)
        control = torch.as_tensor(np.asarray(control), dtype=torch.float32)
        order = torch.as_tensor(np.asarray(order), dtype=torch.long)
        if context.ndim == 1:
            context, control, order = context[None], control[None], order[None]
        if (context.ndim != 2 or control.ndim != 3 or control.shape[0] != len(context)
                or control.shape[-1] != 6 or order.shape != control.shape[:2]
                or not torch.isfinite(context).all() or not torch.isfinite(control).all()):
            raise ValueError('Observed waiting requires finite current public states and actual commands.')
        identified = append_vessel_identity(context, order)
        return state_features(identified if identity_conditioned else context, control)
    for root_id, root in enumerate(data['roots']):
        features.append(public_features(root['context'], root['effect_controls'], root['order']))
        clock = 400*float(root['context'][-1])
        if (abs(clock-root['root_steps']) > 1e-4
                or root.get('maximum_wait_prefix_state_error', float('inf')) >= 2e-5):
            raise ValueError('Observed branches must share the measured factual waiting prefix.')
        futures = root['futures']
        if (not futures or len(set(futures)) != len(futures)
                or set(root['outcomes']) != {(q, f) for q in (0, 1) for f in range(len(futures))}):
            raise ValueError('Every observed waiting branch requires its common-future factual pair.')
        weights = np.asarray(root.get('future_weights', np.ones(len(futures))), np.float64)
        if (weights.shape != (len(futures),) or not np.isfinite(weights).all()
                or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError('Observed waiting requires valid common-future weights.')
        for future, future_id in enumerate(futures):
            endpoints = []
            for q in (0, 1):
                observation = root['outcomes'][q, future]
                value = float(discounted_task_values(observation['success'], observation['remaining_ticks']))
                success = float(observation['success'])
                if success not in (0., 1.):
                    raise ValueError('Observed success targets must be binary terminal outcomes.')
                recorded_value = success if provenance['value_target'] == 'success' else value
                events = observation.get('interventions', [])
                if (observation.get('intervention_count') != len(events) or len(events) > q
                        or any(len(event['subjects']) != 1 or not np.isfinite(event['time'])
                            or event['time'] < .2*(root['root_steps']+5)-1e-4 for event in events)):
                    raise ValueError('Waiting labels must use at most one single-vessel intervention after the delay.')
                if (abs(observation['duration']/.2-root['root_steps']-observation['remaining_ticks']) > 1e-3
                        or not np.isfinite(observation.get('normalized_return', np.nan))
                        or abs(observation['normalized_return']-recorded_value) > 1e-6):
                    raise ValueError('Observed waiting targets must match the recorded terminal objective.')
                endpoints.append(success if value_target == 'success' else value)
            pair_roots.append(root_id); returns.append(endpoints)
            probabilities.append(weights[future]/weights.sum()/len(data['roots']))
            future_ids.append(future_id)
    if (data.get('measured_terminal_observations') != 2*len(returns)
            or provenance.get('root_states') != len(features)):
        raise ValueError('Observed waiting counts disagree with the complete measured data.')
    if factual_bank is not None:
        if (not factual_bank.get('completed') or not factual_bank.get('records')
                or factual_bank.get('predecessor_sha256') != data['predecessor_sha256']):
            raise ValueError('Factual waiting support needs the same completed frozen predecessor.')
        for record in factual_bank['records']:
            x = public_features(record['contexts'], record['controls'], record['orders'])
            values = (np.asarray(record['values'], np.float32) if value_target == 'success' else
                      discounted_task_values(record['values'], record['remaining']))
            if value_target == 'success' and not np.isin(values, [0., 1.]).all():
                raise ValueError('Factual success targets must be binary terminal outcomes.')
            if values.shape != (len(x),):
                raise ValueError('Factual waiting returns must align with current public states.')
            factual_features.append(x); factual_returns.extend(values)
    features = torch.cat(features)
    factual_features = torch.cat(factual_features) if factual_features else features[:0]
    pair_roots = np.asarray(pair_roots, np.int64)
    returns = np.asarray(returns, np.float32)
    probabilities = np.asarray(probabilities, np.float64)
    factual_returns = np.asarray(factual_returns, np.float32)
    observation_digest = hashlib.sha256()
    for value in (features.numpy(), factual_features.numpy(), pair_roots, probabilities,
                  np.asarray(future_ids, np.int64)):
        observation_digest.update(np.ascontiguousarray(value).tobytes())
    target_digest = hashlib.sha256(returns.tobytes()+factual_returns.tobytes()).hexdigest()
    return dict(features=features, pair_roots=pair_roots, returns=returns,
        probabilities=probabilities, factual_features=factual_features, factual_returns=factual_returns,
        value_target=value_target, observation_sha256=observation_digest.hexdigest(), target_sha256=target_digest)


@torch.enable_grad()
def fit_observed_wait_value(model, data, factual_bank, output, *, updates=3000, width=64,
        ensemble_size=3, seed=42, device='cpu', difference_weight=4.):
    """Fit matched absolute endpoints, with paired differences only in the conditional arm."""
    recipe = model['successor_task_value']; fit = model.get('successor_task_fit', {})
    ordinary = recipe['kind'] in ('plain_joint_return', 'plain_mlp_return')
    if not ordinary and (recipe['kind'] not in ('local_difference_joint', 'paired_mlp_return')
                         or recipe.get('ordinary', False)):
        raise ValueError('Observed waiting requires a matched intervention estimator.')
    if (recipe.get('wait_value') is not None or recipe.get('intervention_budget') != 1
            or recipe.get('decision_period', 5) != 5 or recipe.get('commit_factual', False)
            or fit.get('predecessor_sha256') != data.get('predecessor_sha256')
            or fit.get('value_target') not in ('success', 'discounted_return')):
        raise ValueError('Observed waiting needs frozen intervention returns from the same factual source.')
    if (updates < 1 or width < 1 or ensemble_size < 1
            or not np.isfinite(difference_weight) or difference_weight < 0):
        raise ValueError('Observed waiting requires positive capacity and valid fitting settings.')
    policy_matched = data.get('continuation_policy_identity') == intervention_policy_identity(model)
    if fit.get('success_probability_objective', False):
        if not policy_matched:
            raise ValueError('Success waiting requires measured continuations from this frozen intervention policy.')
        protected = {tuple(scene) for scene in fit['validation_layouts']}
        wait_records = data['roots']+(factual_bank['records'] if factual_bank is not None else [])
        if (not all('scene' in record for record in wait_records)
                or protected & {tuple(record['scene']) for record in wait_records}):
            raise ValueError('Waiting supervision must keep the intervention validation layouts untouched.')
    examples = observed_wait_examples(data, factual_bank,
        identity_conditioned=recipe.get('identity_conditioned', False), value_target=fit['value_target'])
    raw = torch.cat((examples['features'], examples['factual_features'])).to(device)
    mean = raw.mean(0); scale = raw.std(0, unbiased=False).clamp_min(.05)
    normalized = (raw-mean)/scale
    features = normalized[:len(examples['features'])]
    factual_features = normalized[len(features):]
    pair_roots = torch.as_tensor(examples['pair_roots'], device=device)
    targets = torch.as_tensor(examples['returns'], device=device)
    factual_targets = torch.as_tensor(examples['factual_returns'], device=device)
    rng = np.random.default_rng(seed+2351); torch.manual_seed(seed+2351)
    heads = nn.ModuleList([ObservedContinuationHead(features.shape[1], width).to(device)
        for _ in range(ensemble_size)])
    optimizer = torch.optim.AdamW(heads.parameters(), lr=3e-4, weight_decay=.01)
    draws = hashlib.sha256()
    for update in range(updates):
        selected = rng.choice(len(targets), size=128, p=examples['probabilities'])
        draws.update(selected.astype(np.int64).tobytes())
        indices = torch.as_tensor(selected, device=device)
        observed = targets[indices]; current = features[pair_roots[indices]]
        factual_indices = None
        if len(factual_features):
            selected_factual = rng.integers(len(factual_features), size=128)
            draws.update(selected_factual.astype(np.int64).tobytes())
            factual_indices = torch.as_tensor(selected_factual, device=device)
        loss = features.new_zeros(())
        for head in heads:
            predicted = head(current)
            loss = loss+F.mse_loss(predicted, observed)
            if not ordinary:
                loss = loss+difference_weight*F.mse_loss(predicted[:, 1]-predicted[:, 0],
                    observed[:, 1]-observed[:, 0])
            if factual_indices is not None:
                loss = loss+F.mse_loss(head(factual_features[factual_indices])[:, 0],
                    factual_targets[factual_indices])
        optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(heads.parameters(), 2.); optimizer.step()
        if (update+1) % 1000 == 0 or update+1 == updates:
            print(dict(phase='observed_wait_fit', ordinary=ordinary, updates_per_member=update+1,
                members=ensemble_size, loss=float(loss.detach())), flush=True)
    learned = dict(input_width=features.shape[1], width=width, observed_returns=True,
        states=[{key: value.detach().cpu() for key, value in head.state_dict().items()} for head in heads],
        mean=mean.cpu(), scale=scale.cpu())
    with torch.no_grad():
        prediction = torch.stack([head(features) for head in heads]).mean(0)[pair_roots]
        weights = torch.as_tensor(examples['probabilities'], dtype=prediction.dtype, device=device)
        difference = prediction[:, 1]-prediction[:, 0]
        measured_difference = targets[:, 1]-targets[:, 0]
        diagnostics = dict(training_endpoint_rmse=float(
            (weights*(prediction-targets).square().mean(-1)).sum().sqrt()),
            training_difference_rmse=float((weights*(difference-measured_difference).square()).sum().sqrt()),
            predicted_wait_mean=float(weights@difference), measured_wait_mean=float(weights@measured_difference))
    result = copy.deepcopy(model)
    result['successor_task_value']['wait_value'] = learned
    result['observed_wait_fit'] = dict(seed=seed, width=width, ensemble_members=ensemble_size,
        updates_per_member=updates, updates=updates*ensemble_size, optimizer_step_calls=updates,
        parameters_per_member=sum(parameter.numel() for parameter in heads[0].parameters()),
        root_states=len(features), paired_futures=len(targets), terminal_observations=2*len(targets),
        factual_trajectories=0 if factual_bank is None else len(factual_bank['records']),
        factual_states=len(factual_features), training_observation_sha256=examples['observation_sha256'],
        training_target_sha256=examples['target_sha256'], training_draw_sha256=draws.hexdigest(),
        value_target=fit['value_target'],
        objective='absolute value' if ordinary else 'absolute value and common-future paired difference',
        difference_weight=0. if ordinary else difference_weight, intervention_values_updated=False,
        future_outcomes_used=True, future_public_states_used=False, current_public_states_only=True,
        predecessor_sha256=data['predecessor_sha256'],
        continuation_policy_sha256=data['continuation_policy_sha256'], defer_ticks=5,
        continuation_policy_identity=data.get('continuation_policy_identity'),
        continuation_policy_matches_intervention=policy_matched,
        collection_context=copy.deepcopy(data['collection_context']),
        target=('measured deferred-policy success minus measured factual success'
            if fit['value_target'] == 'success' else
            'measured discounted deferred-policy return minus measured factual return'),
        margin_subtracted_from_measured_wait=False, **diagnostics)
    if output is not None:
        torch.save(result, output)
    return result


class PlainJointReturnDecision(nn.Module):
    """Absolute-return argmax over the shared complete intervention catalog.

    This execution path contains only fitted networks and candidate programs.
    It has no training-state memory, paired-difference correction, factual
    calibration, uncertainty threshold, inherited value stage or support test.
    """
    def __init__(self, width, states, options, defenders, peer_conditioned=True,
                 context_conditioned=True, motion_basis=True, decision_period=5,
                 inference_device=None, residual_potential=False, finite_motion_basis=False, dual_motion_basis=False,
                 identity_conditioned=False, gain_margin=0., intervention_budget=0, isolate_factual_start=False,
                 response_conditioned_value=False, physics_residual_response=False, shared_factual_response=False,
                 shared_successor_value=False, feedback_relative_hold=False, wait_value=None):
        super().__init__()
        if not np.isfinite(gain_margin) or not 0. <= gain_margin <= 1.:
            raise ValueError('The ordinary graph-return margin must lie between zero and one.')
        if not isinstance(intervention_budget, int) or intervention_budget < 0:
            raise ValueError('The ordinary graph-return intervention budget must be a nonnegative integer.')
        self.gain_margin = float(gain_margin)
        self.intervention_budget = intervention_budget
        self.isolate_factual_start = bool(isolate_factual_start)
        self.feedback_relative_hold = bool(feedback_relative_hold)
        if self.feedback_relative_hold and not self.isolate_factual_start:
            raise ValueError('Relative feedback requires isolated single-vessel interventions.')
        self.models = nn.ModuleList([JointTaskPotential(width, peer_conditioned,
            context_conditioned=context_conditioned, motion_basis=motion_basis,
            residual_potential=residual_potential, finite_motion_basis=finite_motion_basis,
            dual_motion_basis=dual_motion_basis, identity_conditioned=identity_conditioned,
            response_conditioned_value=response_conditioned_value,
            physics_residual_response=physics_residual_response,
            shared_factual_response=shared_factual_response,
            shared_successor_value=shared_successor_value) for _ in states])
        for net, state in zip(self.models, states):
            net.load_state_dict(state)
        self.register_buffer('options', torch.as_tensor(options, dtype=torch.long))
        self.register_buffer('programs', (self.options[:, :1]//(10**torch.arange(defenders))) % 10)
        if self.isolate_factual_start and bool(((self.programs>0).sum(-1)>1).any()):
            raise ValueError('Isolated local execution requires a single-vessel candidate catalog.')
        self.decision_period = decision_period
        self.inference_device = inference_device
        self.prior = None
        self.ordinary = True
        self.record_catalog = False; self.last_catalog = None
        if wait_value is not None and intervention_budget != 1:
            raise ValueError('A fitted waiting value requires a single intervention budget.')
        self.wait_value = None if wait_value is None else FactualWaitValue(**wait_value)

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
        context, order = intervention_task_context(packet, True, True)
        if getattr(self.models[0], 'identity_conditioned', False):
            from conditional_fusion_value import append_vessel_identity
            context = append_vessel_identity(context, order)
        controls = packet['_ordinary_return_controls'].gather(1, order[..., None].expand(-1, -1, 6))
        return self.forward_public(context, controls, order, action, eligible)

    @torch.no_grad()
    def forward_public(self, context, controls, order, action, eligible):
        """Evaluate the same decision from already ordered public measurements."""
        return_device = action.device
        if self.inference_device is not None and self.options.device != torch.device(self.inference_device):
            if str(self.inference_device).startswith('cuda'):
                os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
                torch.use_deterministic_algorithms(True)
                torch.backends.cuda.matmul.allow_tf32 = False
            self.to(self.inference_device)
        if order.shape[1] != self.programs.shape[1]:
            raise ValueError('The return estimator and deployed fleet must have the same size.')
        context = context.to(self.options.device); controls = controls.to(self.options.device)
        order = order.to(self.options.device); action = action.to(self.options.device)
        eligible = eligible.to(self.options.device)
        responses = response_basis(context, controls) if self.models[0].motion_basis else None
        values = torch.stack([net.forward_catalog(context, controls, self.programs,
            self.options[:, 1:], responses=responses).sigmoid() for net in self.models]).mean(0)
        if self.record_catalog:
            gains = values-values[:, :1]
            self.last_catalog = dict(options=self.options.detach().cpu(),
                factual_value=values[:, 0].detach().cpu(),
                predicted_gain=gains.detach().cpu(), score=gains.detach().cpu())
        choice = values.argmax(-1)
        accept = eligible & (choice != 0)
        if self.gain_margin:
            gain = values.gather(1, choice[:, None]).squeeze(1)-values[:, 0]
            accept &= gain > self.gain_margin
        if self.wait_value is not None:
            wait_margin = self.gain_margin if getattr(self.wait_value, 'margin_adjusted', True) else 0.
            immediate = values.gather(1, choice[:, None]).squeeze(1)-values[:, 0]-wait_margin
            continuation = self.wait_value(context, controls)
            accept &= immediate > continuation
            if self.record_catalog:
                self.last_catalog['wait_value'] = continuation.detach().cpu()
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
                 identity_conditioned=False, factual_probability_calibration=True,
                 paired_response_encoding=False, aligned_response_contrast=False, paired_event_difference=False,
                 isolate_factual_start=False, response_conditioned_value=False, physics_residual_response=False,
                 shared_factual_response=False, shared_successor_value=False, feedback_relative_hold=False,
                 wait_value=None, empirical_difference_correction=True):
        super().__init__()
        self.conditional_strata = conditional_strata
        self.paired_event_difference = paired_event_difference
        self.empirical_difference_correction = bool(empirical_difference_correction)
        if paired_event_difference and (ordinary or not conditional_strata or stratum_member_corrections is not None):
            raise ValueError('Paired event differences use one factual head and unconditional paired corrections.')
        self.models = nn.ModuleList([JointTaskPotential(width, peer_conditioned,
            3 if conditional_strata else 1, context_conditioned, local_contrast_head, motion_basis,
            residual_potential, finite_motion_basis, dual_motion_basis, identity_conditioned,
            paired_response_encoding, aligned_response_contrast, paired_event_difference,
            response_conditioned_value, physics_residual_response, shared_factual_response,
            shared_successor_value) for _ in states])
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
        self.isolate_factual_start = bool(isolate_factual_start)
        self.feedback_relative_hold = bool(feedback_relative_hold)
        if self.feedback_relative_hold and not self.isolate_factual_start:
            raise ValueError('Relative feedback requires isolated single-vessel interventions.')
        if identity_conditioned and align_context_roles:
            raise ValueError('Physical vessel identities cannot be reassigned by role matching.')
        self.bandwidth = bandwidth; self.margin = margin; self.confidence = confidence
        self.global_penalty = global_penalty; self.decision_period = decision_period
        self.intervention_budget = intervention_budget; self.prior = None
        self.record_decisions = False; self.last_decision = None
        self.record_catalog = False; self.last_catalog = None
        if wait_value is not None and (intervention_budget != 1 or commit_factual or fleet_transfer is not None):
            raise ValueError('A fitted waiting value requires one opportunity in its fitted fleet without terminal commitment.')
        self.wait_value = None if wait_value is None else FactualWaitValue(**wait_value)
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
        if self.isolate_factual_start and bool(((self.programs>0).sum(-1)>1).any()):
            raise ValueError('Isolated local execution requires a single-vessel candidate catalog.')
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
                    if self.paired_event_difference:
                        raise ValueError('Paired event probabilities cannot be conditioned on a realized future outcome.')
                    baseline = torch.full_like(baseline, float(stratum))
                if self.paired_event_difference:
                    values = baseline+raw[:, :, 1]-raw[:, :, 2]
                    if self.project_task_value:
                        values = values.clamp(0., 1.)
                else:
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
            estimates = ((baseline+estimates[..., 1]-estimates[..., 2]).clamp(0., 1.)
                         if self.paired_event_difference else
                         (1-baseline)*estimates[..., 1]+baseline*(1-estimates[..., 2]))
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
        context, order = intervention_task_context(packet, True, True)
        if getattr(self.models[0], 'identity_conditioned', False):
            from conditional_fusion_value import append_vessel_identity
            context = append_vessel_identity(context, order)
        controls = packet['_local_difference_controls'].gather(1, order[..., None].expand(-1, -1, 6))
        return self.forward_public(context, controls, order, action, eligible)

    @torch.no_grad()
    def forward_public(self, context, controls, order, action, eligible):
        """Keep recorded-state fitting identical to the deployed catalog query."""
        if self.record_decisions:
            self.last_decision = None
        if self.record_catalog:
            self.last_catalog = None
        return_device = action.device
        if self.inference_device is not None and self.options.device != torch.device(self.inference_device):
            if torch.device(self.inference_device).type == 'cuda':
                os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
                torch.use_deterministic_algorithms(True)
                torch.backends.cuda.matmul.allow_tf32 = False
            self.to(self.inference_device)
        b, n = order.shape
        if n != self.programs.shape[1] and self.fleet_transfer is None:
            raise ValueError('Fit a fleet-compatible representation before claiming cross-fleet results.')
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
            if not self.empirical_difference_correction:
                # Keep measured uncertainty, but do not replace a neural effect
                # with an interpolated training outcome before choosing a plan.
                corrected_members = gains
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
        wait = None
        if self.wait_value is not None:
            wait_margin = self.margin if getattr(self.wait_value, 'margin_adjusted', True) else 0.
            if self.ordinary:
                immediate = score.gather(1, choice[:, None]).flatten()-score[:, 0]-wait_margin
            else:
                immediate = score.gather(1, choice[:, None]).flatten()-wait_margin
            wait = self.wait_value(context, controls)
            accept &= immediate > wait
        if self.record_catalog:
            reference = estimates[:, :, 0].mean(0)
            predicted = (estimates.mean(0)-reference[:, None]) if self.ordinary else corrected
            self.last_catalog = dict(options=self.options.detach().cpu(),
                factual_value=reference.detach().cpu(), predicted_gain=predicted.detach().cpu(),
                score=score.detach().cpu())
            if wait is not None:
                self.last_catalog['wait_value'] = wait.detach().cpu()
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
            if wait is not None:
                self.last_decision['wait_value'] = wait.detach().cpu()
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
        tick = context[:, -1].to(action.device)*400.
        remaining = (400-tick.round()).clamp_min(1.)[:, None, None]*one
        committed = torch.cat((action[..., :3], free, remaining, 2*free,
            torch.zeros_like(action[..., :2]), remaining, free.expand(-1, -1, 2),
            free.expand(-1, -1, 2), one), -1)
        factual = torch.where(commit[:, None, None], committed, factual)
        return torch.where(accept[:, None, None], proposed, factual).to(return_device)


def conservative_catalog_loss(proposals, observed_mean, temperature=.1):
    """Penalize catalog optimism relative to actually measured choices.

    Uniform catalog proposals supply values, never fabricated outcome labels.
    The log-mean-exp minus observed-choice mean is invariant to a shared
    factual-value offset, so the same term applies to returns and gains.
    This is the conservative value regularizer of Kumar et al. (NeurIPS 2020)
    applied to supervised finite interventions rather than a Bellman target.
    """
    if proposals.ndim != 2 or observed_mean.shape != proposals.shape[:1] or not proposals.shape[1]:
        raise ValueError('Conservative catalog values need a nonempty proposal matrix and one observed mean per root.')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Conservative catalog temperature must be positive and finite.')
    return temperature*(torch.logsumexp(proposals/temperature, -1)-np.log(proposals.shape[1]))-observed_mean


def _joint_parameter_digest(states):
    digest = hashlib.sha256()
    for state in states:
        for key, value in sorted(state.items()):
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def joint_potential_initialization(model, data, factual_bank=None):
    """Verify a prior fit and preserve its layout separation when continuing it."""
    fit = model['successor_task_fit']; recipe = model['successor_task_value']
    ordinary = bool(fit['ordinary'])
    expected_kind = 'plain_joint_return' if ordinary else 'local_difference_joint'
    if recipe['kind'] != expected_kind or fit['predecessor_sha256'] != data['predecessor_sha256']:
        raise ValueError('Continued learning requires the same estimator and frozen predecessor.')
    if factual_bank is not None and factual_bank['predecessor_sha256'] != data['predecessor_sha256']:
        raise ValueError('Continued factual data require the same frozen predecessor.')
    roots = [observed_programs(root) for root in data['roots']]
    rows = learning_records(dict(data, roots=roots))[0]
    digest = hashlib.sha256()
    for root in roots:
        digest.update(np.asarray(root['context'], np.float32).tobytes())
        digest.update(np.asarray(root['effect_controls'], np.float32).tobytes())
        digest.update(np.asarray(root['options'], np.int64).tobytes())
        digest.update(np.asarray(root.get('future_weights', np.ones(len(root['futures']))), np.float32).tobytes())
        digest.update(np.asarray([(q, f, item['success']) for (q, f), item in
                                  sorted(root['outcomes'].items())], np.float64).tobytes())
    bank_values = None
    if factual_bank is not None:
        arrays = [np.concatenate([record[key] for record in factual_bank['records']]).astype(np.float32)
                  for key in ('contexts', 'controls', 'values', 'remaining')]
        for array in arrays: digest.update(array.tobytes())
        bank_values = (discounted_task_values(arrays[2], arrays[3]).astype(np.float32)
                       if fit['value_target'] == 'discounted_return' else arrays[2])
    if digest.hexdigest() != fit['training_observation_sha256']:
        raise ValueError('Initialization observations do not match the original fit.')
    if fit['value_target'] != 'success':
        measured = [measured_root_values(root, fit['value_target']) for root in roots]
        rows[:, 6] = [measured[int(row[0])][int(row[1]), int(row[2])] for row in rows]
    target_digest = hashlib.sha256(np.asarray(rows[:, 6], np.float32).tobytes())
    if bank_values is not None: target_digest.update(bank_values.tobytes())
    if target_digest.hexdigest() != fit['training_target_sha256']:
        raise ValueError('Initialization return labels do not match the original fit.')
    if fit.get('response_label_sha256') is not None:
        ids, targets = measured_response_records(roots, rows)
        if hashlib.sha256(ids.tobytes()+targets.tobytes()).hexdigest() != fit['response_label_sha256']:
            raise ValueError('Initialization physical responses do not match the original fit.')
    if fit.get('factual_response_label_sha256') is not None:
        ids, targets = measured_factual_response_records(factual_bank)
        if hashlib.sha256(ids.tobytes()+targets.tobytes()).hexdigest() != fit['factual_response_label_sha256']:
            raise ValueError('Initialization factual responses do not match the original fit.')
    states = copy.deepcopy(recipe['states']); members = len(states)
    folds = None
    if fit.get('factual_crossfit', False):
        root_folds, bank_folds, folds = factual_crossfit_groups(roots, factual_bank, members, fit['seed'])
        if (root_folds.tolist() != fit['factual_crossfit_root_folds']
                or bank_folds.tolist() != fit['factual_crossfit_trajectory_folds']):
            raise ValueError('Initialization layout folds do not match the original fit.')
    return dict(initial_states=states, initial_training_context=dict(
        ordinary=ordinary, width=fit['width'], seed=fit['seed'], ensemble_size=members,
        predecessor_sha256=fit['predecessor_sha256'], value_target=fit['value_target'],
        isolate_factual_start=bool(fit.get('isolate_factual_start', False)),
        feedback_relative_hold=bool(fit.get('feedback_relative_hold', False)),
        factual_crossfit=bool(fit.get('factual_crossfit', False)), layout_folds=folds,
        source_parameter_sha256=_joint_parameter_digest(states),
        source_training_observation_sha256=fit['training_observation_sha256'],
        source_training_target_sha256=fit['training_target_sha256'],
        source_updates=fit['updates'],
        source_total_updates=fit.get('cumulative_optimizer_steps', fit['updates'])))


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
                        factual_calibration_kind='probability', paired_response_encoding=False,
                        decision_regret_weight=0., aligned_response_contrast=False, paired_event_difference=False,
                        response_supervision_architecture='mlp', response_conditioned_value=False,
                        response_contrast_weight=0., physics_residual_response=False, response_value_weight=0.,
                        value_target='success', factual_response_supervision=False, shared_factual_response=False,
                        shared_successor_value=False, conservative_catalog_weight=0.,
                        conservative_catalog_temperature=.1, conservative_catalog_samples=64,
                        conservative_catalog_roots=4, initial_training_context=None,
                        refit_all_layouts=False, empirical_difference_correction=True,
                        factual_probability_calibration=True):
    from plain_return import root_sampling_probabilities
    if value_target not in ('success', 'discounted_return'):
        raise ValueError('Choose measured success or discounted_return as the value target.')
    if value_target == 'discounted_return' and not ordinary and not (paired_event_difference or shared_successor_value):
        raise ValueError('Conditional discounted returns require paired difference learning or a shared successor value.')
    if value_target == 'discounted_return' and data['model'].get('success_objective', False):
        raise ValueError('Discounted returns require a predecessor using the original discounted task objective.')
    if not np.isfinite(response_supervision_weight) or response_supervision_weight < 0:
        raise ValueError('Response supervision must have a finite nonnegative weight.')
    if response_supervision_architecture not in ('mlp', 'linear'):
        raise ValueError('Choose a linear or MLP measured-response auxiliary head.')
    if response_conditioned_value and (not response_supervision_weight or response_supervision_architecture != 'linear'
                                      or not finite_motion_basis or local_contrast_head or residual_potential):
        raise ValueError('Response-conditioned value requires supervised linear response heads and finite motion features.')
    if physics_residual_response and not response_conditioned_value:
        raise ValueError('A physical response prior requires the response-conditioned value representation.')
    if shared_factual_response and (not physics_residual_response or response_contrast_weight):
        raise ValueError('Shared factual responses require public physics and omit the fixed, nonlearnable contrast loss.')
    if shared_successor_value and (not response_conditioned_value or conditional_strata
                                  or paired_response_encoding or residual_potential or local_contrast_head):
        raise ValueError('Shared successor values require one response-conditioned scalar potential.')
    if (not np.isfinite(response_contrast_weight) or response_contrast_weight < 0
            or (response_contrast_weight and (ordinary or not response_conditioned_value))):
        raise ValueError('Physical contrast learning requires a nonnegative paired response-conditioned objective.')
    if (not np.isfinite(response_value_weight) or response_value_weight < 0
            or (response_value_weight and (not response_conditioned_value
                or not (ordinary or paired_event_difference or shared_successor_value)))):
        raise ValueError('Measured consequence values require a nonnegative response-conditioned ordinary or paired objective.')
    if factual_response_supervision and (factual_bank is None or not response_conditioned_value):
        raise ValueError('Factual response supervision requires measured trajectories and a response-conditioned value model.')
    if response_supervision_weight and ordinary and ordinary_architecture != 'interaction':
        raise ValueError('Measured-response supervision requires the interaction-return comparator.')
    if dual_motion_basis and not finite_motion_basis:
        raise ValueError('The dual motion basis requires finite motion features.')
    if refit_all_layouts and (ordinary or factual_crossfit or not paired_event_difference
            or initial_training_context is None
            or not initial_training_context.get('factual_crossfit', False)):
        raise ValueError('Full-layout refitting requires a verified cross-fitted paired-event prior and no new calibration folds.')
    if initial_training_context is not None:
        expected = dict(ordinary=ordinary, width=width, seed=seed, ensemble_size=ensemble_size,
            predecessor_sha256=data.get('predecessor_sha256'), value_target=value_target,
            isolate_factual_start=bool(data.get('isolate_factual_start', False)),
            feedback_relative_hold=bool(data.get('feedback_relative_hold', False)),
            factual_crossfit=True if refit_all_layouts else factual_crossfit)
        if initial_states is None or any(initial_training_context.get(key) != value for key, value in expected.items()):
            raise ValueError('Continued learning must preserve its estimator, execution semantics and layout grouping.')
        if (not isinstance(initial_training_context.get('source_total_updates'), int)
                or initial_training_context['source_total_updates'] < 0):
            raise ValueError('Continued learning requires the measured prior optimizer budget.')
    if initial_states is not None and ordinary and (ordinary_architecture != 'interaction'
                                                    or initial_training_context is None):
        raise ValueError('Ordinary return learning cannot inherit conditional parameters.')
    if updates < 0 or (updates == 0 and initial_states is None):
        raise ValueError('Calibration without optimizer updates requires a fitted parameter prior.')
    if initial_states is not None and len(initial_states) != ensemble_size:
        raise ValueError('Warm initialization must provide every ensemble member.')
    if factual_crossfit and (ordinary or not conditional_strata or ensemble_size < 2
                            or initial_states is not None and initial_training_context is None):
        raise ValueError('Factual cross-fitting requires fresh members or verified layout-preserving initialization.')
    if factual_calibration_kind not in ('probability', 'logit'):
        raise ValueError('Unknown factual probability calibration map.')
    if paired_response_encoding and (ordinary or not conditional_strata):
        raise ValueError('Paired response learning needs the two factual outcome strata.')
    if aligned_response_contrast and not paired_response_encoding:
        raise ValueError('Aligned response learning requires paired response encoding.')
    if paired_event_difference and (not paired_response_encoding or stratified_local_calibration):
        raise ValueError('Paired event learning requires paired encodings and marginal outcome calibration.')
    if not np.isfinite(decision_regret_weight) or decision_regret_weight < 0 or (decision_regret_weight and ordinary):
        raise ValueError('Paired decision regret requires a nonnegative conditional-learning weight.')
    if not np.isfinite(conservative_catalog_weight) or conservative_catalog_weight < 0:
        raise ValueError('Conservative catalog weight must be finite and nonnegative.')
    if (not np.isfinite(conservative_catalog_temperature) or conservative_catalog_temperature <= 0
            or not isinstance(conservative_catalog_samples, int) or conservative_catalog_samples < 1
            or not isinstance(conservative_catalog_roots, int) or conservative_catalog_roots < 1):
        raise ValueError('Conservative catalog sampling requires positive temperature and integer sample counts.')
    if conservative_catalog_weight and (not finite_motion_basis
            or (ordinary and ordinary_architecture != 'interaction')
            or (not ordinary and not paired_event_difference)):
        raise ValueError('Conservative catalog learning requires finite motion and paired-event or ordinary interaction values.')
    snapshot_updates = tuple(sorted(set(map(int, snapshot_updates))))
    if any(step <= 0 or step > updates or step % ensemble_size for step in snapshot_updates):
        raise ValueError('Learning snapshots need positive total budgets shared equally across ensemble members.')
    initialization_sha256 = None
    if initial_states is not None:
        initialization_sha256 = _joint_parameter_digest(initial_states)
        if (initial_training_context is not None
                and initialization_sha256 != initial_training_context['source_parameter_sha256']):
            raise ValueError('Initialization parameters do not match their verified training context.')
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
            additional_candidate_options=additional_candidate_options, value_target=value_target)
    roots = [observed_programs(root) for root in data['roots']]
    n = len(roots[0]['effect_controls'])
    if any(len(root['context']) != (8+identity_conditioned)*n+5 for root in roots):
        raise ValueError('Training roots must match the requested public identity representation.')
    conditional_strata = bool(conditional_strata and not ordinary)
    local_contrast_head = local_contrast_head if not ordinary else False
    rows, edges, anchors, diagnostics = learning_records(dict(data, roots=roots))
    measured_values = [measured_root_values(root, value_target) for root in roots]
    if value_target != 'success':
        rows[:, 6] = [measured_values[int(row[0])][int(row[1]), int(row[2])] for row in rows]
    row_probability = root_sampling_probabilities(rows[:, 0], root_sampling_fraction)
    root_folds = bank_folds = layout_folds = None
    if factual_crossfit:
        root_folds, bank_folds, layout_folds = factual_crossfit_groups(roots, factual_bank, ensemble_size, seed)
        if initial_training_context is not None:
            previous_folds = initial_training_context.get('layout_folds')
            if not previous_folds or any(layout_folds.get(scene) != fold for scene, fold in previous_folds.items()):
                raise ValueError('Continued learning would move a previously trained layout into its held-out fold.')
    def population_probability(indices):
        return root_sampling_probabilities(rows[indices, 0], root_sampling_fraction) if len(indices) else None
    response_ids = response_targets = None
    response_label_sha256 = None
    response_pairs = response_pair_targets = response_pair_rows = None
    response_pair_sha256 = response_lookup = None
    if response_supervision_weight:
        response_ids, response_targets = measured_response_records(roots, rows)
        response_probability = population_probability(response_ids)
        response_label_sha256 = hashlib.sha256(response_ids.tobytes()+response_targets.tobytes()).hexdigest()
        if response_contrast_weight or response_value_weight:
            response_pair_rows, response_pair_targets = measured_response_contrasts(
                rows, anchors, response_ids, response_targets)
            response_pair_probability = population_probability(response_pair_rows[:, 0])
            response_pair_sha256 = hashlib.sha256(
                response_pair_rows.tobytes()+response_pair_targets.tobytes()).hexdigest()
            response_pairs = torch.as_tensor(response_pair_rows, dtype=torch.long, device=device)
            response_pair_targets = torch.as_tensor(response_pair_targets, device=device)
        response_ids = torch.as_tensor(response_ids, device=device)
        response_targets = torch.as_tensor(response_targets, device=device)
        if response_value_weight:
            response_lookup = torch.full((len(rows),), -1, dtype=torch.long, device=device)
            response_lookup[response_ids] = torch.arange(len(response_ids), device=device)
    digest = hashlib.sha256()
    for root in roots:
        digest.update(np.asarray(root['context'], np.float32).tobytes())
        digest.update(np.asarray(root['effect_controls'], np.float32).tobytes())
        digest.update(np.asarray(root['options'], np.int64).tobytes())
        digest.update(np.asarray(root.get('future_weights', np.ones(len(root['futures']))), np.float32).tobytes())
        digest.update(np.asarray([(q, f, item['success']) for (q, f), item in
                                  sorted(root['outcomes'].items())], np.float64).tobytes())
    continuation_context = continuation_controls = continuation_target = continuation_remaining = None
    bank_response_ids = bank_response_targets = None
    bank_response_sha256 = None
    if factual_bank is not None:
        if factual_bank['predecessor_sha256'] != data['predecessor_sha256']:
            raise ValueError('Continuation supervision requires the common frozen predecessor.')
        arrays = [np.concatenate([record[key] for record in factual_bank['records']]).astype(np.float32)
                  for key in ('contexts', 'controls', 'values', 'remaining')]
        if arrays[0].shape[-1] != (8+identity_conditioned)*n+5:
            raise ValueError('Factual trajectories must use the same measured identity representation.')
        for array in arrays: digest.update(array.tobytes())
        if value_target == 'discounted_return':
            arrays[2] = discounted_task_values(arrays[2], arrays[3]).astype(np.float32)
        continuation_context, continuation_controls, continuation_target, continuation_remaining = [
            torch.as_tensor(array, device=device) for array in arrays]
        if factual_response_supervision:
            bank_response_ids, bank_targets = measured_factual_response_records(factual_bank)
            bank_digest = hashlib.sha256(bank_response_ids.tobytes())
            bank_digest.update(bank_targets.tobytes())
            bank_response_sha256 = bank_digest.hexdigest()
            bank_response_targets = torch.as_tensor(bank_targets, device=device)
    target_digest = hashlib.sha256(np.asarray(rows[:, 6], np.float32).tobytes())
    if continuation_target is not None:
        target_digest.update(continuation_target.detach().cpu().numpy().tobytes())
    extra_options = {tuple(map(int, option)) for option in additional_candidate_options}
    if any(len(option) != 3 or not (0 <= option[0] < 10**n) or min(option[1:]) < 0
           or (option[0] == 0 and option != (0, 0, 0)) for option in extra_options):
        raise ValueError('The transferred catalog must contain fleet-compatible finite interventions.')
    options = [(0, 0, 0)]+sorted(({tuple(option) for root in roots for option in root['options']
                                  if option[0]} | extra_options)-{(0, 0, 0)})
    isolate_factual_start = bool(data.get('isolate_factual_start', False))
    feedback_relative_hold = bool(data.get('feedback_relative_hold', False))
    if feedback_relative_hold and (not isolate_factual_start or
            not all(root.get('feedback_relative_hold', False) for root in roots)):
        raise ValueError('Relative value learning needs measured relative branches from every root.')
    if isolate_factual_start and any(sum(code//10**boat % 10 != 0 for boat in range(n))>1
                                    for code, _, _ in options):
        raise ValueError('Isolated local data and the deployment catalog must change only one vessel.')
    context = torch.as_tensor(np.asarray([r['context'] for r in roots]), dtype=torch.float32, device=device)
    controls = torch.as_tensor(np.asarray([r['effect_controls'] for r in roots]), dtype=torch.float32, device=device)
    root_ids = torch.as_tensor(rows[:, 0], dtype=torch.long, device=device)
    codes = torch.as_tensor(rows[:, 3], dtype=torch.long, device=device)
    digits = (codes[:, None]//(10**torch.arange(n, device=device))) % 10
    timing = torch.as_tensor(rows[:, 4:6], dtype=torch.long, device=device)
    target = torch.as_tensor(rows[:, 6], dtype=torch.float32, device=device)
    factual_labels = np.asarray([measured_values[int(row[0])][0, int(row[2])] for row in rows])
    factual_target = torch.as_tensor(factual_labels, dtype=torch.float32, device=device)
    importance_array = np.asarray([roots[int(row[0])].get('future_weights',
        np.ones(len(roots[int(row[0])]['futures'])))[int(row[2])] for row in rows], np.float32)
    importance = torch.as_tensor(importance_array, device=device)
    decision_groups = []
    if decision_regret_weight:
        grouped = {}
        for index, row in enumerate(rows):
            grouped.setdefault((int(row[0]), int(row[2])), []).append(index)
        decision_groups = [np.asarray(indices, np.int64) for indices in grouped.values() if len(indices) > 1]
        if not decision_groups:
            raise ValueError('Paired decision regret needs alternatives measured under a common root and future.')
    standardization = np.ones(len(rows), np.float32)
    factual_population = np.asarray([np.average(
        [measured_values[root_id][q, f] for q, f in root['outcomes'] if q == 0],
        weights=[root.get('future_weights', np.ones(len(root['futures'])))[f]
                 for (q, f), v in root['outcomes'].items() if q == 0]) for root_id, root in enumerate(roots)], np.float32)
    if conditional_strata and not paired_event_difference:
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
    conservative_responses = None
    if conservative_catalog_weight:
        catalog_options = torch.as_tensor(options, dtype=torch.long, device=device)
        catalog_digits = (catalog_options[:, :1]//(10**torch.arange(n, device=device))) % 10
        catalog_windows, catalog_window_ids = torch.unique(catalog_options[:, 1:], dim=0, return_inverse=True)
        window_count = len(catalog_windows)
        keys = torch.cat((torch.arange(len(roots), device=device)[:, None, None].expand(-1, window_count, 1),
            catalog_windows[None].expand(len(roots), -1, -1)), -1).reshape(-1, 3)
        chunks = [response_basis(context[key[:, 0]], controls[key[:, 0]], key[:, 1:], dual_motion_basis)
                  for key in keys.split(128)]
        conservative_responses = tuple(torch.cat([part[j] for part in chunks]).reshape(
            len(roots), window_count, *chunks[0][j].shape[1:]) for j in range(3))
        lookup = {option:index for index, option in enumerate(options)}
        measured_options = [[lookup[tuple(option)] for option in root['options']] for root in roots]
        reference_width = max(map(len, measured_options))
        reference_indices = np.zeros((len(roots), reference_width), np.int64)
        reference_mask = np.zeros((len(roots), reference_width), np.float32)
        for index, measured in enumerate(measured_options):
            reference_indices[index, :len(measured)] = measured
            reference_mask[index, :len(measured)] = 1.
        reference_indices = torch.as_tensor(reference_indices, device=device)
        reference_mask = torch.as_tensor(reference_mask, device=device)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    def draw(count, size, probability):
        return rng.integers(count, size=size) if probability is None else rng.choice(count, size=size, p=probability)
    states = []; losses = []; response_contrast_scales = []
    snapshots = {budget:[] for budget in snapshot_updates}
    factual_calibrations = []; factual_calibration_diagnostics = []
    calibration_snapshots = {budget:[] for budget in snapshot_updates} if factual_crossfit else None
    calibration_diagnostic_snapshots = {budget:[] for budget in snapshot_updates} if factual_crossfit else None
    per_member = [updates//ensemble_size+int(i < updates % ensemble_size) for i in range(ensemble_size)]
    for member, iterations in enumerate(per_member):
        if conservative_catalog_weight:
            conservative_rng = np.random.default_rng(np.random.SeedSequence([seed, 840013, member]))
            conservative_training_roots = (np.flatnonzero(root_folds != member) if factual_crossfit
                                           else np.arange(len(roots)))
        net = JointTaskPotential(width, peer_conditioned, 3 if conditional_strata else 1,
                                context_conditioned, local_contrast_head, motion_basis,
                                residual_potential, finite_motion_basis, dual_motion_basis,
                                identity_conditioned, paired_response_encoding, aligned_response_contrast,
                                paired_event_difference, response_conditioned_value, physics_residual_response,
                                shared_factual_response, shared_successor_value).to(device)
        if initial_states is not None:
            net.load_state_dict(initial_states[member])
        train_parameters = list(net.parameters())
        if response_supervision_weight and not response_conditioned_value:
            # Both estimators observe exactly the same measured successor
            # coordinates. Small linear probes keep their auxiliary training
            # capacities close when the ordinary value network is wider.
            if response_supervision_architecture == 'linear':
                response_node = nn.Linear(width, 8).to(device)
                response_global = nn.Linear(3*width+8, 6).to(device)
            else:
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
        if response_contrast_weight or response_value_weight:
            member_response_pair_probability = training_probability(response_pair_probability, response_pair_rows[:, 0])
        if response_contrast_weight:
            mass = torch.as_tensor(np.ones(len(response_pairs)) if member_response_pair_probability is None
                else member_response_pair_probability, dtype=context.dtype, device=device)
            mass = mass*importance[response_pairs[:, 0]]*root_weight[root_ids[response_pairs[:, 0]]]
            # Scale effects using this member's training population only.
            # Held-out layouts and neutral outcome frequencies are preserved;
            # the floor prevents an almost static coordinate dominating.
            response_contrast_scale = ((response_pair_targets.square()*mass[:, None]).sum(0)
                /mass.sum().clamp_min(1e-8)).sqrt().clamp_min(.05)
            response_contrast_scales.append(response_contrast_scale.cpu().tolist())
        member_decision_groups = [group for group in decision_groups if not factual_crossfit
                                  or root_folds[int(rows[group[0], 0])] != member]
        bank_state_folds = bank_state_weights = member_bank_probability = None
        if factual_crossfit and factual_bank is not None:
            lengths = [len(record['values']) for record in factual_bank['records']]
            bank_state_folds = np.repeat(bank_folds, lengths)
            bank_state_weights = np.concatenate([np.full(length, 1./length) for length in lengths if length])
            active = (bank_state_folds != member).astype(np.float64)
            if active.sum() <= 0:
                raise ValueError('A factual cross-fit member has no factual training trajectories.')
            member_bank_probability = active/active.sum()
        member_bank_response_probability = None
        if factual_response_supervision and factual_crossfit:
            mass = (bank_state_folds[bank_response_ids] != member).astype(np.float64)
            if mass.sum() <= 0:
                raise ValueError('A factual cross-fit member has no measured trajectory responses in its training fold.')
            member_bank_response_probability = mass/mass.sum()
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
            if paired_event_difference:
                # Both event heads observe every paired future, including the
                # neutral majority. Their signed difference is the expected
                # intervention effect conditional on the current fleet state.
                value = factual_population[root_ids[index]]+net.event_gain(raw)
                return torch.where(codes[index] == 0, raw[:, 0],
                    torch.logit(value.clamp(1e-6, 1-1e-6)))
            if condition:
                logits = torch.where(factual_target[index] > .5, -raw[:, 2], raw[:, 1])
                return torch.where(codes[index] == 0, 24.*factual_target[index]-12., logits)
            # Candidate contrasts do not move the independently fitted factual
            # value; they learn repair and damage conditional on that reference.
            factual = factual_population[root_ids[index]]
            value = (1-factual)*raw[:, 1].sigmoid()+factual*(1-raw[:, 2].sigmoid())
            return torch.where(codes[index] == 0, raw[:, 0], torch.logit(value.clamp(1e-6, 1-1e-6)))
        def paired_gain(index):
            raw = net(context[root_ids[index]], controls[root_ids[index]], digits[index], timing[index],
                      select_responses(responses, response_index[index]))
            return net.event_gain(raw)
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
                    if response_conditioned_value:
                        node_response, global_response = net.response_prediction(encoded, representation,
                            digits[index], timing[index], net.physical_prior(context[root_ids[index]],
                                controls[root_ids[index]], digits[index], timing[index],
                                select_responses(responses, response_index[index])),
                            net.factual_response_residual(context[root_ids[index]], controls[root_ids[index]],
                                select_responses(responses, response_index[index])) if shared_factual_response else None)
                        prediction = torch.cat((node_response.flatten(1), global_response), -1)
                    else:
                        prediction = torch.cat((response_node(encoded).flatten(1), response_global(representation)), -1)
                    error = F.smooth_l1_loss(prediction, response_targets[positions], reduction='none').mean(-1)
                    mass = importance[index]*root_weight[root_ids[index]]
                    loss += response_supervision_weight*(error*mass).sum()/mass.sum().clamp_min(1e-8)
                if response_contrast_weight:
                    positions = torch.as_tensor(draw(len(response_pairs), 32, member_response_pair_probability), device=device)
                    pair = response_pairs[positions]; index = pair.flatten()
                    encoded, representation = net.encode(context[root_ids[index]], controls[root_ids[index]],
                        digits[index], timing[index], select_responses(responses, response_index[index]))
                    node_response, global_response = net.response_prediction(encoded, representation,
                        digits[index], timing[index], net.physical_prior(context[root_ids[index]],
                            controls[root_ids[index]], digits[index], timing[index],
                            select_responses(responses, response_index[index])))
                    prediction = torch.cat((node_response.flatten(1), global_response), -1).reshape(len(pair), 2, -1)
                    # Shared future disturbances cancel before the loss is
                    # formed. No measured successor enters a decision input.
                    difference = prediction[:, 1]-prediction[:, 0]
                    error = F.smooth_l1_loss(difference/response_contrast_scale,
                        response_pair_targets[positions]/response_contrast_scale, reduction='none').mean(-1)
                    mass = importance[pair[:, 0]]*root_weight[root_ids[pair[:, 0]]]
                    loss += response_contrast_weight*(error*mass).sum()/mass.sum().clamp_min(1e-8)
                if response_value_weight:
                    positions = torch.as_tensor(draw(len(response_pairs), 32, member_response_pair_probability), device=device)
                    pair = response_pairs[positions]
                    if ordinary or shared_successor_value:
                        index = pair.flatten()
                        observed_logits = net.observed_response_logits(context[root_ids[index]], controls[root_ids[index]],
                            digits[index], timing[index], response_targets[response_lookup[index]])
                        error = F.binary_cross_entropy_with_logits(observed_logits, target[index], reduction='none')
                        error = error.reshape(-1, 2).mean(-1)
                        if not ordinary:
                            # The identical scalar evaluator learns both the
                            # absolute outcomes and their same-future contrast.
                            # This contrast has no separate rescue/harm head.
                            predicted = observed_logits.sigmoid().reshape(-1, 2).diff(dim=1).flatten()
                            difference = target[pair[:, 1]]-target[pair[:, 0]]
                            error = error+anchor_weight*(predicted-difference).square()
                    else:
                        index = pair[:, 1]
                        observed_logits = net.observed_response_logits(context[root_ids[index]], controls[root_ids[index]],
                            digits[index], timing[index], response_targets[response_lookup[index]],
                            factual=response_targets[response_lookup[pair[:, 0]]])
                        events = observed_logits[:, 1:].sigmoid()
                        probabilities = torch.cat(((1-events.sum(-1, keepdim=True)).clamp_min(1e-7), events), -1)
                        difference = target[index]-target[pair[:, 0]]
                        error = .5*(paired_difference_loss(probabilities, difference, value_target)+
                            F.binary_cross_entropy_with_logits(observed_logits[:, 0], target[pair[:, 0]], reduction='none'))
                    # Ground the same evaluator used on predicted successors
                    # in actual physical consequences. Only this training
                    # branch receives measured future states.
                    mass = importance[pair[:, 0]]*root_weight[root_ids[pair[:, 0]]]
                    loss += response_value_weight*(error*mass).sum()/mass.sum().clamp_min(1e-8)
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
                if factual_response_supervision:
                    positions = draw(len(bank_response_ids), 32, member_bank_response_probability)
                    index = torch.as_tensor(bank_response_ids[positions], device=device)
                    observed = bank_response_targets[torch.as_tensor(positions, device=device)]
                    programs = torch.zeros((len(index), n), dtype=torch.long, device=device)
                    clock = torch.zeros((len(index), 2), dtype=torch.long, device=device)
                    basis = select_responses(continuation_responses, index)
                    encoded, representation = net.encode(continuation_context[index],
                        continuation_controls[index], programs, clock, basis)
                    node_response, global_response = net.response_prediction(encoded, representation,
                        programs, clock, net.physical_prior(continuation_context[index],
                            continuation_controls[index], programs, clock, basis),
                        net.factual_response_residual(continuation_context[index], continuation_controls[index], basis)
                            if shared_factual_response else None)
                    prediction = torch.cat((node_response.flatten(1), global_response), -1)
                    loss += response_supervision_weight*F.smooth_l1_loss(prediction, observed)
                    if response_value_weight:
                        observed_logits = net.observed_response_logits(continuation_context[index],
                            continuation_controls[index], programs, clock, observed, observed)
                        if conditional_strata: observed_logits = observed_logits[:, 0]
                        loss += response_value_weight*F.binary_cross_entropy_with_logits(
                            observed_logits, continuation_target[index])
                if not ordinary:
                    if conditional_strata:
                        candidates = ids[codes[ids] != 0]
                        if len(candidates):
                            if paired_event_difference:
                                raw = net(context[root_ids[candidates]], controls[root_ids[candidates]],
                                    digits[candidates], timing[candidates],
                                    select_responses(responses, response_index[candidates]))
                                events = raw[:, 1:].sigmoid()
                                probabilities = torch.cat(((1-events.sum(-1, keepdim=True)).clamp_min(1e-7), events), -1)
                                difference = target[candidates]-factual_target[candidates]
                                error = paired_difference_loss(probabilities, difference, value_target)
                                loss += (error*importance[candidates]).sum()/importance[candidates].sum()
                            else:
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
                        elif paired_event_difference:
                            # Every endpoint is centered on the same measured
                            # factual continuation. Its predicted value must
                            # cancel, including when one endpoint is unchanged.
                            # Keeping the signed effect outside probability
                            # clipping also preserves gradients at factual
                            # success rates of zero and one.
                            values = paired_gain(pair.flatten()).reshape(-1, 2)
                            predicted_difference = values[:, 1]-values[:, 0]
                            node_difference = None
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
                        pair_logits = (paired_gain(pair.flatten())/.15 if paired_event_difference else
                            predict(pair.flatten(), condition=conditional_strata)).reshape(-1, 2)
                        preferred = (target[pair[:, 1]] > target[pair[:, 0]]).to(target.dtype)
                        informative = ((target[pair[:, 1]]-target[pair[:, 0]]).abs() if value_target == 'discounted_return'
                            else (target[pair[:, 1]] != target[pair[:, 0]]).to(target.dtype))
                        mass = importance[pair[:, 0]]
                        # Normalize over the entire local comparison population.
                        # Conditioning only on discordant outcomes removes their
                        # frequency and can overweight rare task changes relative
                        # to the value-fitting terms.
                        # Neutral futures have zero rank gradient but remain
                        # in its normalization and in the value/difference fit.
                        loss += rank_weight*(F.binary_cross_entropy_with_logits(pair_logits[:, 1]-pair_logits[:, 0],
                            preferred, reduction='none')*informative*mass).sum()/mass.sum()
                    if decision_regret_weight:
                        # A decision competes against all measured alternatives
                        # under the same future, including doing nothing. Fit
                        # the expected loss of that choice, so neutral futures
                        # and harmful interventions retain their true frequency.
                        selected_groups=[]
                        for group_id in rng.integers(len(member_decision_groups),size=8):
                            group=member_decision_groups[group_id]
                            if len(group)>16:
                                reference=group[rows[group,1]==0]
                                alternatives=group[rows[group,1]!=0]
                                group=np.concatenate((reference,rng.choice(alternatives,
                                    size=16-len(reference),replace=False)))
                            selected_groups.append(group)
                        group_indices=torch.as_tensor(np.concatenate(selected_groups),device=device)
                        decision_values=(paired_gain(group_indices) if paired_event_difference else
                            predict(group_indices,condition=conditional_strata).sigmoid())
                        regrets=[];masses=[];offset=0
                        for group in selected_groups:
                            size=len(group);index=group_indices[offset:offset+size]
                            values=decision_values[offset:offset+size];labels=target[index]
                            probability=(values/.15).softmax(0)
                            regrets.append(labels.max()-(probability*labels).sum())
                            masses.append(importance[index[0]]*root_weight[root_ids[index[0]]])
                            offset+=size
                        mass=torch.stack(masses)
                        loss += decision_regret_weight*(torch.stack(regrets)*mass).sum()/mass.sum().clamp_min(1e-8)
                if conservative_catalog_weight:
                    selected_roots = torch.as_tensor(conservative_rng.choice(conservative_training_roots,
                        size=conservative_catalog_roots, replace=True), device=device)
                    sampled = torch.as_tensor(conservative_rng.integers(len(options),
                        size=(conservative_catalog_roots, conservative_catalog_samples)), device=device)
                    query = torch.cat((sampled, reference_indices[selected_roots]), -1)
                    query_roots = selected_roots[:, None].expand_as(query).flatten()
                    query_options = query.flatten(); query_windows = catalog_window_ids[query_options]
                    raw = net(context[query_roots], controls[query_roots], catalog_digits[query_options],
                        catalog_options[query_options, 1:],
                        tuple(item[query_roots, query_windows] for item in conservative_responses))
                    values = (net.event_gain(raw) if paired_event_difference else raw.sigmoid()).reshape(query.shape)
                    mask = reference_mask[selected_roots]
                    observed_mean = (values[:, conservative_catalog_samples:]*mask).sum(-1)/mask.sum(-1)
                    penalty = conservative_catalog_loss(values[:, :conservative_catalog_samples],
                        observed_mean, conservative_catalog_temperature)
                    mass = root_weight[selected_roots]
                    loss += conservative_catalog_weight*(penalty*mass).sum()/mass.sum().clamp_min(1e-8)
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
        paired_response_encoding=paired_response_encoding,
        aligned_response_contrast=aligned_response_contrast,
        paired_event_difference=paired_event_difference,
        empirical_difference_correction=empirical_difference_correction,
        response_conditioned_value=response_conditioned_value,
        physics_residual_response=physics_residual_response,
        shared_factual_response=shared_factual_response,
        shared_successor_value=shared_successor_value,
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
        isolate_factual_start=isolate_factual_start,
        feedback_relative_hold=feedback_relative_hold,
        neural_factual_calibration=np.asarray(factual_calibrations) if factual_crossfit else None,
        neural_factual_calibration_kind=factual_calibration_kind,
        factual_probability_calibration=factual_probability_calibration,
        factual_difference_transport=factual_difference_transport and not ordinary,
        factual_values=factual_population.cpu() if not ordinary else None,
        project_task_value=True, commit_factual=False,
        replan_delayed=False,
        continuation_contexts=None if ordinary or continuation_context is None else continuation_context.cpu(),
        continuation_controls=None if ordinary or continuation_controls is None else continuation_controls.cpu(),
        continuation_values=None if ordinary or continuation_target is None else continuation_target.cpu(),
        continuation_remaining=None if ordinary or continuation_remaining is None else continuation_remaining.cpu(),
        temporal_value_bounds=not ordinary and factual_bank is not None and value_target == 'success')
    lookup = {option: i for i, option in enumerate(options)}; local_fits = []
    if not ordinary and (empirical_difference_correction or confidence != 0. or stratified_local_calibration):
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
                effective_future_confidence=effective_future_confidence, value_target=value_target)
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
        response_conditioned_value=response_conditioned_value,
        physics_residual_response=physics_residual_response,
        shared_factual_response=shared_factual_response,
        shared_successor_value=shared_successor_value,
        options=options, defenders=n, decision_period=5, isolate_factual_start=isolate_factual_start,
        feedback_relative_hold=feedback_relative_hold) if ordinary else
        dict(kind='local_difference_joint', **recipe))
    removal = tuple(model.get('mechanism_ablation', (False, False, False)))+(False,)*3
    model['mechanism_ablation'] = (removal[0], False, removal[2])
    model.update(gate_period=1, physical_interventions=True, fixed_intervention_clock=True,
                 success_objective=value_target == 'success')
    model['successor_task_fit'] = dict(ordinary=ordinary, roots=len(roots),
        value_target=value_target, training_target_sha256=target_digest.hexdigest(),
        return_gamma=.995 if value_target == 'discounted_return' else None,
        return_tick_cost=.001 if value_target == 'discounted_return' else None,
        return_normalization='(discounted task return + 1.001) / 2; common root shaping offset omitted'
            if value_target == 'discounted_return' else None,
        isolate_factual_start=isolate_factual_start,
        feedback_relative_hold=feedback_relative_hold,
        intervention_start_rule=('replace only the subject command in the factual joint decision; '
            'peers respond after the next measured state' if isolate_factual_start else
            'reselect free peers under the held intervention at its activation state'),
        intervention_window_rule=('retain the subject displacement from each live factual joint choice'
            if feedback_relative_hold else 'hold the requested absolute subject coefficients'),
        root_sampling_fraction=root_sampling_fraction,
        factual_crossfit=factual_crossfit,
        refit_all_layouts=bool(refit_all_layouts),
        empirical_difference_correction=bool(empirical_difference_correction and not ordinary),
        training_layouts=(sorted({tuple(root['scene']) for root in roots} |
            (set() if factual_bank is None else {tuple(row['scene']) for row in factual_bank['records']}))
            if all('scene' in root for root in roots) and (factual_bank is None or
                all('scene' in row for row in factual_bank['records'])) else None),
        member_training_roots=([len(roots)]*ensemble_size if root_folds is None else
            [int(np.count_nonzero(root_folds != member)) for member in range(ensemble_size)]),
        member_factual_trajectories=([0 if factual_bank is None else len(factual_bank['records'])]*ensemble_size
            if bank_folds is None else [int(np.count_nonzero(bank_folds != member))
                for member in range(ensemble_size)]),
        prior_factual_crossfit=(None if initial_training_context is None else
            bool(initial_training_context.get('factual_crossfit', False))),
        released_factual_calibration=(
            'all development training layouts may now update every member; prior fold calibration is discarded'
            if refit_all_layouts else None),
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
        response_supervision_architecture=response_supervision_architecture if response_supervision_weight else None,
        response_contrast_weight=response_contrast_weight,
        measured_response_contrasts=len(response_pairs) if response_contrast_weight else 0,
        response_contrast_label_sha256=response_pair_sha256 if response_contrast_weight else None,
        response_contrast_scales=response_contrast_scales,
        response_contrast_scale_rule='training-fold weighted effect RMS with 0.05 floor' if response_contrast_weight else None,
        response_value_weight=response_value_weight,
        measured_response_value_pairs=len(response_pairs) if response_value_weight else 0,
        response_value_pair_sha256=response_pair_sha256 if response_value_weight else None,
        response_value_supervision='shared value evaluator on measured consequences, training only' if response_value_weight else None,
        response_value_difference_weight=anchor_weight if shared_successor_value and not ordinary and response_value_weight else 0.,
        factual_response_supervision=bool(factual_response_supervision),
        measured_factual_response_labels=0 if bank_response_ids is None else len(bank_response_ids),
        factual_response_label_sha256=bank_response_sha256,
        factual_response_horizon=20 if factual_response_supervision else None,
        factual_response_value_weight=response_value_weight if factual_response_supervision else 0.,
        response_conditioned_value=response_conditioned_value,
        physics_residual_response=physics_residual_response,
        shared_factual_response=shared_factual_response,
        shared_successor_value=shared_successor_value,
        successor_value_pooling='common mean, maximum and minimum physical successor features; public intervention clock retained'
            if shared_successor_value else None,
        response_effect_construction='shared learned factual residual plus public single-vessel physical effect'
            if shared_factual_response else None,
        physical_response_prior='finite public commands and constant observed target velocity; learned closed-loop residual'
            if physics_residual_response else None,
        physical_residual_initialization='zero' if physics_residual_response else None,
        response_heads_used_at_inference=response_conditioned_value,
        paired_response_encoding=paired_response_encoding,
        aligned_response_contrast=aligned_response_contrast,
        paired_event_difference=paired_event_difference,
        paired_event_objective=(('fractional neutral, positive and negative discounted-return mass on all common futures'
            if value_target == 'discounted_return' else 'joint neutral, rescue and harm likelihood on all common futures')
            if paired_event_difference else None),
        paired_event_contrast=(('direct unclipped discounted-return gains with factual value cancelled'
            if value_target == 'discounted_return' else 'direct unclipped event gains with factual value cancelled')
            if paired_event_difference else None),
        decision_regret_weight=decision_regret_weight,
        conservative_catalog_weight=conservative_catalog_weight,
        conservative_catalog_temperature=conservative_catalog_temperature,
        conservative_catalog_samples=conservative_catalog_samples if conservative_catalog_weight else 0,
        conservative_catalog_roots=conservative_catalog_roots if conservative_catalog_weight else 0,
        conservative_catalog_objective=('log-mean-exp of uniform deployment-catalog values minus the mean '
            'over unique measured choices; no synthetic return labels' if conservative_catalog_weight else None),
        conservative_catalog_reference=('uniform measured choices regardless of outcomes or future repetition count'
            if conservative_catalog_weight else None),
        measured_response_labels=0 if response_ids is None else len(response_ids),
        response_label_sha256=response_label_sha256,
        response_auxiliary_heads_at_inference=response_conditioned_value,
        training_parameters_per_member=training_parameter_count,
        member_updates=per_member, width=width, seed=seed, objective='absolute return' if ordinary else
        'whole-plan outcome likelihood plus paired single-vessel value differences and factual contrasts',
        local_reconciliation=bool(local_fits), inherited_stages=False, root_diagnostics=diagnostics,
        return_representation='predicted physical successor interaction network' if response_conditioned_value else
        'candidate-interaction network',
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
        initialization_training_context=copy.deepcopy(initial_training_context),
        cumulative_optimizer_steps=(updates if initial_states is None else
            initial_training_context['source_total_updates']+updates if initial_training_context is not None else None),
        frozen_parameter_calibration=updates == 0,
        effective_future_confidence=effective_future_confidence,
        additional_candidate_catalog_size=len(extra_options-{(0, 0, 0)}),
        inference_joint_value_reconstruction=local_contrast_head,
        receding_option_improvement=not ordinary,
        joint_factual_calibration=joint_factual_calibration and not ordinary,
        intervention_conditioned_locality=intervention_locality and not ordinary,
        factual_choice_keeps_future_decisions=True,
        delayed_choice_keeps_future_decisions=False,
        conditional_factual_strata=conditional_strata and not paired_event_difference,
        empirical_factual_reference=bool(factual_probability_calibration and not ordinary),
        development_future_weight=data.get('development_future_weight', 1.),
        candidate_catalog='all sampled complete programs, including negative outcomes',
        uncertainty_estimator=(('effective-future paired return standard error with neutral variance prior'
            if value_target == 'discounted_return' else 'separate paired rescue lower and harm upper Beta bounds')
            if not ordinary else None),
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
    if output is not None:
        output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); torch.save(model, output)
    return model


@torch.no_grad()
def raw_paired_value_diagnostics(model, data, *, require_unseen_layouts=False, device='cpu', batch_size=128):
    """Measure the neural effect itself, before any factual or local correction.

    Each candidate is compared with its factual branch on the same measured
    futures. Candidate selection uses predictions only. Layout-disjoint checks
    include the factual trajectory bank used to train the network.
    """
    fit = model['successor_task_fit']; recipe = model['successor_task_value']
    if (recipe['kind'] not in ('local_difference_joint', 'plain_joint_return', 'paired_mlp_return', 'plain_mlp_return')
            or fit.get('predecessor_sha256') != data.get('predecessor_sha256') or batch_size < 1):
        raise ValueError('Raw paired diagnostics require a matched neural estimator and factual predecessor.')
    dense = recipe['kind'] in ('paired_mlp_return', 'plain_mlp_return')
    paired = recipe['kind'] == 'paired_mlp_return' or bool(recipe.get('paired_response_encoding', False))
    paired_events = recipe['kind'] == 'paired_mlp_return' or bool(recipe.get('paired_event_difference', False))
    if recipe['kind'] == 'local_difference_joint' and not paired:
        raise ValueError('Raw paired diagnostics require a paired response head.')
    roots = [observed_programs(root) for root in data['roots']]
    if not roots:
        raise ValueError('Raw paired diagnostics require measured candidate pairs.')
    known_layouts = fit.get('training_layouts')
    query_layouts = ({tuple(root['scene']) for root in roots}
                     if all('scene' in root for root in roots) else None)
    overlap = (None if known_layouts is None or query_layouts is None else
               query_layouts & {tuple(scene) for scene in known_layouts})
    if require_unseen_layouts and (overlap is None or overlap):
        raise ValueError('Raw validation requires layouts absent from all gradient and factual-trajectory training data.')
    fields = ('peer_conditioned', 'context_conditioned', 'motion_basis', 'finite_motion_basis',
              'dual_motion_basis', 'identity_conditioned', 'response_conditioned_value',
              'physics_residual_response', 'shared_factual_response', 'shared_successor_value')
    network = {key: recipe[key] for key in fields if key in recipe}
    network.update(output_width=3 if paired else 1, paired_response_encoding=paired,
        aligned_response_contrast=recipe.get('aligned_response_contrast', False), paired_event_difference=paired_events)
    models = []
    states = [recipe['state']] if dense else recipe['states']
    for state in states:
        if dense:
            from plain_return import PlainReturnNetwork, PairedReturnNetwork
            factory = PairedReturnNetwork if paired else PlainReturnNetwork
            net = factory(recipe['defenders'], recipe['width'], recipe.get('identity_conditioned', False),
                action_encoding=recipe.get('action_encoding', 'categorical'))
        else:
            net = JointTaskPotential(recipe['width'], **network)
        net = net.to(device).eval()
        net.load_state_dict(state); models.append(net)
    contexts = torch.as_tensor(np.asarray([root['context'] for root in roots]), dtype=torch.float32, device=device)
    controls = torch.as_tensor(np.asarray([root['effect_controls'] for root in roots]), dtype=torch.float32, device=device)
    samples = []; truth = []; group_sizes = []; paired_futures = 0
    observation_digest = hashlib.sha256(); target_digest = hashlib.sha256()
    for root_id, root in enumerate(roots):
        values = measured_root_values(root, fit.get('value_target', 'success'))
        weights = np.asarray(root.get('future_weights', np.ones(len(root['futures']))), np.float64)
        if (weights.shape != (len(root['futures']),) or not np.isfinite(weights).all()
                or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError('Raw diagnostics require finite common-future population weights.')
        group_size = 0
        for option_id, option in enumerate(root['options']):
            if option_id == 0:
                continue
            futures = [future for future in range(len(root['futures']))
                       if (0, future) in values and (option_id, future) in values]
            if not futures or weights[futures].sum() <= 0:
                continue
            delta = np.asarray([values[option_id, future]-values[0, future] for future in futures])
            truth.append(float(np.average(delta, weights=weights[futures])))
            samples.append((root_id, *option)); group_size += 1; paired_futures += len(futures)
            target_digest.update(np.asarray(delta, np.float32).tobytes())
            target_digest.update(weights[futures].tobytes())
        if not group_size:
            raise ValueError('Every diagnostic root must have a measured intervention and its factual pair.')
        group_sizes.append(group_size)
        for field in ('context', 'effect_controls', 'options', 'futures'):
            observation_digest.update(np.ascontiguousarray(root[field]).tobytes())
    sample = torch.as_tensor(samples, dtype=torch.long, device=device)
    predictions = []
    for part in sample.split(batch_size):
        root_id = part[:, 0]; programs = (part[:, 1:2]//(10**torch.arange(controls.shape[1], device=device))) % 10
        estimates = []
        for net in models:
            raw = net(contexts[root_id], controls[root_id], programs, part[:, 2:])
            if paired:
                if dense:
                    gain = net.signed_gain(raw, (programs > 0).any(-1))
                elif paired_events:
                    gain = net.event_gain(raw)
                else:
                    factual, rescue, harm = raw.sigmoid().unbind(-1)
                    gain = (1-factual)*rescue-factual*harm
            else:
                factual = net(contexts[root_id], controls[root_id], torch.zeros_like(programs), torch.zeros_like(part[:, 2:]))
                gain = raw.sigmoid()-factual.sigmoid()
            estimates.append(gain)
        predictions.append(torch.stack(estimates).mean(0).cpu().numpy())
    prediction = np.concatenate(predictions); truth = np.asarray(truth, np.float64)
    error = prediction-truth
    weights = np.concatenate([np.full(size, 1./size/len(roots)) for size in group_sizes])
    informative = np.abs(truth) > .02
    selected_true = []; selected_predicted = []; regret = []; offset = 0
    for size in group_sizes:
        predicted = np.r_[0., prediction[offset:offset+size]]
        observed = np.r_[0., truth[offset:offset+size]]
        choice = int(predicted.argmax())
        selected_true.append(float(observed[choice])); selected_predicted.append(float(predicted[choice]))
        regret.append(float(observed.max()-observed[choice])); offset += size
    return dict(value_target=fit.get('value_target', 'success'), roots=len(roots), candidate_pairs=len(truth),
        common_futures=paired_futures, empirical_corrections_used=False, factual_calibration_used=False,
        evaluated_parameter_sha256=_joint_parameter_digest(states),
        observation_sha256=observation_digest.hexdigest(), target_sha256=target_digest.hexdigest(),
        layout_overlap=None if overlap is None else len(overlap),
        layouts_disjoint=overlap is not None and not overlap,
        mean_absolute_error=float(weights@np.abs(error)), rmse=float(np.sqrt(weights@error**2)),
        bias=float(weights@error), informative_pairs=int(informative.sum()),
        sign_accuracy=(float(np.average((prediction[informative] > 0) == (truth[informative] > 0),
            weights=weights[informative])) if informative.any() else None),
        selection='neural argmax over measured alternatives including unchanged continuation',
        selected_predicted_gain=float(np.mean(selected_predicted)), selected_measured_gain=float(np.mean(selected_true)),
        selected_harmful_roots=int(np.count_nonzero(np.asarray(selected_true) < -1e-8)),
        measured_candidate_regret=float(np.mean(regret)))


def fit_success_intervention_pair(data, paired_output, ordinary_output, *, validation_data,
        factual_bank=None, updates=24000, ensemble_size=3, width=64, seed=42, device='cpu',
        network_options=None, margin=.02, architecture='interaction', parameter_budget=None,
        conditional_objective='marginal_events'):
    """Train both arms on every training layout, then audit untouched layouts.

    The event likelihood and squared paired differences are probability-valued
    objectives. Ranking and softmax decision losses are excluded because they
    can improve ordering by changing the magnitude of the reported effect.
    """
    if updates < ensemble_size or updates % ensemble_size:
        raise ValueError('Matched success training requires equal positive optimizer budgets for all members.')
    if architecture not in ('interaction', 'mlp') or (architecture == 'mlp' and ensemble_size != 1):
        raise ValueError('Choose an interaction ensemble or one matched dense model per arm.')
    if (conditional_objective not in ('marginal_events', 'factual_strata')
            or architecture == 'mlp' and conditional_objective != 'marginal_events'):
        raise ValueError('Factual-stratum success learning requires the paired interaction network.')
    if (not data.get('roots') or not validation_data.get('roots')
            or data.get('predecessor_sha256') != validation_data.get('predecessor_sha256')):
        raise ValueError('Matched success training needs measured training and validation data from the same predecessor.')
    for key in ('isolate_factual_start', 'feedback_relative_hold'):
        if bool(data.get(key, False)) != bool(validation_data.get(key, False)):
            raise ValueError('Training and validation must use identical intervention execution semantics.')
    training_records = data['roots']+(factual_bank['records'] if factual_bank is not None else [])
    if not all('scene' in root for root in training_records+validation_data['roots']):
        raise ValueError('Matched success training requires recorded layouts for both populations.')
    training_layouts = {tuple(root['scene']) for root in training_records}
    validation_layouts = {tuple(root['scene']) for root in validation_data['roots']}
    if training_layouts & validation_layouts:
        raise ValueError('Validation layouts must be absent from paired and factual training data.')
    options = dict(network_options or {})
    allowed = {'peer_conditioned', 'context_conditioned', 'motion_basis', 'finite_motion_basis',
        'dual_motion_basis', 'identity_conditioned', 'response_conditioned_value', 'physics_residual_response',
        'shared_factual_response', 'response_supervision_weight', 'response_supervision_architecture',
        'response_value_weight', 'factual_response_supervision', 'learning_rate', 'factual_bank_weight',
        'additional_candidate_options', 'root_sampling_fraction', 'edge_weight', 'anchor_weight', 'action_encoding',
        'snapshot_updates'}
    if set(options)-allowed:
        raise ValueError('Matched success settings cannot override targets, coverage, calibration or optimizer budgets.')
    common = dict(updates=updates, ensemble_size=ensemble_size, seed=seed, device=device,
        motion_basis=True, finite_motion_basis=True, context_conditioned=True, peer_conditioned=True,
        ordinary_architecture='interaction', value_target='success', factual_crossfit=False,
        empirical_difference_correction=False, joint_factual_calibration=False, factual_probability_calibration=False,
        rank_weight=0., decision_regret_weight=0., root_sampling_fraction=.5, margin=margin,
        confidence=0., global_penalty=0.)
    common.update(options)
    if not common['motion_basis']:
        raise ValueError('Paired success learning requires a public motion representation.')
    network_fields = ('peer_conditioned', 'context_conditioned', 'motion_basis', 'finite_motion_basis',
                      'dual_motion_basis', 'identity_conditioned', 'response_conditioned_value',
                      'physics_residual_response', 'shared_factual_response')
    network_architecture = {key: common[key] for key in network_fields if key in common}
    def capacity(candidate_width, paired):
        net = JointTaskPotential(candidate_width, output_width=3 if paired else 1,
            paired_response_encoding=paired, aligned_response_contrast=paired,
            paired_event_difference=paired and conditional_objective == 'marginal_events', **network_architecture)
        return sum(parameter.numel() for parameter in net.parameters())
    paired_capacity = capacity(width, True)
    ordinary_width = min(range(max(1, width//2), 2*width+1),
        key=lambda candidate: abs(capacity(candidate, False)-paired_capacity))
    conditional = dict(conditional_strata=True, paired_response_encoding=True,
        aligned_response_contrast=True, paired_event_difference=conditional_objective == 'marginal_events',
        response_contrast_weight=(1. if common.get('response_conditioned_value') and
                                 not common.get('shared_factual_response') else 0.))
    if architecture == 'interaction':
        if 'action_encoding' in options:
            raise ValueError('Action encoding is a dense-network option.')
        if parameter_budget is not None:
            raise ValueError('Interaction capacity is controlled by width and ensemble size.')
        paired = fit_joint_potential(data, None, width=width, factual_bank=factual_bank, **common, **conditional)
        ordinary = fit_joint_potential(data, None, width=ordinary_width, factual_bank=factual_bank, ordinary=True, **common)
        paired['successor_task_value'].update(temporal_value_bounds=False, intervention_budget=1,
            project_task_value=False, factual_bound_pruning=False, candidate_bound_pruning=False)
        ordinary['successor_task_value'].update(intervention_budget=1, gain_margin=margin)
        ordinary_capacity = capacity(ordinary_width, False)
    else:
        from plain_return import fit_plain_return, fit_paired_mlp_return
        dense_options = {'identity_conditioned', 'learning_rate', 'factual_bank_weight', 'action_encoding',
                         'additional_candidate_options', 'root_sampling_fraction', 'snapshot_updates'}
        if set(options)-dense_options:
            raise ValueError('Dense success learning uses public state/action inputs without response auxiliary heads.')
        settings = dict(factual_bank=factual_bank, updates=updates,
            parameter_budget=paired_capacity if parameter_budget is None else parameter_budget,
            seed=seed, device=device, value_target='success', intervention_budget=1,
            gain_margin=margin, root_sampling_fraction=.5)
        settings.update(options)
        paired = fit_paired_mlp_return(data, None, **settings)
        ordinary = fit_plain_return(data, None, **settings,
            expected_observation_sha256=paired['successor_task_fit']['training_observation_sha256'],
            expected_training_target_sha256=paired['successor_task_fit']['training_target_sha256'])
        paired_capacity = paired['successor_task_fit']['trainable_parameters']
        ordinary_capacity = ordinary['successor_task_fit']['trainable_parameters']
        for model in (paired, ordinary):
            model['successor_task_fit'].update(training_layouts=sorted(training_layouts),
                member_training_roots=[len(data['roots'])],
                member_factual_trajectories=[0 if factual_bank is None else len(factual_bank['records'])],
                factual_crossfit=False, empirical_difference_correction=False,
                empirical_factual_reference=False)
    matched_fields = ('training_observation_sha256', 'training_target_sha256', 'member_training_roots',
        'member_factual_trajectories', 'member_updates', 'updates', 'predecessor_sha256')
    if any(paired['successor_task_fit'][key] != ordinary['successor_task_fit'][key] for key in matched_fields):
        raise RuntimeError('Matched success training produced unequal data coverage or optimizer budgets.')
    for model in (paired, ordinary):
        fit = model['successor_task_fit']
        fit['success_probability_objective'] = True
        fit['validation_layouts'] = sorted(validation_layouts)
        fit['rank_weight'] = 0.; fit['decision_regret_weight'] = 0.
        fit['raw_training_diagnostics'] = raw_paired_value_diagnostics(model, data, device=device)
        fit['raw_validation_diagnostics'] = raw_paired_value_diagnostics(model, validation_data,
            require_unseen_layouts=True, device=device)
        fit['matched_success_training'] = dict(all_members_see_all_training_layouts=True,
            training_layouts=len(training_layouts), validation_layouts=len(validation_layouts),
            validation_used_for_gradients=False, validation_used_for_calibration=False,
            paired_parameters_per_member=paired_capacity,
            ordinary_parameters_per_member=ordinary_capacity,
            probability_difference_units='success probability', time_reward_weight=0.)
    for model, output in ((paired, paired_output), (ordinary, ordinary_output)):
        if output is not None:
            output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); torch.save(model, output)
    return paired, ordinary


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
    if recipe.get('paired_event_difference'):
        raise ValueError('Paired event differences retain their marginal paired-outcome calibration.')
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
            effective_future_confidence=effective_future_confidence,
            value_target=result['successor_task_fit'].get('value_target', 'success'))
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
            lower = (np.asarray(detail['gain_lower']) if 'gain_lower' in detail else
                (1-detail['factual_value'])*np.asarray(detail['repair_lower'])
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
                                raw = ((factual+raw[:, 1]-raw[:, 2]).clamp(0., 1.)
                                       if value.paired_event_difference else
                                       (1-factual)*raw[:, 1]+factual*(1-raw[:, 2]))
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


class FactualProposalRootSelector:
    """Find the first native intervention proposal without applying it to the facts."""
    def __init__(self, learners, model_sha256, *, trigger_estimators=None):
        from plain_return import PlainReturnDecision
        self.learners = list(learners)
        hashes = list(model_sha256)
        if (not self.learners or len(hashes) != len(self.learners) or
                any(not isinstance(value, str) or len(value) != 64 or
                    any(character not in '0123456789abcdef' for character in value) for value in hashes)):
            raise ValueError('Proposal roots require their actual frozen estimator hashes.')
        triggers = tuple(range(len(self.learners))) if trigger_estimators is None else tuple(trigger_estimators)
        if (not triggers or len(set(triggers)) != len(triggers) or
                any(type(index) is not int or not 0 <= index < len(self.learners) for index in triggers)):
            raise ValueError('Proposal triggers must name a nonempty subset of the observed estimators.')
        self.trigger_estimators = triggers
        for learner in self.learners:
            if (not isinstance(learner, (LocalDifferenceDecision, PlainJointReturnDecision, PlainReturnDecision)) or
                    learner.intervention_budget != 1 or learner.decision_period != 5 or
                    not learner.isolate_factual_start or getattr(learner, 'feedback_relative_hold', False) or
                    getattr(learner, 'fleet_transfer', None) is not None):
                raise ValueError('Proposal roots require the matched native single-vessel decision rules.')
            if not torch.equal(learner.options.cpu(), self.learners[0].options.cpu()):
                raise ValueError('The proposal estimators must share the complete candidate catalog.')
            learner.eval()
        self.specification = dict(kind='first_accepted_proposal_else_uniform',
            proposal_model_sha256=hashes, decision_period=5,
            selection_inputs='current public state and factual commands',
            acceptance='native decision including confidence, margin and any waiting value',
            fallback='uniform eligible factual state when every proposal is declined',
            common_for_all_estimators=True)
        if set(triggers) != set(range(len(self.learners))):
            self.specification['trigger_estimators'] = sorted(triggers)
        self.last_observations = None

    @torch.no_grad()
    def __call__(self, packet, actor):
        from conditional_fusion_value import (intervention_task_context, paired_effect_controls,
                                               append_vessel_identity)
        from plain_return import PlainReturnDecision
        if actor.successor_task_value is not None:
            raise ValueError('Proposal sampling must follow the unchanged factual predecessor.')
        action = actor(packet, deterministic=True, gate_only=True)[0]
        context, order = intervention_task_context(packet, True, True)
        native_controls = paired_effect_controls(actor, packet, action)
        controls = native_controls.gather(1, order[..., None].expand(-1, -1, 6))
        accepted = torch.zeros(len(context), dtype=torch.bool, device=action.device)
        native_options = []
        for estimator_index, learner in enumerate(self.learners):
            if isinstance(learner, PlainReturnDecision):
                proposed = learner(dict(packet, _plain_return_controls=native_controls), action)
                coefficients = learner.coefficients.to(action.device)
            else:
                public = (append_vessel_identity(context, order)
                    if getattr(learner.models[0], 'identity_conditioned', False) else context)
                proposed = learner.forward_public(public, controls, order, action, learner.eligible(packet))
                coefficients = learner.models[0].coefficients.to(action.device)
            option = torch.zeros((len(context), 3), dtype=torch.long, device=action.device)
            if proposed.shape[-1] == 14:
                changed = (proposed[..., 9:13] >= 0.).any(-1)
                if bool((changed.sum(-1) > 1).any()):
                    raise ValueError('A proposal root cannot be selected by a multi-vessel intervention.')
                active = changed.any(-1)
                preferences = proposed[..., 11:13].gather(1, order[..., None].expand(-1, -1, 2))
                matches = (preferences[:, :, None] == coefficients[None, None]).all(-1)
                if bool((active[:, None] & (matches.sum(-1) != 1)).any()):
                    raise ValueError('A native intervention must use the shared fusion alphabet.')
                digits = matches.to(torch.long).argmax(-1)
                option[:, 0] = (digits*10**torch.arange(order.shape[1], device=action.device)).sum(-1)
                option[:, 1] = proposed[:, 0, 8].round().to(torch.long)
                option[:, 2] = (proposed[:, 0, 4]-proposed[:, 0, 8]).round().to(torch.long)
                option[~active] = 0
                if estimator_index in self.trigger_estimators:
                    accepted |= active
            elif proposed.shape != action.shape:
                raise ValueError('Proposal sampling received an unknown native decision program.')
            native_options.append(option)
        self.last_observations = {key:value.detach().cpu().numpy().copy() for key,value in dict(
            context=context, controls=controls, order=order, options=torch.stack(native_options, 1)).items()}
        return accepted


def factual_trajectory_batch(tasks, *, device='cpu', root_sample_seed=None, root_selector=None):
    """Measure factual trajectories with batched copies of the same controller."""
    import conditional_fusion_value as c
    import develop_source_fusion as d
    import run_remote_evidence as r
    policy = Path(tasks[0][0])
    actor = d.joint_load(str(policy), policy.stat().st_mtime_ns)
    if actor.successor_task_value is not None:
        raise ValueError('Factual sampling requires the frozen predecessor.')
    jobs = None
    if root_sample_seed is None:
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
        batch_size=len(tasks), device=device, keep_steps=True, initial_jobs=jobs,
        root_sample_seed=root_sample_seed, root_selector=root_selector)
    records = []
    actor = d.joint_load(str(policy), policy.stat().st_mtime_ns).to(device)
    try:
        for task, trajectory in zip(tasks, measured):
            _, scene, future, expected, period = task
            row = trajectory['row']; selected = trajectory['steps'][::period]
            if expected is not None and row['success'] != expected:
                if root_sample_seed is not None:
                    raise ValueError('A sampled factual root must retain its actual batched trajectory.')
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
            if root_sample_seed is not None:
                records[-1]['root_snapshot']=dict(trajectory['root_snapshot'],
                    predecessor_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
                    inference_batch=len(tasks),inference_device=device,root_sample_seed=int(root_sample_seed))
    finally:
        actor.cpu()
    return records


def collect_onpolicy_factual_roots(policy, tasks, output, *, root_sample_seed,
                                  period=5, inference_batch=32, inference_device='cuda:0', root_selector=None):
    """Sample one eligible decision state from each fresh factual episode.

    Reservoir sampling follows the deployed batched controller without
    changing its random stream. Each retained snapshot is a real predecessor
    state; subsequent paired continuations use fresh common futures. The
    terminal factual labels also supervise every recorded state in the same
    episode, with no additional environment observations.
    """
    import run_remote_evidence as r
    if period < 1 or inference_batch < 1 or int(root_sample_seed) < 0:
        raise ValueError('Factual root sampling needs a positive period and batch, and a nonnegative seed.')
    if root_selector is not None and (period != 5 or
            not isinstance(root_selector, FactualProposalRootSelector)):
        raise ValueError('Directed factual roots require a native proposal selector at the deployed decision period.')
    policy=Path(policy);output=Path(output)
    digest=hashlib.sha256(policy.read_bytes()).hexdigest()
    tasks=[(tuple(scene),future) for scene,future in tasks]
    if len(set(tasks))!=len(tasks) or not tasks:
        raise ValueError('Fresh factual root tasks must be nonempty and unique.')
    specification=dict(predecessor_sha256=digest,tasks=tasks,period=period,
        root_sample_seed=int(root_sample_seed),inference_batch=inference_batch,
        inference_device=inference_device,
        root_sampling='uniform eligible factual decision state; separate reservoir random stream')
    if root_selector is not None:
        specification.update(root_sampling='first accepted native proposal; uniform fallback; unchanged factual trajectory',
            root_selection=copy.deepcopy(root_selector.specification))
    bank=(torch.load(output,map_location='cpu',weights_only=False) if output.exists() else
        dict(records=[],simulation_steps=0,trajectories=0,state_labels=0,completed=False,
            **specification))
    if any(bank.get(key)!=value for key,value in specification.items()):
        raise ValueError('Saved factual roots use a different predecessor, tasks, sampler or inference batch.')
    records=bank['records']
    if [(tuple(row['scene']),row['future']) for row in records]!=tasks[:len(records)]:
        raise ValueError('Saved factual roots must preserve the ordered training task prefix.')
    if len(records)>len(tasks) or (len(records)%inference_batch and len(records)!=len(tasks)):
        raise ValueError('Saved factual roots must end at an unchanged inference batch boundary.')
    if bank['completed']:
        if len(records)!=len(tasks):
            raise ValueError('A completed factual bank must contain every requested episode.')
        return bank
    for begin in range(len(records),len(tasks),inference_batch):
        group=[(str(policy),scene,future,None,period) for scene,future in tasks[begin:begin+inference_batch]]
        measured=factual_trajectory_batch(group,device=inference_device,root_sample_seed=root_sample_seed,
            root_selector=root_selector)
        for row in measured:
            snapshot=row['root_snapshot']
            if snapshot['predecessor_sha256']!=digest or snapshot['root_steps']>=row['physical_steps']:
                raise ValueError('A sampled root must precede the factual terminal state under the same controller.')
        records.extend(measured)
        bank.update(simulation_steps=sum(row['physical_steps'] for row in records),
            trajectories=len(records),state_labels=sum(len(row['values']) for row in records),
            completed=len(records)==len(tasks))
        r.atomic_save(bank,output)
        print(dict(phase='onpolicy_factual_roots',trajectories=len(records),total=len(tasks),
            state_labels=bank['state_labels'],physical_steps=bank['simulation_steps'],
            noninitial_roots=sum(row['root_snapshot']['root_steps']>0 for row in records)),flush=True)
    return bank


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


def source_successor_batches(tasks, *, batch_size=128, device='cpu', physical_response_horizon=0):
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
            response_tick=[tick for _, _, _, _, tick in group],
            physical_response_tick=physical_response_horizon or None)
        for (index, _, _, _, _), row in zip(group, rows):
            result[index] = row
    return result


def breadth_local_design(blueprint, candidate_options, *, pairs_per_root=6, repetitions=2, seed=0,
                         vessel_balanced=False):
    """Allocate fresh paired queries to every root without inspecting labels.

    Each queried edge changes one vessel at a common intervention window.
    Stratifying edges by plan size and window spreads the bounded root budget
    over the deployment catalog, including roots previously sampled only
    factually. The vessel-balanced allocation samples two alternatives for
    each queried vessel at a shared window, anchored to the factual program.
    The returned outcomes contain scheduling bounds, never labels.
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
            if vessel_balanced:
                groups = {boat:{} for boat in range(n)}
                for index, (code, delay, span) in enumerate(options[1:], 1):
                    changed = [boat for boat in range(n) if code//10**boat % 10]
                    if len(changed) != 1:
                        raise ValueError('Vessel-balanced evidence requires factual single-vessel alternatives.')
                    groups[changed[0]].setdefault((delay, span), []).append(index)
                if any(not windows or any(len(indices)<2 for indices in windows.values())
                       for windows in groups.values()):
                    raise ValueError('Every vessel and intervention window needs two measured alternatives.')
                by_fleet[n] = [[windows[key] for key in sorted(windows)] for windows in groups.values()]
            else:
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
            if vessel_balanced:
                window = group[int(rng.integers(len(group)))]
                selected.update(map(int, rng.choice(window, size=2, replace=False)))
            else:
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
        if 'root_snapshot' in original:
            root['root_snapshot']=original['root_snapshot']
        roots.append(root)
        signatures.append((scene, root['prefix_future'], root['root_steps'], root['lookback'],
                           tuple(chosen), tuple(futures)))
    design = dict(kind='breadth_local_pairs', seed=int(seed), pairs_per_root=int(pairs_per_root),
        repetitions=int(repetitions), roots=len(roots), maximum_observations_per_root=
        (1+2*pairs_per_root)*repetitions, planned_observations=sum(len(x['outcomes']) for x in roots),
        catalog_sha256=hashlib.sha256(repr(options).encode()).hexdigest(),
        design_sha256=hashlib.sha256(repr(signatures).encode()).hexdigest(),
        inherited_labels=False, allocation='uniform plan-size/window strata; fresh paired futures')
    if vessel_balanced:
        design.update(vessel_balanced=True,
            allocation='balanced physical vessels; two single-vessel alternatives at a shared uniformly sampled window; common factual anchor')
    return dict(roots=roots, acquisition_design=design)


def confirmation_local_design(blueprint, *, source_data_sha256, candidates=3,
                              repetitions=16, seed=0, value_target='discounted_return'):
    """Confirm measured candidate gains on fresh common futures at every root.

    The original observations remain a separate, immutable acquisition stage.
    Even roots whose best measured gain is negative receive confirmation.
    This is adaptive training acquisition, not an independent policy test;
    conditional and ordinary learners receive the same additional outcomes.
    """
    if (not blueprint.get('completed') or not blueprint.get('roots') or
            any(not isinstance(value, int) or value < 1 for value in (candidates, repetitions)) or
            not isinstance(seed, int) or seed < 0):
        raise ValueError('Confirmation requires complete observations and positive allocation sizes.')
    if (not isinstance(source_data_sha256, str) or len(source_data_sha256) != 64 or
            any(character not in '0123456789abcdef' for character in source_data_sha256)):
        raise ValueError('Confirmation must identify its actual source observations by SHA256.')
    roots = []; signatures = []; allocations = []
    for root_id, original in enumerate(blueprint['roots']):
        options = list(map(tuple, original['options']))
        scene = tuple(original['scene']); n = int(scene[1])
        if (options[0] != (0, 0, 0) or len(options) <= candidates or
                len(set(options)) != len(options) or 'root_snapshot' not in original or
                any(code <= 0 or code >= 10**n or delay < 0 or span < 1 or
                    sum(code//10**boat % 10 != 0 for boat in range(n)) != 1
                    for code, delay, span in options[1:])):
            raise ValueError('Confirmation needs recorded factual roots and single-vessel alternatives.')
        weights = np.asarray(original.get('future_weights', np.ones(len(original['futures']))), np.float64)
        if (weights.shape != (len(original['futures']),) or not np.isfinite(weights).all() or
                (weights < 0).any() or len(set(original['futures'])) != len(original['futures'])):
            raise ValueError('Confirmation requires distinct measured futures with valid population weights.')
        measured = measured_root_values(original, value_target)
        ranking = []
        for q in range(1, len(options)):
            common = [f for f in range(len(weights))
                      if weights[f] > 0 and (q, f) in measured and (0, f) in measured]
            if len(common) < 2:
                raise ValueError('Each confirmed alternative needs at least two actual paired futures.')
            difference = np.asarray([measured[q, f]-measured[0, f] for f in common])
            if not np.isfinite(difference).all():
                raise ValueError('Confirmation cannot rank nonfinite measured gains.')
            ranking.append((q, float(np.average(difference, weights=weights[common])), len(common)))
        ranking.sort(key=lambda item: (-item[1], item[0]))
        selected = [0]+sorted(item[0] for item in ranking[:candidates])
        chosen = [options[q] for q in selected]
        rng = np.random.default_rng(np.random.SeedSequence([seed, root_id, 418903]))
        existing = set(original['futures']); futures = []
        while len(futures) < repetitions:
            future = int(rng.integers(1, 2**32-1))
            if future not in existing and future not in futures:
                futures.append(future)
        roots.append(dict(scene=scene, prefix_future=original['prefix_future'],
            root_steps=int(original['root_steps']), lookback=original.get('lookback'),
            options=chosen, futures=futures, future_weights=np.ones(repetitions, np.float32),
            outcomes={(q, f): dict(duration=400) for q in range(len(chosen)) for f in range(repetitions)},
            root_snapshot=original['root_snapshot'], blueprint_root=root_id))
        allocations.append(dict(source_root=root_id, selected_source_options=selected[1:],
            measured_gain_ranking=ranking, source_futures=len(original['futures'])))
        signatures.append((scene, original['prefix_future'], int(original['root_steps']),
            tuple(chosen), tuple(futures), tuple(ranking)))
    design = dict(kind='paired_effect_confirmation', seed=seed, roots=len(roots),
        source_data_sha256=source_data_sha256, value_target=value_target,
        selected_alternatives=candidates, repetitions=repetitions,
        maximum_observations_per_root=(candidates+1)*repetitions,
        planned_observations=sum(len(root['outcomes']) for root in roots),
        inherited_labels=False, same_observations_for_ordinary=True,
        root_outcome_filtering=False, adaptive_training_acquisition=True,
        allocation='top measured conditional gains at every root; fresh common confirmation futures',
        selected_source_observations=allocations,
        design_sha256=hashlib.sha256(repr(signatures).encode()).hexdigest())
    return dict(roots=roots, acquisition_design=design)


def policy_directed_local_design(blueprint, candidate_options, proposal_scores, *,
                                 proposal_model_sha256, queries_per_model=2,
                                 uniform_queries=2, repetitions=4, seed=0):
    """Measure the common union of estimator proposals and fresh exploration.

    Scores must be predictions at the supplied factual states, in the exact
    supplied catalog order. Every estimator subsequently receives every
    measured outcome, including the other estimator's proposed interventions.
    No factual or intervention outcome participates in query selection.
    """
    options = list(map(tuple, candidate_options))
    scores = np.asarray(proposal_scores, np.float64)
    sources = tuple(proposal_model_sha256)
    originals = blueprint['roots']
    if (not originals or not options or options[0] != (0, 0, 0) or
            len(set(options)) != len(options) or scores.ndim != 3 or
            scores.shape[1:] != (len(originals), len(options)) or not scores.shape[0] or
            not np.isfinite(scores).all()):
        raise ValueError('Common proposals require finite model-by-root-by-catalog scores and a factual anchor.')
    if (len(sources) != scores.shape[0] or any(not isinstance(value, str) or len(value) != 64 or
            any(character not in '0123456789abcdef' for character in value) for value in sources)):
        raise ValueError('Each proposal estimator needs its actual checkpoint SHA256.')
    if any(not isinstance(value, int) or value < 1
           for value in (queries_per_model, uniform_queries, repetitions)):
        raise ValueError('Use positive proposal, exploration and common-future counts.')
    alternative_count = scores.shape[0]*queries_per_model+uniform_queries
    if alternative_count >= len(options):
        raise ValueError('The catalog must support the complete common query budget.')
    for n in {int(root['scene'][1]) for root in originals}:
        if any(code <= 0 or code >= 10**n or delay < 0 or span < 1 or
               sum(code//10**boat % 10 != 0 for boat in range(n)) != 1
               for code, delay, span in options[1:]):
            raise ValueError('Directed queries require finite single-vessel alternatives for every root fleet.')
    roots = []; signatures = []; proposal_overlap = 0
    for root_id, original in enumerate(originals):
        selected = set()
        for model_scores in scores[:, root_id]:
            ranked = np.argsort(-model_scores[1:], kind='stable')[:queries_per_model]+1
            selected.update(map(int, ranked))
        proposal_overlap += scores.shape[0]*queries_per_model-len(selected)
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), root_id, 418901]))
        remaining = np.asarray(sorted(set(range(1, len(options)))-selected))
        selected.update(map(int, rng.choice(remaining, size=alternative_count-len(selected), replace=False)))
        chosen = [options[index] for index in [0]+sorted(selected)]
        future_rng = np.random.default_rng(np.random.SeedSequence([int(seed), root_id, 418902]))
        futures = []
        while len(futures) < repetitions:
            future = int(future_rng.integers(1, 2**32-1))
            if future not in futures:
                futures.append(future)
        root = dict(scene=tuple(original['scene']), prefix_future=original['prefix_future'],
            root_steps=int(original['root_steps']), lookback=original.get('lookback'),
            options=chosen, futures=futures, future_weights=np.ones(repetitions, np.float32),
            outcomes={(q, f):dict(duration=400) for q in range(len(chosen))
                      for f in range(repetitions)}, blueprint_root=root_id)
        if 'root_snapshot' in original:
            root['root_snapshot'] = original['root_snapshot']
        roots.append(root)
        signatures.append((root['scene'], root['prefix_future'], root['root_steps'],
            root['lookback'], tuple(chosen), tuple(futures)))
    design = dict(kind='breadth_local_pairs', seed=int(seed), roots=len(roots),
        repetitions=repetitions, proposal_models=len(sources), proposal_model_sha256=sources,
        queries_per_model=queries_per_model, minimum_uniform_queries=uniform_queries,
        overlapping_proposals=proposal_overlap, alternatives_per_root=alternative_count,
        maximum_observations_per_root=(1+alternative_count)*repetitions,
        planned_observations=sum(len(root['outcomes']) for root in roots),
        catalog_sha256=hashlib.sha256(repr(options).encode()).hexdigest(),
        proposal_score_sha256=hashlib.sha256(scores.tobytes()).hexdigest(),
        design_sha256=hashlib.sha256(repr((signatures, sources)).encode()).hexdigest(),
        inherited_labels=False, shared_by_all_estimators=True,
        allocation='union of each estimator highest-score alternatives; uniformly fill common budget; fresh paired futures')
    return dict(roots=roots, acquisition_design=design)


def _recollect_breadth_root(task):
    import tempfile
    policy, root_id, root, inference_batch, inference_device = task[:5]
    physical_response_horizon = task[5] if len(task) > 5 else 0
    isolate_factual_start = task[6] if len(task) > 6 else False
    # Keep only the combined acquisition on disk. The single-root collector
    # uses its existing bounded replay path and removes its temporary state.
    with tempfile.TemporaryDirectory(prefix='arboids-local-root-') as temporary:
        value = recollect_local_design(policy, dict(roots=[root]),
            Path(temporary)/'root.pth', simulation_budget=800+400*len(root['outcomes']),
            workers=1, inference_batch=inference_batch, inference_device=inference_device,
            physical_response_horizon=physical_response_horizon, isolate_factual_start=isolate_factual_start)
    if not value['all_design_roots_measured'] or len(value['roots']) != 1:
        raise ValueError('A breadth root did not finish all its paired queries.')
    measured = value['roots'][0]
    measured['design_root'] = root_id
    return measured, value['simulation_steps']


def _recollect_snapshot_root_batch(task):
    """Measure sparse queries from several real roots in shared inference batches."""
    import conditional_fusion_value as c
    import develop_source_fusion as d
    import run_remote_evidence as r
    policy, scheduled, inference_batch, inference_device, physical_response_horizon = task[:5]
    isolate_factual_start = task[5] if len(task) > 5 else False
    response_feature_batch = task[6] if len(task) > 6 else 0
    feedback_relative_hold = task[7] if len(task) > 7 else False
    policy = Path(policy)
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    r.initialize_source_branches(str(d.SOURCE))
    actor = d.joint_load(str(policy), policy.stat().st_mtime_ns).cpu()
    if actor.successor_task_value is not None:
        raise ValueError('Snapshot comparisons require their frozen predecessor without M2.')
    coefficients = np.asarray([[-1., -1.]]+[[a, b] for a in (0., .5, 1.) for b in (0., .5, 1.)])
    roots = []; jobs = []; identities = []; steps = []
    for root_id, original in scheduled:
        original = observed_programs(original)
        snapshot = original.get('root_snapshot')
        if (snapshot is None or snapshot['predecessor_sha256'] != digest or
                int(snapshot['root_steps']) != int(original['root_steps'])):
            raise ValueError('Batched roots need actual snapshots from this exact predecessor and clock.')
        scene = tuple(original['scene']); n = int(scene[1])
        env = copy.deepcopy(snapshot['environment']); obs = snapshot['observation'].copy()
        packet = d.mechanism_tensor(d.joint_packet(env, obs))
        with torch.no_grad():
            control = actor(packet, deterministic=True, gate_only=True)[0][0].numpy()[:, :3]
            context, order = c.intervention_task_context(packet, True, True)
            effect = c.paired_effect_controls(actor, packet, torch.as_tensor(control[None]))
        inverse = order[0].argsort().numpy(); one = np.ones((n, 1), np.float32)
        options = list(map(tuple, original['options']))
        actions = []
        for code, delay, span in options:
            if not code:
                actions.append(control)
            else:
                preference = coefficients[(code//(10**np.arange(n))) % 10][inverse].astype(np.float32)
                first = preference if delay == 0 else np.full_like(preference, -1.)
                actions.append(np.concatenate((control, -one, (delay+span)*one, -2*one,
                    np.zeros((n, 2), np.float32), delay*one, first, preference,
                    np.zeros_like(one)), -1))
        position = len(roots)
        roots.append(dict(context=context[0].numpy(), effect_controls=effect[0, order[0]].numpy(),
            options=options, outcomes={}, futures=list(original['futures']), scene=scene,
            feedback_relative_hold=feedback_relative_hold,
            prefix_future=original['prefix_future'], root_steps=int(snapshot['root_steps']),
            lookback=original.get('lookback'), order=order[0].numpy(),
            future_weights=np.asarray(original.get('future_weights', np.ones(len(original['futures']))), np.float32),
            design_root=root_id, root_sampling=dict(kind=snapshot.get('selection_reason', 'uniform_eligible_factual_state'),
                eligible_states=snapshot['eligible_states'], root_sample_seed=snapshot['root_sample_seed'],
                inference_batch=snapshot['inference_batch'], inference_device=snapshot['inference_device'])))
        steps.append(0)
        for q, f in sorted(original['outcomes']):
            jobs.append((str(policy), dict(environment=env, observation=obs,
                random_state=snapshot['random_state'], action=actions[q],
                isolate_factual_start=isolate_factual_start,
                feedback_relative_hold=feedback_relative_hold), scene, original['futures'][f],
                max(1, physical_response_horizon, options[q][1]+options[q][2])))
            identities.append((position, q, f))
    measured = source_successor_batches(jobs, batch_size=inference_batch, device=inference_device,
        physical_response_horizon=physical_response_horizon)
    continuation_controls = {}
    if response_feature_batch:
        # These are features of already measured states. Encoding them in
        # batches cannot change the simulated branches or their outcomes.
        groups = {}
        for index, row in enumerate(measured):
            response = row.get('response')
            if response is not None and not response['terminal']:
                packet = dict(response['packet'])
                packet.setdefault('_conditional_count', np.zeros(1, np.float32))
                key = tuple((name, np.asarray(value).shape) for name, value in sorted(packet.items()))
                groups.setdefault(key, []).append((index, packet))
        actor.to(inference_device)
        try:
            with torch.no_grad():
                for group in groups.values():
                    for begin in range(0, len(group), response_feature_batch):
                        batch = group[begin:begin+response_feature_batch]
                        packet = {name:torch.as_tensor(np.stack([record[name] for _, record in batch]),
                            device=inference_device, dtype=torch.float32) for name in batch[0][1]}
                        native = actor(packet, deterministic=True, gate_only=True)[0]
                        controls = c.paired_effect_controls(actor, packet, native).cpu().numpy()
                        continuation_controls.update((index, controls[local])
                            for local, (index, _) in enumerate(batch))
        finally:
            actor.cpu()
    for index, ((position, q, f), row) in enumerate(zip(identities, measured)):
        root = roots[position]; n = len(root['order'])
        item = dict(success=float(row['success']), outcome=row['outcome'], duration=row['duration'])
        physical_response = row.get('physical_response', row.get('response'))
        if (physical_response_horizon and physical_response is not None
                and physical_response['steps'] <= physical_response_horizon):
            with torch.no_grad():
                physical_context, physical_order = c.intervention_task_context(
                    d.mechanism_tensor(physical_response['packet']), True, True)
                nodes = physical_context[0, :8*n].reshape(n, 8)
                nodes = nodes[physical_order[0].argsort()][torch.as_tensor(root['order'])]
                item.update(following=torch.cat((nodes.flatten(), physical_context[0, 8*n:])).numpy(),
                    response_steps=physical_response['steps'], response_sampling='physical_tick')
        response = row.get('response')
        if response is not None:
            successor = d.mechanism_tensor(response['packet'])
            with torch.no_grad():
                later_context, later_order = c.intervention_task_context(successor, True, True)
                if not response['terminal']:
                    if response_feature_batch:
                        later_control = continuation_controls[index][later_order[0].numpy()]
                    else:
                        native = actor(successor, deterministic=True, gate_only=True)[0]
                        later_control = c.paired_effect_controls(actor, successor, native)[0, later_order[0]].numpy()
                    item.update(continuation_context=later_context[0].numpy(),
                        continuation_controls=later_control,
                        continuation_order=later_order[0].numpy(),
                        continuation_remaining=row['physical_steps']-response['steps'])
        root['outcomes'][q, f] = item
        steps[position] += row['physical_steps']
    if any(len(root['outcomes']) != len(original['outcomes'])
           for root, (_, original) in zip(roots, scheduled)):
        raise ValueError('Every scheduled root and paired future must be measured exactly once.')
    return list(zip(roots, steps))


def recollect_breadth_design(policy, blueprint, output, *, workers=4,
                            inference_batch=32, inference_device='cpu', physical_response_horizon=0,
                            root_batch_size=1, isolate_factual_start=False, response_feature_batch=0,
                            resume_inference_batch=False, feedback_relative_hold=False):
    """Measure bounded roots with an explicit, resumable inference schedule."""
    import develop_source_fusion as d
    import run_remote_evidence as r
    policy = Path(policy); output = Path(output)
    design = blueprint.get('acquisition_design')
    if design is None or design.get('kind') not in ('breadth_local_pairs', 'paired_effect_confirmation'):
        raise ValueError('Root-parallel acquisition requires a bounded paired design.')
    if workers < 1 or inference_batch < 0:
        raise ValueError('Use a positive worker count and nonnegative branch batch size.')
    if (not isinstance(root_batch_size, int) or root_batch_size < 1 or
            (root_batch_size > 1 and (not inference_batch or
             any('root_snapshot' not in root for root in blueprint['roots'])))):
        raise ValueError('Batched roots need a positive batch size and actual predecessor snapshots.')
    if not isinstance(physical_response_horizon, int) or not 0 <= physical_response_horizon <= 20:
        raise ValueError('The optional measured response must use a horizon from zero to twenty control ticks.')
    if not isinstance(isolate_factual_start, bool):
        raise ValueError('The intervention-start execution rule must be boolean.')
    if not isinstance(feedback_relative_hold, bool) or (feedback_relative_hold and
            (not isolate_factual_start or root_batch_size <= 1 or not inference_batch)):
        raise ValueError('Relative feedback collection requires isolated snapshot batches.')
    if not isinstance(resume_inference_batch, bool):
        raise ValueError('Resuming with a different inference batch must be explicitly enabled.')
    if (not isinstance(response_feature_batch, int) or response_feature_batch < 0 or
            (response_feature_batch and root_batch_size == 1)):
        raise ValueError('Batched response features require snapshot root batches and a positive batch size.')
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    result = (torch.load(output, map_location='cpu', weights_only=False) if output.exists() else
        dict(model=torch.load(policy, map_location='cpu', weights_only=False), roots=[],
            predecessor_sha256=digest, acquisition_design=copy.deepcopy(design),
            simulation_steps=0, next_design_root=0, completed=False, inherited_labels=False,
            training_data=str(output), inference_batch=inference_batch,
            inference_device=inference_device, root_dispatch_kind='independent_bounded_roots',
            root_batch_size=root_batch_size))
    changed_batch = result['inference_batch'] != inference_batch
    if (result['predecessor_sha256'] != digest or result['acquisition_design'] != design
            or (changed_batch and not resume_inference_batch) or result['inference_device'] != inference_device
            or result.get('root_batch_size', 1) != root_batch_size
            or (result.get('isolate_factual_start', False) != isolate_factual_start and result['roots'])
            or (result.get('feedback_relative_hold', False) != feedback_relative_hold and result['roots'])
            or (result.get('physical_response_horizon', 0) != physical_response_horizon and result['roots'])):
        raise ValueError('The saved breadth data uses a different predecessor, design or branch batch.')
    if physical_response_horizon:
        result['physical_response_horizon'] = physical_response_horizon
    if isolate_factual_start:
        result['isolate_factual_start'] = True
    if feedback_relative_hold:
        result['feedback_relative_hold'] = True
    if root_batch_size > 1:
        result.update(root_batch_size=root_batch_size, root_dispatch_kind='snapshot_root_batches')
    if result['model'].get('successor_task_value') is not None:
        raise ValueError('Breadth sampling requires its frozen controller without M2.')
    if result['completed']:
        return result
    if (not 0 <= result['next_design_root'] <= len(blueprint['roots']) or
            [root['design_root'] for root in result['roots']] != list(range(result['next_design_root'])) or
            (result['next_design_root'] % root_batch_size and result['next_design_root'] != len(blueprint['roots']))):
        raise ValueError('The saved breadth roots must retain their original design order.')
    if changed_batch:
        if root_batch_size == 1 or min(result['inference_batch'], inference_batch) <= 0:
            raise ValueError('Batch changes require complete snapshot root batches and positive inference sizes.')
        # Completed roots and all their measured adverse outcomes stay intact.
        # Only untouched root batches use the new execution allocation. Keep
        # the boundary explicit because batching can affect numerical ties.
        runs = result.setdefault('inference_batch_runs', [dict(first_design_root=0,
            inference_batch=result['inference_batch'], inference_device=inference_device)])
        runs.append(dict(first_design_root=result['next_design_root'],
            inference_batch=inference_batch, inference_device=inference_device))
        result['inference_batch'] = inference_batch
    result.update(workers=workers, design_root_count=len(blueprint['roots']),
        planned_step_ceiling=sum(800+400*len(root['outcomes']) for root in blueprint['roots']))
    if root_batch_size > 1 and physical_response_horizon:
        runs = result.setdefault('physical_response_capture_runs', [])
        if not runs and result['next_design_root']:
            runs.append(dict(first_design_root=0, kind='maximum_of_window_end_and_horizon'))
        if not runs or runs[-1]['kind'] != 'fixed_horizon_and_separate_window_end':
            runs.append(dict(first_design_root=result['next_design_root'],
                kind='fixed_horizon_and_separate_window_end'))
    if response_feature_batch or result.get('response_feature_encoding_runs'):
        runs = result.setdefault('response_feature_encoding_runs', [])
        if not runs and result['next_design_root']:
            runs.append(dict(first_design_root=0, batch_size=0, device='cpu'))
        encoding = dict(batch_size=response_feature_batch,
            device=inference_device if response_feature_batch else 'cpu')
        if not runs or any(runs[-1][key] != value for key, value in encoding.items()):
            runs.append(dict(first_design_root=result['next_design_root'], **encoding))
    if root_batch_size > 1:
        pending = [(str(policy), [(index, blueprint['roots'][index])
            for index in range(begin, min(begin+root_batch_size, len(blueprint['roots'])))],
            inference_batch, inference_device, physical_response_horizon)
            + ((isolate_factual_start, response_feature_batch, feedback_relative_hold) if feedback_relative_hold else
               (isolate_factual_start, response_feature_batch) if response_feature_batch else
               (True,) if isolate_factual_start else ())
            for begin in range(result['next_design_root'], len(blueprint['roots']), root_batch_size)]
        collect = _recollect_snapshot_root_batch
    else:
        pending = [(str(policy), index, blueprint['roots'][index], inference_batch, inference_device)
                   + ((physical_response_horizon, True) if isolate_factual_start else
                      (physical_response_horizon,) if physical_response_horizon else ())
                   for index in range(result['next_design_root'], len(blueprint['roots']))]
        collect = _recollect_breadth_root
    with (ProcessPoolExecutor(max_workers=workers, initializer=r.initialize_source_branches,
          initargs=(str(d.SOURCE),)) if workers > 1 else nullcontext(None)) as pool:
        measured = pool.map(collect, pending, chunksize=1) if pool else map(collect, pending)
        for group in measured:
            for root, steps in (group if root_batch_size > 1 else [group]):
                result['roots'].append(root)
                result['simulation_steps'] += steps
                result['next_design_root'] += 1
            if root_batch_size > 1 or len(result['roots']) % 4 == 0 or len(result['roots']) == len(blueprint['roots']):
                output.parent.mkdir(parents=True, exist_ok=True)
                r.atomic_save(result, output)
                print(dict(phase='breadth_root_collection', roots=len(result['roots']),
                    total=len(blueprint['roots']), physical_steps=result['simulation_steps']), flush=True)
    result.update(completed=True, all_design_roots_measured=True,
        measured_terminal_observations=sum(len(root['outcomes']) for root in result['roots']))
    r.atomic_save(result, output)
    return result


def recollect_local_design(policy, blueprint, output, *, simulation_budget, workers=4,
                           inference_batch=0, inference_device='cpu', physical_response_horizon=0,
                           isolate_factual_start=False):
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
    if not isinstance(physical_response_horizon, int) or not 0 <= physical_response_horizon <= 20:
        raise ValueError('The optional measured response must use a horizon from zero to twenty control ticks.')
    if not isinstance(isolate_factual_start, bool):
        raise ValueError('The intervention-start execution rule must be boolean.')
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
    if (result.get('physical_response_horizon', 0) != physical_response_horizon
            and (result['roots'] or result.get('partial_root'))):
        raise ValueError('The saved replacement data uses a different physical response horizon.')
    if (result.get('isolate_factual_start', False) != isolate_factual_start
            and (result['roots'] or result.get('partial_root'))):
        raise ValueError('The saved replacement data uses a different intervention-start execution rule.')
    if physical_response_horizon:
        result['physical_response_horizon'] = physical_response_horizon
    if isolate_factual_start:
        result['isolate_factual_start'] = True
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
            snapshot=original.get('root_snapshot')
            if snapshot is not None:
                if (snapshot['predecessor_sha256']!=digest
                        or int(snapshot['root_steps'])!=int(original['root_steps'])):
                    raise ValueError('A sampled state must retain its exact factual controller and clock.')
                env=copy.deepcopy(snapshot['environment']);obs=snapshot['observation'].copy()
                np.random.set_state(snapshot['random_state'])
                elapsed=int(snapshot['root_steps']);done=0
            else:
                np.random.seed(scene[0]); torch.manual_seed(scene[0])
                env = d.numerical_source().TADEnv(scene[1])
                obs, _ = env.reset(agility=scene[2], noisy_agility=False)
                if future is not None: np.random.seed(future)
                elapsed = done = 0
            history = []
            if original['root_steps'] and snapshot is None:
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
                    random_state=random_state, action=actions[q], isolate_factual_start=isolate_factual_start),
                    scene, original['futures'][f],
                    max(1, physical_response_horizon, options[q][1]+options[q][2])) for q, f in selected]
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
                    if physical_response_horizon and response is not None and response['steps'] <= physical_response_horizon:
                        successor = d.mechanism_tensor(response['packet'])
                        later_context, later_order = c.intervention_task_context(successor, True, True)
                        nodes = later_context[0, :8*scene[1]].reshape(scene[1], 8)
                        nodes = nodes[later_order[0].argsort()][order[0]]
                        item.update(following=torch.cat((nodes.flatten(), later_context[0, 8*scene[1]:])).numpy(),
                            response_steps=response['steps'], response_sampling='physical_tick')
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
                if snapshot is not None:
                    result['roots'][-1]['root_sampling']=dict(kind='uniform_eligible_factual_state',
                        eligible_states=snapshot['eligible_states'],root_sample_seed=snapshot['root_sample_seed'],
                        inference_batch=snapshot['inference_batch'],inference_device=snapshot['inference_device'])
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
