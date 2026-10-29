"""Mechanism one: joint selection of source/Boids actuator candidates."""
import study_runtime  # Configure numerical runtimes before PyTorch.
from math import isfinite

import torch
from torch import nn
from torch.nn import functional as F


def fleet_context(packet,return_order=False,version=2):
    """Canonical public fleet state for candidate selection and matched controls."""
    motion=packet['motion'];heading=motion[...,2]
    c,s=heading[:,0].cos(),heading[:,0].sin();relative=packet['node'][:,0,2:4]*30.
    target=motion[:,0,:2]+torch.stack((c*relative[:,0]-s*relative[:,1],s*relative[:,0]+c*relative[:,1]),-1)
    direction=-target/target.norm(dim=-1,keepdim=True).clamp_min(1.)
    def rotate(vector):
        return torch.stack(((vector*direction[:,None]).sum(-1),
            direction[:,None,0]*vector[...,1]-direction[:,None,1]*vector[...,0]),-1)
    relative=rotate(motion[...,:2]-target[:,None])/30.
    body=rotate(torch.stack((heading.cos(),heading.sin()),-1))
    rows=torch.cat((relative,rotate(motion[...,3:5])/3.,body,motion[...,5:6],
        motion[...,:2].norm(dim=-1,keepdim=True)/30.),-1)
    order=torch.atan2(relative[...,1],relative[...,0]).argsort(-1)
    context=rows.gather(1,order[...,None].expand_as(rows)).flatten(1)
    if version>=2:
        # Interception credit depends on target motion and remaining time, even
        # when the defenders occupy the same relative configuration. Recover
        # target velocity from the public body-frame measurement, not water or
        # any realized future disturbance.
        velocity=packet['node'][:,0,4:6]*5.
        velocity=torch.stack((c*velocity[:,0]-s*velocity[:,1],
            s*velocity[:,0]+c*velocity[:,1]),-1)
        context=torch.cat((context,rotate(velocity[:,None])[:,0]/5.,
            packet['node'][:,0,15:16]),-1)
    return (context,order) if return_order else context


def fleet_task_context(packet,return_order=False,absolute_frame=False):
    context,order=fleet_context(packet,True,2)
    if absolute_frame:
        # The author's reset sets attacker heading to minus its world
        # bearing. Rotating away that observable bearing therefore merges
        # states with different future dynamics, especially at zero speed.
        motion=packet['motion'];heading=motion[:,0,2]
        relative=30.*packet['node'][:,0,2:4]
        target=motion[:,0,:2]+torch.stack((heading.cos()*relative[:,0]-heading.sin()*relative[:,1],
            heading.sin()*relative[:,0]+heading.cos()*relative[:,1]),-1)
        bearing=target/target.norm(dim=-1,keepdim=True).clamp_min(1.)
        context=torch.cat((context[:,:-3],bearing,context[:,-3:]),-1)
    return (context,order) if return_order else context


@torch.no_grad()
def fusion_controls(actor,packet,action):
    """Return both source commands and the selected physical fleet command."""
    source=action[...,:2];rule=packet['node'][...,11:13]
    physical=actor.control(packet,action[...,:3])
    factual=physical[...,:2]*physical[...,2:3]+rule*(1.-physical[...,2:3])
    return torch.cat((source,rule,factual),-1)


@torch.no_grad()
def actuator_joint_design(n,device,interaction_order=2):
    """Balanced complete plans for pair or three-vessel feedback effects."""
    if interaction_order not in (2,3):
        raise ValueError('Actuator designs support pair or three-vessel interactions.')
    if n>10:raise ValueError('The balanced actuator design supports at most ten vessels.')
    index=torch.arange(9**interaction_order,device=device)
    first=index//(9**(interaction_order-1))
    second=(index//(9**(interaction_order-2)))%9
    third=index%9 if interaction_order==3 else None
    columns=[]
    # Evaluate degree-one/two polynomials over GF(9)=GF(3)[a]/(a*a+1).
    # Any selected two/three vessel columns cover every joint assignment.
    def multiply(left,right):
        return (((left%3)*(right%3)+2*(left//3)*(right//3))%3
            +3*(((left%3)*(right//3)+(left//3)*(right%3))%3))
    for vessel in range(n):
        slope=vessel
        if slope==9:columns.append(second if third is None else third);continue
        product=multiply(slope,second)
        real=first%3+product%3;imaginary=first//3+product//3
        if third is not None:
            quadratic=multiply(multiply(slope,slope),third)
            real=real+quadratic%3;imaginary=imaginary+quadratic//3
        columns.append(real%3+3*(imaginary%3))
    return torch.stack(columns,-1)


@torch.no_grad()
def coupled_actuator_risk(actor,packet,source,plans,return_clearance=False,synchronized=False,horizon=None,
                          tail_horizon=0.,target_paths=None,target_response=None):
    """Forecast unchanged source feedback inside each complete fleet plan."""
    b,m,n=plans.shape[:3];dt=min(.05,actor.prediction_step) if synchronized else actor.prediction_step
    initial=(torch.cat((packet['motion'][...,:3],5.*packet['water']),-1)
        if synchronized else actor.prediction_motion(packet))
    state=initial[:,None].expand(-1,m,-1,-1).clone()
    measured_velocity=packet['motion'][:,None,:,3:].expand(-1,m,-1,-1)
    force=750.*(plans*source[:,None]+(1.-plans)*packet['node'][:,None,:,11:13])+250.
    minimum=None;feedback_steps=max(1,round(.2/dt))
    trajectories=[] if return_clearance in ('paths','task_paths') else None
    target_trajectories=[] if return_clearance=='task_paths' else None
    steps=round((actor.risk_horizon if horizon is None else horizon)/dt)
    if target_paths is not None and (target_paths.shape!=(b,steps,2)
                                     or not bool(torch.isfinite(target_paths).all())):
        raise ValueError('Public target scenarios must align with the complete feedback forecast.')
    if trajectories is not None and tail_horizon:
        raise ValueError('Complete feedback paths require an explicit simulated horizon.')
    if target_response is not None:
        if target_paths is not None or target_trajectories is None or abs(feedback_steps*dt-.2)>1e-9:
            raise ValueError('Reactive target prediction needs complete fleet paths at the source control interval.')
        public_target,public_velocity=actor.public_target_motion(packet)
        response_target=public_target[:,None].expand(-1,m,-1).clone()
        response_velocity=public_velocity[:,None].expand(-1,m,-1).clone()
        response_boids=packet['node'][:,None,:,11:13].expand(-1,m,-1,-1)
    task_guard=return_clearance=='task'
    if task_guard:
        target,target_velocity=actor.public_target_motion(packet)
        captured=torch.zeros((b,m),dtype=torch.bool,device=source.device)
        exposure=None
    for step in range(steps):
        response_boundary=target_response is not None and step%feedback_steps==0
        if response_boundary and step:
            response_target=next_target;response_velocity=next_velocity
        if actor.risk_feedback and step and step%feedback_steps==0:
            observed=torch.cat((state[...,:3],measured_velocity),-1) if synchronized else state
            if target_response is not None:
                obs,boids=actor.feedback_observation(packet,observed,step*dt,joint=True,
                    target_override=(response_target,response_velocity))
                response_boids=boids
            elif target_paths is None:
                obs,boids=actor.feedback_observation(packet,observed,step*dt,joint=True)
            else:
                preceding=(target_paths[:,step-2] if step>1 else actor.public_target_motion(packet)[0])
                target_state=(target_paths[:,step-1],(target_paths[:,step-1]-preceding)/dt)
                obs,boids=actor.feedback_observation(packet,observed,step*dt,joint=True,
                    target_override=target_state)
            future=actor.proposal_distribution(obs.reshape(b*m,n,-1))[0].tanh().reshape(b,m,n,2)
            force=750.*(plans*future+(1.-plans)*boids)+250.
        if response_boundary:
            observed=torch.cat((state[...,:3],measured_velocity),-1) if synchronized else state
            if step and not actor.risk_feedback:
                _,response_boids=actor.feedback_observation(packet,observed,step*dt,joint=True,
                    target_override=(response_target,response_velocity))
            next_target,next_velocity=target_response.predict_state(observed.reshape(b*m,n,6),
                response_boids.reshape(b*m,n,2),response_target.reshape(b*m,2),response_velocity.reshape(b*m,2))
            next_target=next_target.reshape(b,m,2);next_velocity=next_velocity.reshape(b,m,2)
        x,y,heading,vx,vy,yaw_rate=state.unbind(-1)
        measured_velocity=state[...,3:]
        x=x+dt*vx;y=y+dt*vy;heading=heading+dt*yaw_rate
        cosine,sine=heading.cos(),heading.sin();u=cosine*vx+sine*vy;v=-sine*vx+cosine*vy
        du=(400.*yaw_rate*v-(100.+150.*u.abs())*u+force.sum(-1))/380.
        dv=(-380.*yaw_rate*u-(100.+150.*v.abs())*v)/400.
        dr=(-20.*u*v-(980.+950.*yaw_rate.abs())*yaw_rate+1.25*(force[...,0]-force[...,1]))/1430.
        u=u+dt*du;v=v+dt*dv
        state=torch.stack((x,y,heading,cosine*u-sine*v,sine*u+cosine*v,yaw_rate+dt*dr),-1)
        if trajectories is not None:
            trajectories.append(state[...,:2])
            if target_trajectories is not None:
                if target_response is not None:
                    fraction=(step%feedback_steps+1)/feedback_steps
                    target_trajectories.append(response_target+fraction*(next_target-response_target))
                else:
                    target,velocity=actor.public_target_motion(packet)
                    predicted=(target_paths[:,step] if target_paths is not None else
                        actor.target_motion.predict(packet,(step+1)*dt)[0]
                        if getattr(actor,'target_motion',None) is not None and actor.interaction_enabled
                        else target+(step+1)*dt*velocity)
                    target_trajectories.append(predicted[:,None].expand(-1,m,-1))
            continue
        distance=(state[:,:,:,:2][:,:,:,None]-state[:,:,:,:2][:,:,None]).norm(dim=-1)
        minimum=distance if minimum is None else torch.minimum(minimum,distance)
        if task_guard:
            time=(step+1)*dt
            predicted_target=(target_paths[:,step] if target_paths is not None else
                (actor.target_motion.predict(packet,time)[0]
                 if getattr(actor,'target_motion',None) is not None and actor.interaction_enabled
                 else target+time*target_velocity))
            reach=(state[...,:2]-predicted_target[:,None,None]).norm(dim=-1).amin(-1)
            if (step+1)%feedback_steps==0:
                captured|=reach<5.
            # The source task fails when the target passes every defender
            # relative to the protected asset. Preserve its best interceptor,
            # rather than summing pursuit distances and encouraging crowding.
            outside=state[...,:2].norm(dim=-1).amin(-1)-predicted_target.norm(dim=-1)[:,None]
            exposure=outside if exposure is None else torch.maximum(exposure,outside)
    if trajectories is not None:
        paths=torch.stack(trajectories,-2)
        return (paths,torch.stack(target_trajectories,-2)) if target_trajectories is not None else paths
    if tail_horizon:
        # Short feedback forecasts can look safe while leaving two boats
        # rapidly closing at their endpoint. Continue their predicted relative
        # velocity to its closest approach, without compounding the source
        # policy's long-horizon feedback prediction error.
        displacement=state[:,:,:,:2][:,:,:,None]-state[:,:,:,:2][:,:,None]
        velocity=state[:,:,:,3:5][:,:,:,None]-state[:,:,:,3:5][:,:,None]
        closest=(-(displacement*velocity).sum(-1)/velocity.square().sum(-1).clamp_min(1e-8)).clamp(0.,tail_horizon)
        continuation=(displacement+closest[...,None]*velocity).norm(dim=-1)
        minimum=torch.minimum(minimum,continuation)
    diagonal=torch.eye(n,dtype=torch.bool,device=source.device)[None,None]
    clearance=minimum.masked_fill(diagonal,float('inf')).amin((-1,-2))
    cost=(actor.distance_risk(minimum)*torch.triu(torch.ones((n,n),device=source.device),1)).sum((-1,-2))
    if task_guard:
        reach=torch.where(captured,reach.new_full((),5.),reach.clamp_min(5.))
        return cost,clearance,torch.stack((reach,exposure),-1)
    if return_clearance=='both':return cost,clearance
    return clearance if return_clearance else cost


def refine_pair_blocks(unary, risk, plans, risk_weight, sweeps=1):
    """Optimize two vessels together while retaining every other fleet choice."""
    plans = plans.clone()
    b, n, k = unary.shape
    for _ in range(sweeps):
        previous = plans.clone()
        for first in range(n):
            for second in range(first+1, n):
                first_cost = unary[:, first, None].expand(-1, plans.shape[1], -1).clone()
                second_cost = unary[:, second, None].expand(-1, plans.shape[1], -1).clone()
                for other in range(n):
                    if other in (first, second):
                        continue
                    choices = plans[:, None, :, other].expand(-1, k, -1)
                    first_cost += risk_weight*risk[:, first, other].gather(2, choices).transpose(1, 2)
                    second_cost += risk_weight*risk[:, second, other].gather(2, choices).transpose(1, 2)
                cost = (first_cost[..., None]+second_cost[..., None, :]
                        +risk_weight*risk[:, first, second, None])
                flat = cost.flatten(-2)
                choice = flat.argmin(-1)
                current = plans[:, :, first]*k+plans[:, :, second]
                # Preserve the existing maneuver on an exact tie. A block
                # change must improve this fleet's interaction objective.
                better = flat.gather(2, choice[..., None]).squeeze(-1) < flat.gather(
                    2, current[..., None]).squeeze(-1)
                plans[:, :, first] = torch.where(better, choice//k, plans[:, :, first])
                plans[:, :, second] = torch.where(better, choice % k, plans[:, :, second])
        if torch.equal(previous, plans):
            break
    return plans


def distinct_fleet_shortlist(plans, scores, limit):
    """Retain distinct fleet maneuvers in increasing proposal-cost order."""
    if limit < 1:
        raise ValueError('A fleet shortlist must retain at least one plan.')
    count = min(limit, plans.shape[1])
    indices = []
    for rows, ranking in zip(plans.detach().cpu().tolist(), scores.argsort(
            dim=1, stable=True).detach().cpu().tolist()):
        selected = []; seen = set()
        for index in ranking:
            code = tuple(rows[index])
            if code in seen:
                continue
            seen.add(code); selected.append(index)
            if len(selected) == count:
                break
        indices.append(selected+[selected[-1]]*(count-len(selected)))
    indices = torch.tensor(indices, dtype=torch.long, device=plans.device)
    return plans.gather(1, indices[..., None].expand(-1, -1, plans.shape[-1]))


def pareto_fleet_refinement(unary, risk, plans, reference, risk_weight, componentwise=False):
    """Replace a fleet plan only when both control deviation and risk improve."""
    b, n, k = unary.shape
    catalog = torch.cat((reference[:, None], plans), 1)
    deviation = unary.new_zeros(catalog.shape[:2])
    interaction = torch.zeros_like(deviation)
    protected = torch.ones_like(deviation, dtype=torch.bool)
    for vessel in range(n):
        own = unary[:, vessel].gather(1, catalog[:, :, vessel])
        deviation += own
        if componentwise:
            protected &= own <= own[:, :1]
        for other in range(vessel):
            code = catalog[:, :, other]*k+catalog[:, :, vessel]
            pair = risk[:, other, vessel].flatten(1).gather(1, code)
            interaction += pair
            if componentwise:
                protected &= pair <= pair[:, :1]
    score = deviation+risk_weight*interaction
    admissible = ((deviation <= deviation[:, :1]) & (interaction <= interaction[:, :1])
                  & (score < score[:, :1]) & protected)
    admissible[:, 0] = True
    selected = score.masked_fill(~admissible, float('inf')).argmin(-1)
    return catalog[torch.arange(b, device=unary.device), selected]


def feedback_guarded_refinement(actor, packet, source, candidates, order, reference, proposed, horizons,
                                tail_horizon=0., task_guard=False):
    """Keep a proposal only when shared fleet feedback agrees across horizons."""
    if not horizons or any(not isfinite(horizon) or horizon <= 0 for horizon in horizons):
        raise ValueError('Feedback refinement requires positive forecast horizons.')
    if not isfinite(tail_horizon) or tail_horizon < 0:
        raise ValueError('Feedback refinement requires a finite nonnegative tail horizon.')
    changed = (reference != proposed).any(-1)
    if not bool(changed.any()):
        return reference
    subset = {key: (value[changed] if torch.is_tensor(value) and value.ndim
                   and value.shape[0] == len(source) else value) for key, value in packet.items()}
    choices = torch.stack((reference[changed], proposed[changed]), 1)
    catalog = candidates[changed, None].expand(-1, 2, -1, -1, -1)
    gates = catalog.gather(3, choices[..., None, None].expand(-1, -1, -1, 1, 2)).squeeze(3)
    physical = gates.gather(2, order[changed].argsort(-1)[:, None, :, None].expand(-1, 2, -1, 2))
    admissible = torch.ones(choices.shape[0], dtype=torch.bool, device=source.device)
    for horizon in horizons:
        settings = dict(synchronized=True, horizon=horizon)
        if tail_horizon:
            settings['tail_horizon'] = tail_horizon
        forecast = coupled_actuator_risk(
            actor, subset, source[changed], physical, 'task' if task_guard else 'both', **settings)
        risk, clearance = forecast[:2]
        admissible &= ((risk[:, 1] <= risk[:, 0]) & (clearance[:, 1] >= clearance[:, 0])
                       & torch.isfinite(risk).all(-1) & torch.isfinite(clearance).all(-1))
        if task_guard:
            progress = forecast[2]
            admissible &= ((progress[:, 1] <= progress[:, 0]).all(-1)
                           & torch.isfinite(progress).all((-1, -2)))
    result = reference.clone()
    result[changed] = torch.where(admissible[:, None], proposed[changed], reference[changed])
    return result


def actuator_candidate_grid(preferred, subdivisions=1, *, include_uniform_midpoint=False):
    """Resolve each side of the source preference without changing its hull."""
    if isinstance(subdivisions, bool) or not isinstance(subdivisions, int) or subdivisions < 1:
        raise ValueError('Candidate subdivisions must be a positive integer.')
    if not isinstance(include_uniform_midpoint, bool):
        raise ValueError('The common midpoint option must be boolean.')
    if subdivisions == 1:
        values = torch.stack((torch.zeros_like(preferred), preferred, torch.ones_like(preferred)), -2)
    else:
        fraction = torch.arange(subdivisions, device=preferred.device, dtype=preferred.dtype)/subdivisions
        lower = preferred[..., None, :]*fraction[:, None]
        upper = preferred[..., None, :] + (1.-preferred[..., None, :])*fraction[:, None]
        values = torch.cat((lower, upper, torch.ones_like(preferred[..., None, :])), -2)
    if include_uniform_midpoint:
        values = torch.cat((values, torch.full_like(preferred[..., None, :], .5)), -2)
    levels = values.shape[-2]
    digits = torch.arange(levels*levels, device=preferred.device)
    return torch.stack((values[..., digits//levels, 0], values[..., digits%levels, 1]), -1)


def independent_fleet_minimum_cost(values, probabilities):
    """Exact expected fleet minimum when only one candidate choice is free."""
    if values.shape != probabilities.shape or values.ndim != 3:
        raise ValueError('Fleet candidate values and probabilities must have shape batch, vessel, candidate.')
    support = torch.cat((values.new_zeros((len(values), 1)), values.flatten(1)), -1).sort(-1).values
    lower, intervals = support[:, :-1], support.diff(dim=-1)
    survives = ((values[..., None] > lower[:, None, None])*probabilities[..., None]).sum(2)
    costs = []
    for vessel in range(values.shape[1]):
        peers = [other for other in range(values.shape[1]) if other != vessel]
        survival = survives[:, peers].prod(1)
        costs.append((intervals[:, None]*survival[:, None]*
                      (values[:, vessel, :, None] > lower[:, None])).sum(-1))
    return torch.stack(costs, 1)


def task_stopped_fusion_factors(paths, target_path, unary, distance_risk, *, control_stride=2,
                               capture_radius=5.):
    """Reuse candidate paths as a fleet task that ends at its first capture.

    A collision or radial escape at the capture tick still has a cost, matching
    the source environment's termination priority. Later predicted encounters
    have no cost. All factors are shared by joint and independent selection.
    """
    b, n, k, horizon, coordinates = paths.shape
    if coordinates != 2 or target_path.shape != (b, horizon, 2) or unary.shape != (b, n, k):
        raise ValueError('Task-stopped fusion needs aligned candidate and target paths.')
    if control_stride < 1 or capture_radius <= 0:
        raise ValueError('Task stopping needs a positive control stride and capture radius.')
    ticks = torch.arange(horizon, device=paths.device)
    observed = (ticks+1) % control_stride == 0
    capture = ((paths-target_path[:, None, None]).norm(dim=-1) < capture_radius) & observed
    first = torch.where(capture, ticks, horizon-1).amin(-1)
    alive = torch.cat((torch.ones_like(capture[..., :1]),
                       ~capture[..., :-1].cummax(-1).values), -1)
    pairs, cumulative, increments = [], [], []
    for left in range(n):
        for right in range(left+1, n):
            distance = (paths[:, left, :, None]-paths[:, right, None]).norm(dim=-1)
            risk = distance_risk(distance).cummax(-1).values
            pairs.append((left, right)); cumulative.append(risk)
            increments.append(torch.cat((risk[..., :1], risk.diff(dim=-1)), -1))
    # The source task fails when every defender is farther from the asset
    # than the attacker. The fleet minimum preserves a remaining interceptor;
    # adding individual pursuit distances would instead encourage crowding.
    exposure = (paths.norm(dim=-1)-target_path.norm(dim=-1)[:, None, None]).clamp_min(0.).square()
    exposure = exposure*observed
    return dict(unary=unary, first_capture=first, alive=alive, pairs=pairs,
                pair_indices=torch.tensor(pairs, dtype=torch.long, device=paths.device),
                cumulative_risk=torch.stack(cumulative, 1), risk_increments=increments,
                exposure=exposure, observed_ticks=max(1, int(observed.sum().item())))


def target_turning_envelope(position, central_path, degrees):
    """Construct public target futures with a common observed motion prefix.

    Deflection grows from zero with forecast time; an instantaneous rotation
    of the measured velocity would otherwise invent an unobserved maneuver.
    The center is unchanged and no controller, agility, or future is queried.
    """
    if not isfinite(degrees) or not 0. <= degrees <= 90.:
        raise ValueError('The public target turning envelope must be within zero to ninety degrees.')
    if central_path.ndim != 3 or position.shape != central_path[:, 0].shape:
        raise ValueError('The target envelope requires aligned public positions and forecast paths.')
    if not degrees:
        return central_path[:, None], central_path.new_ones(1)
    fractions = torch.arange(1, central_path.shape[1]+1, device=central_path.device,
                             dtype=central_path.dtype)/central_path.shape[1]
    branches = central_path.new_tensor([-1., -.5, 0., .5, 1.])
    angle = branches[:, None]*fractions[None]*(degrees*torch.pi/180.)
    displacement = central_path-position[:, None]
    x, y = displacement[:, None, :, 0], displacement[:, None, :, 1]
    rotated = torch.stack((angle.cos()*x-angle.sin()*y,
                           angle.sin()*x+angle.cos()*y), -1)
    return position[:, None, None]+rotated, central_path.new_tensor([.1, .2, .4, .2, .1])


def task_envelope_fusion_factors(paths, target_paths, unary, distance_risk, *, weights,
                                 interception_weight, control_stride=2, capture_radius=5.):
    """Score complementary interception of shared public target futures.

    Both selectors reuse precisely the same forecasts. Five-dimensional paths
    share each vessel forecast across scenarios; six-dimensional paths carry
    the separately simulated feedback response to each shared target future.
    Each future contributes the fleet's best interception and its own first
    capture stopping time, allowing different vessels to cover different turns.
    """
    if paths.ndim not in (5,6):
        raise ValueError('Candidate paths need an optional target-scenario axis.')
    b=paths.shape[0];n,k,horizon,coordinates=paths.shape[-4:]
    if target_paths.ndim != 4 or target_paths.shape[0] != b or target_paths.shape[2:] != (horizon, 2):
        raise ValueError('Target futures must align with the common candidate forecast horizon.')
    scenarios = target_paths.shape[1]
    weights = torch.as_tensor(weights, device=paths.device, dtype=paths.dtype)
    if weights.shape != (scenarios,) or not torch.isfinite(weights).all() or not (weights > 0.).all():
        raise ValueError('Every shared target future needs a finite positive weight.')
    if not isfinite(interception_weight) or interception_weight < 0.:
        raise ValueError('The fleet interception weight must be finite and nonnegative.')
    if paths.ndim==6 and paths.shape[1]!=scenarios:
        raise ValueError('Each target future needs its matching fleet feedback response.')
    repeated = (paths[:, None].expand(-1, scenarios, -1, -1, -1, -1)
                if paths.ndim==5 else paths).reshape(b*scenarios, n, k, horizon, coordinates)
    targets = target_paths.reshape(b*scenarios, horizon, 2)
    repeated_unary = unary[:, None].expand(-1, scenarios, -1, -1).reshape(b*scenarios, n, k)
    components = task_stopped_fusion_factors(repeated, targets, repeated_unary, distance_risk,
        control_stride=control_stride, capture_radius=capture_radius)
    observed = (torch.arange(horizon, device=paths.device)+1) % control_stride == 0
    if not bool(observed.any()):
        raise ValueError('Fleet interception needs at least one forecast control tick.')
    distance = (repeated-targets[:, None, None]).norm(dim=-1).masked_fill(~observed, float('inf'))
    components['interception_deficit'] = (distance.amin(-1)-capture_radius).clamp_min(0.)/capture_radius
    components['interception_weight'] = interception_weight
    return dict(unary=unary, scenario_factors=components, scenario_weights=weights/weights.sum())


def task_stopped_fleet_cost(factors, plans, risk_weight=32., escape_weight=0.):
    """Evaluate complete fleet plans without making any extra motion forecast."""
    unary = factors['unary']; b, n, k = unary.shape
    if plans.ndim != 3 or plans.shape[0] != b or plans.shape[-1] != n:
        raise ValueError('Complete candidate plans must have shape batch, plan, vessel.')
    if 'scenario_factors' in factors:
        weights = factors['scenario_weights']; scenarios = len(weights)
        repeated = plans[:, None].expand(-1, scenarios, -1, -1).reshape(b*scenarios, -1, n)
        costs = task_stopped_fleet_cost(factors['scenario_factors'], repeated, risk_weight,
                                        escape_weight).reshape(b, scenarios, -1)
        return (costs*weights[None, :, None]).sum(1)
    index = plans.transpose(1, 2)
    stop = factors['first_capture'].gather(2, index).amin(1)
    cost = unary.gather(2, index).sum(1)
    horizon = factors['alive'].shape[-1]
    pairs = factors['pair_indices']
    pair_index = ((plans[:, :, pairs[:, 0]]*k+plans[:, :, pairs[:, 1]])*horizon
                  +stop[..., None]).transpose(1, 2)
    cost = cost+risk_weight*factors['cumulative_risk'].flatten(2).gather(2, pair_index).sum(1)
    if escape_weight:
        exposure = factors['exposure'].gather(2, index[..., None].expand(-1, -1, -1, horizon)).amin(1)
        alive = torch.arange(horizon, device=plans.device) <= stop[..., None]
        cost = cost+escape_weight*(exposure*alive).sum(-1)/factors['observed_ticks']
    if factors.get('interception_weight', 0.):
        cost = cost+factors['interception_weight']*factors['interception_deficit'].gather(2, index).amin(1)
    return cost


def task_stopped_independent_cost(factors, probabilities, risk_weight=32., escape_weight=0.):
    """Exact conditional expectation of the SAME fleet cost for each vessel.

    Each subject optimizes independently against the peers' unchanged source
    distribution. Products of capture-survival probabilities integrate every
    peer assignment exactly; no Monte Carlo or coupled choice is hidden here.
    """
    unary = factors['unary']; b, n, k = unary.shape
    if probabilities.shape != unary.shape:
        raise ValueError('Independent task stopping needs the shared candidate distribution.')
    if 'scenario_factors' in factors:
        weights = factors['scenario_weights']; scenarios = len(weights)
        repeated = probabilities[:, None].expand(-1, scenarios, -1, -1).reshape(b*scenarios, n, k)
        costs = task_stopped_independent_cost(factors['scenario_factors'], repeated, risk_weight,
                                              escape_weight).reshape(b, scenarios, n, k)
        return (costs*weights[None, :, None, None]).sum(1)
    alive = factors['alive'].to(unary.dtype)
    weighted_alive = probabilities[..., None]*alive
    survival = weighted_alive.sum(2)
    expected_unary = (unary.masked_fill(probabilities == 0., 0.)*probabilities).sum(-1)
    cost = unary+expected_unary.sum(-1)[:, None, None]-expected_unary[..., None]
    for (left, right), increment in zip(factors['pairs'], factors['risk_increments']):
        pair = torch.einsum('bgt,bht,bght->bt',
            weighted_alive[:, left], weighted_alive[:, right], increment)
        for subject in range(n):
            if subject == left:
                term = torch.einsum('bht,bght->bgt', weighted_alive[:, right], increment)*alive[:, left]
            elif subject == right:
                term = torch.einsum('bgt,bght->bht', weighted_alive[:, left], increment)*alive[:, right]
            else:
                term = pair[:, None]*alive[:, subject]
            others = [vessel for vessel in range(n) if vessel not in (left, right, subject)]
            if others:
                term = term*survival[:, others].prod(1)[:, None]
            cost[:, subject] += risk_weight*term.sum(-1)
    if escape_weight:
        horizon = alive.shape[-1]
        # Captured candidates carry an absorbing zero exposure after capture.
        # Taking the fleet minimum therefore stops the escape cost at exactly
        # the same first capture as the joint objective.
        values = (factors['exposure']*alive).permute(0, 3, 1, 2).reshape(b*horizon, n, k)
        probability = probabilities[:, None].expand(-1, horizon, -1, -1).reshape(b*horizon, n, k)
        expected = independent_fleet_minimum_cost(values, probability).reshape(b, horizon, n, k)
        cost += escape_weight*expected.sum(1)/factors['observed_ticks']
    if factors.get('interception_weight', 0.):
        cost += factors['interception_weight']*independent_fleet_minimum_cost(
            factors['interception_deficit'], probabilities)
    return cost


def select_task_stopped_fusion(factors, *, selection, beam_width, risk_weight, escape_weight):
    unary = factors['unary']; b, n, k = unary.shape
    reference = unary.argmin(-1)
    if selection in ('independent', 'single_reference'):
        probability = ((-unary).softmax(-1) if selection == 'independent' else
                       F.one_hot(reference, k).to(unary.dtype))
        return task_stopped_independent_cost(factors, probability, risk_weight, escape_weight).argmin(-1)
    if selection != 'joint':
        raise ValueError('Unknown task-stopped fusion selection rule.')
    choices = torch.arange(k, device=unary.device)
    plans = reference[:, None]
    for vessel in range(n):
        expanded = plans[:, :, None].expand(-1, -1, k, -1).clone()
        expanded[..., vessel] = choices
        expanded = expanded.flatten(1, 2)
        scores = task_stopped_fleet_cost(factors, expanded, risk_weight, escape_weight)
        index = scores.topk(min(beam_width, scores.shape[1]), largest=False).indices
        plans = expanded.gather(1, index[..., None].expand(-1, -1, n))
    uniform = choices[None, :, None].expand(b, -1, n)
    plans = torch.cat((plans, reference[:, None], uniform), 1)
    initial = plans.clone()
    for _ in range(3):
        previous = plans.clone()
        for vessel in range(n):
            alternatives = plans[:, :, None].expand(-1, -1, k, -1).clone()
            alternatives[..., vessel] = choices
            scores = task_stopped_fleet_cost(factors, alternatives.flatten(1, 2), risk_weight,
                                             escape_weight).reshape(b, plans.shape[1], k)
            best = scores.argmin(-1)
            current = scores.gather(-1, plans[:, :, vessel, None]).squeeze(-1)
            improve = scores.gather(-1, best[..., None]).squeeze(-1) < current
            plans[:, :, vessel] = torch.where(improve, best, plans[:, :, vessel])
        if torch.equal(previous, plans):
            break
    plans = torch.cat((initial, plans), 1)
    scores = task_stopped_fleet_cost(factors, plans, risk_weight, escape_weight)
    return plans[torch.arange(b, device=plans.device), scores.argmin(-1)]


class FeedbackFleetPolicy(nn.Module):
    """Learn whole-fleet candidate consequences from real closed-loop returns.

    Each graph is one complete candidate plan, including the response of its
    unchanged teammates. The independent control queries this exact same
    scorer on the reference-change slice of the common forecast library.
    """
    def __init__(self, width=48, analytic_temperature=2., state=None, candidate_contrast=False,
                 sampling_temperature=1., representation='candidate_graph', scenario_evaluation=False,
                 scenario_regret_weight=0.):
        super().__init__()
        if width < 8 or not isfinite(analytic_temperature) or analytic_temperature <= 0:
            raise ValueError('The fleet policy needs a positive width and analytic temperature.')
        self.width=width;self.analytic_temperature=float(analytic_temperature)
        if not isfinite(sampling_temperature) or sampling_temperature<=0:
            raise ValueError('Fleet exploration requires a positive finite sampling temperature.')
        self.sampling_temperature=float(sampling_temperature)
        self.candidate_contrast=bool(candidate_contrast)
        if not isinstance(scenario_evaluation,bool):
            raise ValueError('Separate scenario evaluation must be an explicit Boolean.')
        self.scenario_evaluation=scenario_evaluation
        if (not isfinite(scenario_regret_weight) or not 0.<=scenario_regret_weight<=1.
                or scenario_regret_weight and not scenario_evaluation):
            raise ValueError('Scenario regret requires complete futures and a weight between zero and one.')
        self.scenario_regret_weight=float(scenario_regret_weight)
        if representation not in ('candidate_graph','candidate_attention'):
            raise ValueError('Unknown complete-candidate representation.')
        if representation=='candidate_attention' and width%4:
            raise ValueError('Ordinary candidate attention needs a width divisible by four.')
        self.representation=representation
        self.node=nn.Sequential(nn.Linear(37,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        if representation=='candidate_graph':
            self.message=nn.Sequential(nn.Linear(2*width+17,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
            self.update=nn.Sequential(nn.Linear(3*width,width),nn.SiLU())
        else:
            # The control receives the same complete-candidate node features.
            # Ordinary attention replaces explicit pair geometry and messages;
            # at width 48 its total capacity differs by only five parameters.
            self.attention=nn.TransformerEncoderLayer(width,4,width+5,dropout=0.,
                activation='gelu',batch_first=True,norm_first=True)
        self.plan=nn.Sequential(nn.Linear(2*width+1,width),nn.SiLU())
        self.actor=nn.Linear(width,1)
        self.critic=nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.Linear(width,1))
        nn.init.zeros_(self.actor.weight);nn.init.zeros_(self.actor.bias)
        self.sample=False;self.collect=False;self.last_decision=None
        self.collection_dtype=torch.float16
        if state is not None:self.load_state_dict(state)

    @staticmethod
    def features(packet,source,preferred,plans,paths,futures,weights,order,target):
        b,count,_,n,horizon,_=paths.shape
        context,_=fleet_context(packet,True,2)
        rows=context[:,:8*n].reshape(b,n,8)
        source=source.gather(1,order[...,None].expand_as(source))
        rule=packet['node'][...,11:13].gather(1,order[...,None].expand(-1,-1,2))
        indices=torch.tensor([0,(horizon-1)//3,2*(horizon-1)//3,horizon-1],device=paths.device)
        future=futures.index_select(-2,indices)
        relative=paths.index_select(-2,indices)-(future[:,None,:,None] if future.ndim==4 else future[:,:,:,None])
        direction=-target/target.norm(dim=-1,keepdim=True).clamp_min(1.)
        forward=direction[:,None,None,None,None]
        relative=torch.stack(((relative*forward).sum(-1),
            forward[...,0]*relative[...,1]-forward[...,1]*relative[...,0]),-1)/30.
        mass=weights.to(paths).reshape(1,1,-1,1,1,1)
        mean=(relative*mass).sum(2)
        spread=(((relative-mean[:,:,None]).square().sum(-1)*mass[...,0]).sum(2)).sqrt()
        distance=(relative.norm(dim=-1)*mass[...,0]).sum(2)
        global_state=torch.cat((context[:,-3:],target.norm(dim=-1,keepdim=True)/30.,
            context.new_full((b,1),n/6.)),-1)
        shared=torch.cat((rows,source,rule,preferred),-1)[:,None].expand(-1,count,-1,-1)
        return torch.cat((shared,plans,mean.flatten(-2),spread,distance,
            global_state[:,None,None].expand(-1,count,n,-1)),-1)

    @classmethod
    def scenario_features(cls,packet,source,preferred,plans,paths,futures,weights,order,target,costs):
        """Keep each complete future intact through the nonlinear fleet scorer."""
        b,count,scenarios,n=paths.shape[:4]
        if costs.shape!=(b,count,scenarios) or weights.shape!=(scenarios,):
            raise ValueError('Each complete future needs its matching fleet cost and probability.')
        pieces=[cls.features(packet,source,preferred,plans,paths[:,:,index:index+1],
            futures[:,:,index:index+1],weights.new_ones(1),order,target) for index in range(scenarios)]
        cost=costs[...,None,None].expand(-1,-1,-1,n,1)
        mass=weights[None,None,:,None,None].expand(b,count,-1,n,1)
        return torch.cat((torch.stack(pieces,2),cost,mass),-1)

    def forward(self,features,costs):
        if features.ndim==5:
            if not self.scenario_evaluation or features.shape[-1]!=39 or not features.shape[2]:
                raise ValueError('Scenario scoring needs complete public features, costs and probabilities.')
            logits=[];values=[];scenario_logits=[]
            for index in range(features.shape[2]):
                logit,value=self(features[:,:,index,:,:37],features[:,:,index,0,37])
                scenario_logits.append(logit)
                logits.append(logit*features[:,:,index,0,38])
                values.append(value*features[:,0,index,0,38])
            if self.scenario_regret_weight:
                scores=torch.stack(scenario_logits,-1)
                probabilities=features[:,:,:,0,38]
                regret=scores.amax(1,keepdim=True)-scores
                expected=(regret*probabilities).sum(-1)
                worst=regret.masked_fill(probabilities<=0.,-torch.inf).amax(-1)
                weight=self.scenario_regret_weight
                return -((1.-weight)*expected+weight*worst),torch.stack(values).sum(0)
            return torch.stack(logits).sum(0),torch.stack(values).sum(0)
        b,count,n,_=features.shape
        x=features.reshape(b*count,n,-1)
        if self.representation=='candidate_graph':
            difference=x[:,:,None]-x[:,None]
            trajectory=difference[...,16:24].reshape(b*count,n,n,4,2)
            geometry=torch.cat((difference[...,:4],difference[...,:2].norm(dim=-1,keepdim=True),
                trajectory.flatten(-2),trajectory.norm(dim=-1)),-1)
            own=self.node(x)
            mask=~torch.eye(n,dtype=torch.bool,device=x.device)[None,:,:,None]
            for _ in range(2):
                message=self.message(torch.cat((own[:,:,None].expand(-1,-1,n,-1),
                    own[:,None].expand(-1,n,-1,-1),geometry),-1))
                own=own+self.update(torch.cat((own,(message*mask).sum(2)/max(n-1,1),
                    message.masked_fill(~mask,-1e4).amax(2)),-1))
        else:
            own=self.node(x)
            for _ in range(2):own=self.attention(own)
        raw_cost=costs.clamp(0.,1e4)
        plans=self.plan(torch.cat((own.mean(1),own.amax(1),raw_cost.log1p().reshape(-1,1)),-1))
        plans=plans.reshape(b,count,-1)
        prior=-(raw_cost-raw_cost.amin(-1,keepdim=True)).clamp(max=80.)/self.analytic_temperature
        policy_features=plans
        if self.candidate_contrast:
            # Shared scene features can dominate the much smaller differences
            # between complete fleet plans. Assign policy credit to those
            # within-state differences while retaining absolute scene features
            # for the return baseline. The scale is a detached statistic of
            # this common candidate library, not a learned target or label.
            reference=plans[:,:81+9*n]
            center=reference.mean(1,keepdim=True)
            contrast=plans-center
            scale=(reference-center).square().mean(1,keepdim=True).sqrt().detach().clamp_min(.02)
            policy_features=contrast/scale
        logits=prior+self.actor(policy_features).squeeze(-1)
        value=self.critic(torch.cat((plans.mean(1),plans.amax(1)),-1)).squeeze(-1)
        return logits,value

    @staticmethod
    def policy_distribution(logits,selection,defenders,reference_start=81,temperature=1.):
        if not isfinite(temperature) or temperature<=0:
            raise ValueError('Fleet sampling temperature must be positive and finite.')
        logits=logits/temperature
        if selection=='joint':
            return torch.distributions.Categorical(logits=logits)
        if selection=='single_reference':
            stop=reference_start+9*defenders
            if logits.shape[-1]<stop:
                raise ValueError('Independent learning requires all single-vessel reference queries.')
            local=logits[:,reference_start:stop].reshape(len(logits),defenders,9)
            return torch.distributions.Independent(torch.distributions.Categorical(logits=local),1)
        raise ValueError('Fleet learning supports joint or single-reference independent selection.')

    def choose(self,features,costs,library,reference_start,selection):
        logits,value=self(features,costs)
        if not self.sample and not self.collect:
            return select_feedback_fleet_plan(-logits,library,reference_start,selection)
        distribution=self.policy_distribution(logits,selection,library.shape[-1],reference_start,
            self.sampling_temperature)
        index=(distribution.sample() if self.sample else logits.argmax(-1) if selection=='joint' else
            select_feedback_fleet_plan(-logits,library,reference_start,selection))
        if self.collect:
            self.last_decision=dict(features=features.detach().to('cpu',self.collection_dtype),
                costs=costs.detach().cpu(),index=index.detach().cpu(),
                logp=distribution.log_prob(index).detach().cpu(),value=value.detach().cpu())
        return (library[torch.arange(len(library),device=library.device),index]
            if selection=='joint' else index)


def feedback_fleet_library(unary):
    """Share balanced fleet plans and every single-vessel reference change.

    The same complete feedback forecasts serve both selectors. The reference
    control changes each vessel independently while fixing every teammate at
    its source-preferred candidate; it never uses the selected joint plan.
    """
    if unary.ndim != 3 or unary.shape[-1] != 9:
        raise ValueError('The balanced fleet feedback library requires nine actuator candidates.')
    b,n,k=unary.shape
    reference=unary.argmin(-1)
    balanced=actuator_joint_design(n,unary.device)[None].expand(b,-1,-1)
    local=reference[:,None,None].expand(-1,n,k,-1).clone()
    for vessel in range(n):
        local[:,vessel,:,vessel]=torch.arange(k,device=unary.device)
    return torch.cat((balanced,local.flatten(1,2)),1),balanced.shape[1]


def select_feedback_fleet_plan(costs,library,reference_start,selection):
    """Choose jointly or select each vessel against a fixed teammate population."""
    b,count,n=library.shape
    reference_stop=reference_start+n*9
    if costs.shape!=(b,count) or count<reference_stop:
        raise ValueError('Feedback costs and the common fleet candidate library must align.')
    if selection=='joint':
        return library[torch.arange(b,device=costs.device),costs.argmin(-1)]
    if selection=='single_reference':
        return costs[:,reference_start:reference_stop].reshape(b,n,9).argmin(-1)
    if selection=='balanced_independent':
        # Every candidate is compared against its teammates in the same
        # balanced fleet design. No vessel conditions on another vessel's
        # selected action, and all choices are combined only after scoring.
        design=library[:,:reference_start]
        membership=F.one_hot(design,9).to(costs.dtype)
        count=membership.sum(1)
        if not reference_start or bool((count==0).any()):
            raise ValueError('Marginal independent selection needs every candidate in the balanced design.')
        marginal=(costs[:,:reference_start,None,None]*membership).sum(1)/count
        return marginal.argmin(-1)
    raise ValueError('Complete task-feedback prediction needs joint, reference or balanced-independent selection.')


def refine_actuator_interactions(actor,packet,source_action,preferred_action,*,beam_width=128,
                                 risk_weight=32.,preference_width=.4,selection='joint',
                                 source_preference=False,prior_map=None,coupled_prediction=False,
                                 contact_distance=5.,synchronized_prediction=False,
                                 refine_all_plans=False,pair_refinements=0,
                                 joint_feedback_topk=0,joint_feedback_horizon=None,
                                 pareto_refinement=False,pareto_feedback_horizons=(),
                                 pareto_componentwise=False,pareto_feedback_tail=0.,
                                 pareto_task_guard=False,
                                 coupled_horizon=None,coupled_recovery=False,recovery_horizon=.6,
                                 recovery_distance=None,interaction_order=2,expand_interactions=False,
                                 coupled_tail=0.,coupled_refinements=0,interior_fusion=None,
                                 candidate_subdivisions=1,capture_weight=0.,include_uniform_midpoint=False,
                                 task_stopping=False,escape_weight=0.,
                                 target_envelope_degrees=0.,target_envelope_weight=0.,
                                 task_feedback_prediction=False,independent_feedback_proposal=False,
                                 nominal_feedback_proposals=False,feedback_refinement_rounds=0,
                                 return_coefficients=False,candidate_output=None):
    """Choose source/Boids actuator mixtures from common candidate forecasts.

    Joint selection optimizes complete fleet plans; the independent control
    uses the same candidates and predictions with independent decisions.
    """
    if (not isinstance(beam_width,int) or beam_width < 1 or not isfinite(risk_weight) or risk_weight < 0
            or not isfinite(preference_width) or preference_width <= 0):
        raise ValueError('Candidate selection requires a positive beam and width and a nonnegative risk weight.')
    if candidate_output is not None:
        if not isinstance(candidate_output,dict) or candidate_output:
            raise ValueError('The joint candidate output must be an empty dictionary.')
        if (selection not in ('joint','independent') or task_stopping or coupled_prediction
                or coupled_recovery or capture_weight or joint_feedback_topk or pareto_feedback_horizons):
            raise ValueError('Candidate export requires the factorized M1 planner.')
    if return_coefficients and not (task_stopping and task_feedback_prediction):
        raise ValueError('Exact fusion coefficients require the complete task-feedback planner.')
    if independent_feedback_proposal and not (task_stopping and task_feedback_prediction):
        raise ValueError('Independent completion requires the common complete-feedback candidate library.')
    if (nominal_feedback_proposals or feedback_refinement_rounds) and not (task_stopping and task_feedback_prediction):
        raise ValueError('Feedback proposal search requires complete task-feedback prediction.')
    if not isinstance(feedback_refinement_rounds,int) or feedback_refinement_rounds<0:
        raise ValueError('Feedback refinement rounds must be a nonnegative integer.')
    if (pareto_feedback_horizons or pareto_componentwise) and (not pareto_refinement or not refine_all_plans
                                    or selection != 'joint' or coupled_prediction or joint_feedback_topk):
        raise ValueError('Protected refinement requires the Pareto fleet proposal and its exact reference.')
    if (pareto_feedback_tail or pareto_task_guard) and not pareto_feedback_horizons:
        raise ValueError('Tail and task protection require explicit feedback forecast horizons.')
    if (candidate_subdivisions != 1 or include_uniform_midpoint) and (coupled_prediction or coupled_recovery):
        raise ValueError('The balanced coupled design uses the original three-level candidates.')
    if not isfinite(capture_weight) or capture_weight < 0:
        raise ValueError('Fleet interception weight must be finite and nonnegative.')
    if not isfinite(escape_weight) or escape_weight < 0 or (escape_weight and not task_stopping):
        raise ValueError('Radial escape protection requires the task-stopped objective and a nonnegative weight.')
    if (not isfinite(target_envelope_weight) or target_envelope_weight < 0.
            or (target_envelope_weight and not task_stopping)
            or (target_envelope_degrees and not target_envelope_weight)):
        raise ValueError('Target-future coverage requires task stopping and a positive interception weight.')
    if not isfinite(target_envelope_degrees) or not 0. <= target_envelope_degrees <= 90.:
        raise ValueError('The public target turning envelope must be within zero to ninety degrees.')
    if task_feedback_prediction and (not task_stopping or candidate_subdivisions!=1
                                    or include_uniform_midpoint
                                    or selection not in ('joint','single_reference','balanced_independent')):
        raise ValueError('Task feedback needs the shared nine-candidate joint or independent comparison.')
    if task_stopping and (coupled_prediction or coupled_recovery or capture_weight or joint_feedback_topk
                          or pareto_feedback_horizons):
        raise ValueError('Task-stopped fusion reuses the common factorized candidate paths.')
    if capture_weight and (coupled_prediction or coupled_recovery or pair_refinements
                           or joint_feedback_topk or pareto_refinement):
        raise ValueError('Fleet interception uses the shared factorized candidate forecast and objective.')
    if (selection in ('independent','single_reference','balanced_independent') and not source_preference
            and getattr(actor,'fusion_selection','joint')!=selection):
        raise ValueError('Independent actuator selection requires source_preference=True; '
            'or the preceding fusion selector must also be independent.')
    source=source_action[...,:2];rule=packet['node'][...,11:13]
    command=preferred_action[...,:2]*preferred_action[...,2:3]+rule*(1.-preferred_action[...,2:3])
    difference=source-rule
    preferred=torch.where(difference.abs()>1e-5,(command-rule)/difference.masked_fill(difference.abs()<=1e-5,1.),
                          source_action[...,2:3]).clamp(0.,1.)
    explicit=packet.get('_preferred_actuator_coefficients')
    if explicit is not None:
        # These coefficients are already the decision variables. Recovering
        # them from a mixed force divides by source minus rule, which is
        # ill-conditioned when the source policies produce similar commands.
        preferred=explicit
    else:
        native=(preferred_action[...,:2]==source).all(-1,keepdim=True)
        preferred=torch.where(native,preferred_action[...,2:3],preferred)
    interior=preferred if interior_fusion is None else torch.full_like(preferred,float(interior_fusion))
    candidates=actuator_candidate_grid(interior, candidate_subdivisions,
        include_uniform_midpoint=include_uniform_midpoint)
    if task_stopping:
        if task_feedback_prediction:
            _,order=fleet_context(packet,True,2)
            b,n=order.shape
            candidates=candidates.gather(1,order[...,None,None].expand_as(candidates))
            preferred=preferred.gather(1,order[...,None].expand_as(preferred))
            unary=((candidates-preferred[:,:,None])/preference_width).square().mean(-1)
            library,reference_start=feedback_fleet_library(unary)
            dt=min(.05,actor.prediction_step) if synchronized_prediction else actor.prediction_step
            target,velocity=actor.public_target_motion(packet)
            horizon=round((actor.risk_horizon if coupled_horizon is None else coupled_horizon)/dt)
            times=torch.arange(1,horizon+1,device=source.device)*dt
            central=(torch.stack([actor.target_motion.predict(packet,float(time))[0] for time in times],1)
                if actor.target_motion is not None else target[:,None]+times[None,:,None]*velocity[:,None])
            futures,weights=target_turning_envelope(target,central,target_envelope_degrees)
            scenarios=futures.shape[1]
            learner=getattr(actor,'feedback_fleet_policy',None)
            target_response=getattr(actor,'target_feedback',None)
            if target_response is not None and scenarios!=1:
                raise ValueError('Learned reactive target motion replaces the fixed turning envelope.')
            response_components=(tuple(target_response.component(index) for index in range(len(target_response.models)))
                if target_response is not None and getattr(target_response,'scenario_mode','mean')=='members'
                else (target_response,))
            if target_response is not None:
                if hasattr(target_response,'trajectory_components'):
                    response_components=target_response.trajectory_components()
                scenarios=len(response_components)
                probabilities=getattr(target_response,'scenario_weights',None)
                weights=(source.new_full((scenarios,),1./scenarios) if probabilities is None
                    else source.new_tensor(probabilities))
                kinematic_weight=getattr(target_response,'kinematic_weight',0.)
                if kinematic_weight:
                    # Preserve distinct fleet responses instead of averaging
                    # incompatible target accelerations before simulation.
                    learned=(response_components if kinematic_weight<1. else ())
                    response_components=(*learned,target_response.kinematic_component())
                    weights=torch.cat((weights[:len(learned)]*(1.-kinematic_weight),
                        source.new_tensor([kinematic_weight])))
                    scenarios=len(response_components)
            def forecast(choices):
                count=choices.shape[1]
                plans=candidates[:,None].expand(-1,count,-1,-1,-1).gather(3,
                    choices[...,None,None].expand(-1,-1,-1,1,2)).squeeze(3)
                physical=plans.gather(2,order.argsort(-1)[:,None,:,None].expand(-1,count,-1,2))
                if target_response is not None:
                    responses=[coupled_actuator_risk(actor,packet,source,physical,'task_paths',
                        synchronized_prediction,coupled_horizon,target_response=component)
                        for component in response_components]
                    paths=torch.stack([response[0] for response in responses],2)
                    plan_futures=torch.stack([response[1] for response in responses],2)
                elif scenarios==1:
                    paths=coupled_actuator_risk(actor,packet,source,physical,'paths',
                        synchronized_prediction,coupled_horizon)[:,:,None]
                else:
                    expanded={key:(value[:,None].expand(-1,scenarios,*value.shape[1:]).reshape(
                        b*scenarios,*value.shape[1:]) if torch.is_tensor(value) and value.ndim
                        and value.shape[0]==b else value) for key,value in packet.items()}
                    expanded_source=source[:,None].expand(-1,scenarios,-1,-1).reshape(b*scenarios,n,2)
                    expanded_plans=physical[:,None].expand(-1,scenarios,-1,-1,-1).reshape(b*scenarios,count,n,2)
                    paths=coupled_actuator_risk(actor,expanded,expanded_source,expanded_plans,'paths',
                        synchronized_prediction,coupled_horizon,
                        target_paths=futures.reshape(b*scenarios,horizon,2))
                    paths=paths.reshape(b,scenarios,count,n,horizon,2).permute(0,2,1,3,4,5)
                if target_response is None:
                    plan_futures=futures[:,None].expand(-1,count,-1,-1,-1)
                paths=paths.gather(3,order[:,None,None,:,None,None].expand_as(paths))
                factors=task_envelope_fusion_factors(paths.reshape(b*count,scenarios,n,1,horizon,2),
                    plan_futures.reshape(b*count,scenarios,horizon,2),
                    ((plans-preferred[:,None])/preference_width).square().mean(-1).reshape(b*count,n,1),
                    actor.distance_risk,weights=weights,interception_weight=target_envelope_weight,
                    control_stride=max(1,round(.2/dt)))
                if learner is not None and learner.scenario_evaluation:
                    scenario_costs=task_stopped_fleet_cost(factors['scenario_factors'],
                        torch.zeros((b*count*scenarios,1,n),dtype=torch.long,device=source.device),
                        risk_weight,escape_weight).reshape(b,count,scenarios)
                    mass=factors['scenario_weights']
                    costs=(scenario_costs*mass[None,None]).sum(-1)
                    features=learner.scenario_features(packet,source,preferred,plans,paths,
                        plan_futures,mass,order,target,scenario_costs)
                else:
                    costs=task_stopped_fleet_cost(factors,torch.zeros((b*count,1,n),
                        dtype=torch.long,device=source.device),risk_weight,escape_weight).reshape(b,count)
                    features=(learner.features(packet,source,preferred,plans,paths,plan_futures,weights,order,target)
                        if learner is not None else None)
                return costs,features
            scores,features=forecast(library)
            extra_search=independent_feedback_proposal or nominal_feedback_proposals or feedback_refinement_rounds
            if extra_search:
                local_scores=scores if learner is None else -learner(features,scores)[0]
                proposal=select_feedback_fleet_plan(local_scores,library,reference_start,'single_reference')
            proposals=[]
            if independent_feedback_proposal:
                # A simultaneous independent recombination need not occur in
                # the balanced library. Complete it with an actual feedback
                # forecast, available to both selectors at the same cost.
                # Contrast normalization uses the original common library so
                # adding this proposal cannot change the local query scores.
                proposals.append(proposal[:,None])
            if nominal_feedback_proposals:
                # Retain the nominal planner's coordination basins before
                # evaluating their complete teammate feedback. Nominal costs
                # propose plans but never select the feedback action.
                physical_candidates=candidates.gather(1,
                    order.argsort(-1)[...,None,None].expand_as(candidates))
                nominal_paths=actor.candidate_risk(dict(packet),source,
                    candidate_grid=physical_candidates,return_paths=True)
                nominal_paths=nominal_paths.gather(1,order[...,None,None,None].expand_as(nominal_paths))
                nominal_times=torch.arange(1,nominal_paths.shape[-2]+1,device=source.device)*actor.prediction_step
                nominal_target=(torch.stack([actor.target_motion.predict(packet,float(t))[0] for t in nominal_times],1)
                    if actor.target_motion is not None else target[:,None]+nominal_times[None,:,None]*velocity[:,None])
                nominal_futures,nominal_weights=target_turning_envelope(target,nominal_target,target_envelope_degrees)
                identical=(candidates[:,:,:,None]-candidates[:,:,None,:]).abs().amax(-1)<1e-6
                duplicate=(identical & torch.tril(torch.ones((9,9),dtype=torch.bool,device=source.device),-1)).any(-1)
                nominal_factors=task_envelope_fusion_factors(nominal_paths,nominal_futures,
                    unary.masked_fill(duplicate,float('inf')),actor.distance_risk,
                    weights=nominal_weights,interception_weight=target_envelope_weight,
                    control_stride=max(1,round(.2/actor.prediction_step)))
                for nominal_selection in ('joint','independent','single_reference'):
                    proposals.append(select_task_stopped_fusion(nominal_factors,selection=nominal_selection,
                        beam_width=beam_width,risk_weight=risk_weight,escape_weight=escape_weight)[:,None])
            if proposals:
                extra=torch.cat(proposals,1)
                extra_scores,extra_features=forecast(extra)
                library=torch.cat((library,extra),1)
                scores=torch.cat((scores,extra_scores),1)
                if learner is not None:features=torch.cat((features,extra_features),1)
            for _ in range(feedback_refinement_rounds):
                ranked=scores if learner is None else -learner(features,scores)[0]
                current=library[torch.arange(b,device=source.device),ranked.argmin(-1)]
                local=current[:,None,None].expand(-1,n,9,-1).clone()
                for vessel in range(n):local[:,vessel,:,vessel]=torch.arange(9,device=source.device)
                extra=local.flatten(1,2)
                extra_scores,extra_features=forecast(extra)
                library=torch.cat((library,extra),1)
                scores=torch.cat((scores,extra_scores),1)
                if learner is not None:features=torch.cat((features,extra_features),1)
            if learner is None:
                choice=select_feedback_fleet_plan(scores,library,reference_start,selection)
            else:
                choice=learner.choose(features,scores,library,reference_start,selection)
            if (extra_search and selection=='single_reference'
                    and not (learner is not None and learner.sample)):
                # Retain the exact local choices that proposed this plan;
                # a common logit offset from the extra forecast must not
                # perturb floating-point tie breaking in the control arm.
                choice=proposal
            gates=candidates.gather(2,choice[...,None,None].expand(-1,-1,1,2)).squeeze(2)
            gates=gates.gather(1,order.argsort(-1)[...,None].expand_as(gates))
            physical=torch.cat((gates*source+(1.-gates)*rule,torch.ones_like(source_action[...,2:3])),-1)
            return (physical,gates) if return_coefficients else physical
        paths = actor.candidate_risk(dict(packet), source, candidate_grid=candidates, return_paths=True)
        target, velocity = actor.public_target_motion(packet)
        times = torch.arange(1, paths.shape[-2]+1, device=source.device)*actor.prediction_step
        target_path = (torch.stack([actor.target_motion.predict(packet, float(time))[0] for time in times], 1)
            if actor.target_motion is not None else target[:, None]+times[None, :, None]*velocity[:, None])
        _, order = fleet_context(packet, True, 2)
        candidates = candidates.gather(1, order[..., None, None].expand_as(candidates))
        preferred = preferred.gather(1, order[..., None].expand_as(preferred))
        paths = paths.gather(1, order[..., None, None, None].expand_as(paths))
        unary = ((candidates-preferred[:, :, None])/preference_width).square().mean(-1)
        k = candidates.shape[2]
        identical = (candidates[:, :, :, None]-candidates[:, :, None, :]).abs().amax(-1) < 1e-6
        duplicate = (identical & torch.tril(torch.ones((k, k), dtype=torch.bool,
            device=source.device), -1)).any(-1)
        unary = unary.masked_fill(duplicate, float('inf'))
        if target_envelope_weight:
            target_paths, weights = target_turning_envelope(target, target_path, target_envelope_degrees)
            factors = task_envelope_fusion_factors(paths, target_paths, unary, actor.distance_risk,
                weights=weights, interception_weight=target_envelope_weight,
                control_stride=max(1, round(.2/actor.prediction_step)))
        else:
            factors = task_stopped_fusion_factors(paths, target_path, unary, actor.distance_risk,
                control_stride=max(1, round(.2/actor.prediction_step)))
        choices = select_task_stopped_fusion(factors, selection=selection, beam_width=beam_width,
            risk_weight=risk_weight, escape_weight=escape_weight)
        gates = candidates.gather(2, choices[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
        gates = gates.gather(1, order.argsort(-1)[..., None].expand_as(gates))
        return torch.cat((gates*source+(1.-gates)*rule, torch.ones_like(source_action[..., 2:3])), -1)
    if coupled_prediction:
        _,order=fleet_context(packet,True,2);b,n=order.shape
        candidates=candidates.gather(1,order[...,None,None].expand_as(candidates))
        preferred=preferred.gather(1,order[...,None].expand_as(preferred))
        design=actuator_joint_design(n,source.device,interaction_order);count=len(design)
        plans=candidates[:,None].expand(-1,count,-1,-1,-1).gather(3,
            design[None,:,:,None,None].expand(b,-1,-1,1,2)).squeeze(3)
        physical_plans=plans.gather(2,order.argsort(-1)[:,None,:,None].expand(-1,count,-1,2))
        risk,clearance=coupled_actuator_risk(actor,packet,source,physical_plans,'both',synchronized_prediction,
            coupled_horizon,tail_horizon=coupled_tail)
        score=((plans-preferred[:,None])/preference_width).square().mean(-1).sum(-1)+risk_weight*risk
        safe=clearance>=contact_distance
        # Feasible joint forecasts take priority over the soft clearance sum.
        # When no plan is feasible, maximize the worst predicted separation.
        score=score+torch.where(safe.any(-1,keepdim=True),1e6*(~safe).to(score.dtype),
            1e6*(clearance.amax(-1,keepdim=True)-clearance))
        if interaction_order==2:
            identical=(plans.flatten(2)[:,:,None]-plans.flatten(2)[:,None]).abs().amax(-1)<1e-6
            duplicate=(identical & torch.tril(torch.ones((count,count),dtype=torch.bool,device=source.device),-1)).any(-1)
        else:
            # Coalescing equal actuator candidates before sorting complete
            # plans avoids a quadratic fleet-by-plan distance tensor.
            equivalent=(candidates[:,:,:,None]-candidates[:,:,None]).abs().amax(-1)<1e-6
            labels=torch.arange(9,device=source.device)[None,None,None].expand_as(equivalent)
            canonical=labels.masked_fill(~equivalent,9).amin(-1)
            assignment=canonical[:,None].expand(-1,count,-1,-1).gather(3,
                design[None,:,:,None].expand(b,-1,-1,1)).squeeze(-1)
            codes=(assignment*(9**torch.arange(n,device=source.device))).sum(-1)
            ordered,permutation=codes.sort(dim=-1,stable=True)
            repeated=torch.cat((torch.zeros_like(ordered[:,:1],dtype=torch.bool),ordered[:,1:]==ordered[:,:-1]),-1)
            duplicate=torch.zeros_like(repeated).scatter(1,permutation,repeated)
        refining=~safe.any(-1)
        if coupled_refinements and bool(refining.any()):
            # A balanced library covers low-order effects, but its best fleet
            # plan need not be locally optimal under the actual coupled
            # feedback forecast. Refine several coordination basins, retaining
            # every evaluated complete plan for BOTH selection controls.
            seeds=score.masked_fill(duplicate,float('inf')).topk(min(4,count),largest=False).indices
            current=plans.gather(1,seeds[:,:,None,None].expand(-1,-1,n,2))
            plan_bank=[plans];risk_bank=[risk];clearance_bank=[clearance]
            for _ in range(int(coupled_refinements)):
                for vessel in range(n):
                    alternatives=current[:,:,None].expand(-1,-1,9,-1,-1).clone()
                    alternatives[:,:,:,vessel]=candidates[:,None,vessel]
                    alternatives=torch.where(refining[:,None,None,None,None],alternatives,current[:,:,None])
                    alternatives=alternatives.flatten(1,2)
                    physical_alternatives=alternatives.gather(2,
                        order.argsort(-1)[:,None,:,None].expand(-1,alternatives.shape[1],-1,2))
                    new_risk,new_clearance=coupled_actuator_risk(actor,packet,source,physical_alternatives,
                        'both',synchronized_prediction,coupled_horizon,tail_horizon=coupled_tail)
                    local=((alternatives-preferred[:,None])/preference_width).square().mean(-1).sum(-1)+risk_weight*new_risk
                    local=local.reshape(b,-1,9);separation=new_clearance.reshape(b,-1,9)
                    feasible=separation>=contact_distance
                    local+=torch.where(feasible.any(-1,keepdim=True),1e6*(~feasible).to(local.dtype),
                        1e6*(separation.amax(-1,keepdim=True)-separation))
                    selected=local.argmin(-1)
                    current=alternatives.reshape(b,-1,9,n,2).gather(2,
                        selected[:,:,None,None,None].expand(-1,-1,1,n,2)).squeeze(2)
                    plan_bank.append(alternatives);risk_bank.append(new_risk);clearance_bank.append(new_clearance)
            plans=torch.cat(plan_bank,1);risk=torch.cat(risk_bank,1);clearance=torch.cat(clearance_bank,1)
            count=plans.shape[1];safe=clearance>=contact_distance
            score=((plans-preferred[:,None])/preference_width).square().mean(-1).sum(-1)+risk_weight*risk
            score+=torch.where(safe.any(-1,keepdim=True),1e6*(~safe).to(score.dtype),
                1e6*(clearance.amax(-1,keepdim=True)-clearance))
            matches=(plans[:,:,:,None]-candidates[:,None]).abs().amax(-1)<1e-6
            assignment=matches.to(torch.long).argmax(-1)
            codes=(assignment*(9**torch.arange(n,device=source.device))).sum(-1)
            ordered,permutation=codes.sort(dim=-1,stable=True)
            repeated=torch.cat((torch.zeros_like(ordered[:,:1],dtype=torch.bool),ordered[:,1:]==ordered[:,:-1]),-1)
            duplicate=torch.zeros_like(repeated).scatter(1,permutation,repeated)
        if selection=='joint':
            choice=score.masked_fill(duplicate,float('inf')).argmin(-1)
            gates=plans[torch.arange(b,device=source.device),choice]
        elif selection=='independent':
            # Both selectors see the same jointly predicted fleet library.
            # This control chooses each vessel from its conditional marginal.
            matches=(plans[:,:,:,None]-candidates[:,None]).abs().amax(-1)<1e-6
            weights=matches & ~duplicate[:,:,None,None]
            expected=(score[:,:,None,None]*weights).sum(1)/weights.sum(1).clamp_min(1)
            choices=expected.argmin(-1)
            gates=candidates.gather(2,choices[...,None,None].expand(-1,-1,1,2)).squeeze(2)
        else:raise ValueError('Actuator candidate selection must be joint or independent.')
        gates=gates.gather(1,order.argsort(-1)[...,None].expand_as(gates))
        command=gates*source+(1.-gates)*rule
        physical=torch.cat((command,torch.ones_like(source_action[...,2:3])),-1)
        if expand_interactions and interaction_order==2:
            # Expand the complete fleet search only when the pair-balanced
            # library supplies no feasible feedback trajectory. Both joint
            # and independent selectors receive the same expanded library.
            expand=~safe.any(-1)
            if bool(expand.any()):
                subset={key:(value[expand] if torch.is_tensor(value) and value.ndim
                    and value.shape[0]==b else value) for key,value in packet.items()}
                physical[expand]=refine_actuator_interactions(actor,subset,source_action[expand],
                    preferred_action[expand],beam_width=beam_width,risk_weight=risk_weight,
                    preference_width=preference_width,selection=selection,source_preference=source_preference,
                    prior_map=prior_map,coupled_prediction=True,contact_distance=contact_distance,
                    synchronized_prediction=synchronized_prediction,
                    coupled_horizon=coupled_horizon,interaction_order=3,coupled_tail=coupled_tail,
                    coupled_refinements=coupled_refinements,interior_fusion=interior_fusion)
        return physical
    def interception_cost(paths):
        target, velocity = actor.public_target_motion(packet)
        times = torch.arange(1, paths.shape[-2]+1, device=source.device)*actor.prediction_step
        target_path = (torch.stack([actor.target_motion.predict(packet, float(time))[0] for time in times], 1)
            if actor.target_motion is not None else target[:, None]+times[None, :, None]*velocity[:, None])
        return ((paths-target_path[:, None, None]).norm(dim=-1).amin(-1)-5.).clamp_min(0.)/5.

    if selection=='single_reference':
        # Only the subject's candidate varies. Each neighbor follows its
        # single source-preferred trajectory; no neighbor candidate pair or
        # joint optimizer enters this comparison.
        paths=actor.candidate_risk(dict(packet),source,candidate_grid=candidates,return_paths=True)
        reference_index=candidate_subdivisions*(2*candidate_subdivisions+1+include_uniform_midpoint)+candidate_subdivisions
        reference=paths[:,:,reference_index]
        separation=(paths[:,:,None]-reference[:,None,:,None]).norm(dim=-1)
        risk=actor.distance_risk(separation.amin(-1))
        mask=~torch.eye(source.shape[1],dtype=torch.bool,device=source.device)
        risk=(risk*mask[None,:,:,None]).sum(2)
        unary=((candidates-preferred[:,:,None])/preference_width).square().mean(-1)
        if capture_weight:
            reach = interception_cost(paths)
            index = reference_index
            others = reach[:, :, index][:, None].expand(-1, source.shape[1], -1)
            others = others.masked_fill(~mask[None], float('inf')).amin(-1)
            unary = unary+capture_weight*torch.minimum(reach, others[..., None])
        choice=(unary+risk_weight*risk).argmin(-1)
        gates=candidates.gather(2,choice[...,None,None].expand(-1,-1,1,2)).squeeze(2)
        return torch.cat((gates*source+(1.-gates)*rule,torch.ones_like(source_action[...,2:3])),-1)
    forecast_packet = dict(packet)
    if capture_weight or candidate_output is not None:
        forecast_packet['_retain_candidate_paths'] = True
    risk=actor.candidate_risk(forecast_packet,source,pairwise=True,candidate_grid=candidates)
    reach = interception_cost(forecast_packet['_retained_candidate_paths']) if capture_weight else None
    _,order=fleet_context(packet,True,2)
    batch=torch.arange(len(source),device=source.device)
    candidates=candidates.gather(1,order[...,None,None].expand_as(candidates))
    preferred=preferred.gather(1,order[...,None].expand_as(preferred))
    if reach is not None:
        reach = reach.gather(1, order[..., None].expand_as(reach))
    risk=risk[batch[:,None,None],order[:,:,None],order[:,None,:]]
    unary=((candidates-preferred[:,:,None])/preference_width).square().mean(-1)
    b,n,k=unary.shape
    identical=(candidates[:,:,:,None]-candidates[:,:,None,:]).abs().amax(-1)<1e-6
    duplicate=(identical & torch.tril(torch.ones((k,k),dtype=torch.bool,device=source.device),-1)).any(-1)
    unary=unary.masked_fill(duplicate,float('inf'))
    if selection=='independent':
        probability=(-unary).softmax(-1)
        mask=~torch.eye(n,dtype=torch.bool,device=source.device)
        expected=torch.einsum('bijgh,bjh,ij->big',risk,probability,mask.to(source.dtype))
        cost = unary+risk_weight*expected
        if capture_weight:
            cost = cost+capture_weight*independent_fleet_minimum_cost(reach, probability)
        choices=cost.argmin(-1)
    elif selection=='joint':
        scores=source.new_zeros((b,1));plans=torch.empty((b,1,0),dtype=torch.long,device=source.device)
        pursuit = source.new_zeros((b, 1))
        for vessel in range(n):
            cost=scores[:,:,None]+unary[:,vessel,None]
            for other in range(vessel):
                pair=risk[:,other,vessel]
                cost=cost+risk_weight*pair.gather(1,plans[:,:,other,None].expand(-1,-1,k))
            if capture_weight:
                next_pursuit = reach[:, vessel, None].expand(-1, plans.shape[1], -1)
                if vessel:
                    next_pursuit = torch.minimum(pursuit[..., None], next_pursuit)
                cost = cost+capture_weight*(next_pursuit-(pursuit[..., None] if vessel else 0.))
            flat=cost.flatten(1);length=min(beam_width,flat.shape[-1])
            scores,index=flat.topk(length,largest=False)
            if capture_weight:
                pursuit = next_pursuit.flatten(1).gather(1, index)
            ancestor=index//k
            plans=torch.cat((plans.gather(1,ancestor[...,None].expand(-1,-1,vessel)),(index%k)[...,None]),-1)
        if candidate_output is not None:
            candidate_output['beam_plans']=plans.clone()
        if refine_all_plans:
            # Joint fusion is nonconvex. Refining only the cheapest beam
            # endpoint discards the other coordination basins already found.
            # Retain that endpoint and refine every distinct beam plan under
            # the SAME candidate-interaction objective before choosing one.
            proposal_plans = plans.clone() if joint_feedback_topk else None
            if pareto_refinement:
                if joint_feedback_topk:
                    raise ValueError('Pareto refinement compares the same factorized risk objective.')
                reference = plans[:, 0].clone()
                for _ in range(3):
                    for vessel in range(n):
                        cost = unary[:, vessel].clone()
                        for other in range(n):
                            if other == vessel:
                                continue
                            cost += risk_weight*risk[:, vessel, other].gather(
                                2, reference[:, other, None, None].expand(-1, k, 1)).squeeze(-1)
                        reference[:, vessel] = cost.argmin(-1)
            for _ in range(3):
                for vessel in range(n):
                    cost=unary[:,vessel,None].expand(-1,plans.shape[1],-1).clone()
                    for other in range(n):
                        if other==vessel:continue
                        cost+=risk_weight*risk[:,vessel,other].gather(2,
                            plans[:,None,:,other].expand(-1,k,-1)).transpose(1,2)
                    if capture_weight:
                        peer_reach = torch.stack([reach[:, other].gather(1, plans[:, :, other])
                            for other in range(n) if other != vessel], -1).amin(-1)
                        cost += capture_weight*torch.minimum(reach[:, vessel, None], peer_reach[..., None])
                    plans[:,:,vessel]=cost.argmin(-1)
            if pair_refinements:
                plans = refine_pair_blocks(unary, risk, plans, risk_weight, pair_refinements)
            energy=source.new_zeros(plans.shape[:2])
            for vessel in range(n):
                energy+=unary[:,vessel].gather(1,plans[:,:,vessel])
                for other in range(vessel):
                    code=plans[:,:,other]*k+plans[:,:,vessel]
                    energy+=risk_weight*risk[:,other,vessel].flatten(1).gather(1,code)
            if capture_weight:
                energy += capture_weight*torch.stack([reach[:, vessel].gather(1, plans[:, :, vessel])
                    for vessel in range(n)], -1).amin(-1)
            if joint_feedback_topk:
                # Factorized paths are inexpensive proposals. Re-evaluate the
                # most promising distinct maneuvers with all boats responding
                # to the same evolving fleet state and source-policy feedback.
                shortlist = distinct_fleet_shortlist(torch.cat((proposal_plans, plans), 1),
                    torch.cat((scores, energy), 1), joint_feedback_topk)
                proposed = candidates[:, None].expand(-1, shortlist.shape[1], -1, -1, -1).gather(
                    3, shortlist[..., None, None].expand(-1, -1, -1, 1, 2)).squeeze(3)
                physical_proposals = proposed.gather(2, order.argsort(-1)[:, None, :, None].expand(
                    -1, shortlist.shape[1], -1, 2))
                feedback_risk = coupled_actuator_risk(actor, packet, source, physical_proposals,
                    synchronized=True, horizon=joint_feedback_horizon)
                feedback_score = (((proposed-preferred[:, None])/preference_width).square().mean(-1).sum(-1)
                                  +risk_weight*feedback_risk)
                choices = shortlist[torch.arange(b, device=source.device), feedback_score.argmin(-1)]
            elif pareto_refinement:
                choices = pareto_fleet_refinement(
                    unary, risk, plans, reference, risk_weight, componentwise=pareto_componentwise)
                if pareto_feedback_horizons:
                    choices = feedback_guarded_refinement(
                        actor, packet, source, candidates, order, reference, choices, pareto_feedback_horizons,
                        tail_horizon=pareto_feedback_tail, task_guard=pareto_task_guard)
            else:
                choices=plans[torch.arange(b,device=source.device),energy.argmin(-1)]
        else:
            choices=plans[:,0].clone()
            for _ in range(3):
                for vessel in range(n):
                    cost=unary[:,vessel].clone()
                    for other in range(n):
                        if other==vessel:continue
                        cost+=risk_weight*risk[:,vessel,other].gather(2,
                            choices[:,other,None,None].expand(-1,k,1)).squeeze(-1)
                    if capture_weight:
                        peer_reach = torch.stack([reach[:, other].gather(1, choices[:, other, None]).squeeze(1)
                            for other in range(n) if other != vessel], -1).amin(-1)
                        cost += capture_weight*torch.minimum(reach[:, vessel], peer_reach[:, None])
                    choices[:,vessel]=cost.argmin(-1)
            if pair_refinements:
                choices = refine_pair_blocks(
                    unary, risk, choices[:, None], risk_weight, pair_refinements)[:, 0]
    else:raise ValueError('Actuator candidate selection must be joint or independent.')
    if candidate_output is not None:
        candidate_output.update(candidates=candidates,unary=unary,pair_risk=risk,
            order=order,reference=choices,
            refined_plans=(plans if selection=='joint' else choices[:,None]),
            prediction_paths=forecast_packet['_retained_candidate_paths'],source=source,rule=rule)
    gates=candidates.gather(2,choices[...,None,None].expand(-1,-1,1,2)).squeeze(2)
    gates=gates.gather(1,order.argsort(-1)[...,None].expand_as(gates))
    command=gates*source+(1.-gates)*rule
    physical=torch.cat((command,torch.ones_like(source_action[...,2:3])),-1)
    if coupled_recovery:
        # A long factorized forecast that still predicts contact after joint
        # optimization supplies no feasible continuation. Replan that state
        # with a shorter, mutually consistent feedback forecast.
        selected_risk=torch.stack([risk[batch,i,j,choices[:,i],choices[:,j]]
            for i in range(n) for j in range(i+1,n)],-1).amax(-1)
        recover=selected_risk>actor.distance_risk(source.new_tensor(5.))
        if recovery_distance is not None:
            # A short feedback forecast resolves an imminent encounter. It
            # cannot replace the long pursuit forecast merely because a
            # distant, uncertain trajectory pair crosses later in time.
            position=packet['motion'][...,:2]
            separation=(position[:,:,None]-position[:,None,:]).norm(dim=-1)
            separation=separation.masked_fill(torch.eye(n,dtype=torch.bool,device=source.device)[None],float('inf'))
            recover&=separation.flatten(1).amin(-1)<recovery_distance
        if bool(recover.any()):
            subset={key:(value[recover] if torch.is_tensor(value) and value.ndim and value.shape[0]==b else value)
                for key,value in packet.items()}
            physical[recover]=refine_actuator_interactions(actor,subset,source_action[recover],
                preferred_action[recover],beam_width=beam_width,risk_weight=risk_weight,
                preference_width=preference_width,selection=selection,source_preference=source_preference,
                prior_map=prior_map,coupled_prediction=True,contact_distance=contact_distance,
                synchronized_prediction=True,
                coupled_horizon=recovery_horizon,interaction_order=interaction_order,
                expand_interactions=expand_interactions,coupled_tail=coupled_tail,
                coupled_refinements=coupled_refinements,interior_fusion=interior_fusion)
    return physical
