"""Zero-future-disturbance nominal integration matching the archived WAM-V.

Input velocity is the updated, observable ground velocity. It is treated as the
initial nominal velocity, not read from simulator-private future water/current.
"""
import torch

def step(state,thrust,dt=.2,inner=.05,*,return_legacy_velocity=False):
    """Advance measured state; optionally retain the source's pre-final-update velocity."""
    if state.shape[-1]!=6 or thrust.shape!=state.shape[:-1]+(2,):raise ValueError('State [...,6], thrust [...,2]')
    if abs(round(dt/inner)*inner-dt)>1e-8:raise ValueError('Nonintegral integration steps')
    p=state[...,:2];yaw=state[...,2];vel=state[...,3:6]
    legacy_velocity=vel[...,:2]
    total_thrust=thrust.sum(-1)
    yaw_moment=1.25*(thrust[...,0]-thrust[...,1])
    for _ in range(round(dt/inner)):
        legacy_velocity=vel[...,:2]
        p=p+inner*vel[...,:2];yaw=torch.remainder(yaw+inner*vel[...,2],2*torch.pi)
        c,s=yaw.cos(),yaw.sin();u=c*vel[...,0]+s*vel[...,1];v=-s*vel[...,0]+c*vel[...,1];r=vel[...,2]
        du=(400*r*v-(100+150*u.abs())*u+total_thrust)/380
        dv=(-380*r*u-(100+150*v.abs())*v)/400
        dr=(-20*u*v-(980+950*r.abs())*r+yaw_moment)/1430
        u=u+inner*du;v=v+inner*dv;r=r+inner*dr
        vel=torch.stack((c*u-s*v,s*u+c*v,r),-1)
    result=torch.cat((p,yaw[...,None],vel),-1)
    return (result,legacy_velocity) if return_legacy_velocity else result

def forecast(state,thrust,steps=16,dt=.2):
    result=[];x=state
    for _ in range(steps):x=step(x,thrust,dt);result.append(x[...,:2])
    return torch.stack(result,-2)
