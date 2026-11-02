
"""Fast, measured-state nominal motion. No future source calls or private state."""
import torch
from ci_core import actuator_candidate_grid,public_return_context
from vector_world import physical_step
@torch.no_grad()
def nominal_from_inputs(history,positions,candidate,steps=16,initial_motion=None):
 # history is in a target-centred rotated frame; its last state is observed.
 b,c,e,_=candidate.shape;n=e-1;last=history[:,:,-1]
 yaw=torch.atan2(last[...,4],last[...,5])
 prev=torch.atan2(history[:,:,-2,4],history[:,:,-2,5])
 r=torch.atan2(torch.sin(yaw-prev),torch.cos(yaw-prev))/.2
 vel=torch.cat((last[...,2:4]*5,r[...,None]),-1)
 motion=torch.cat((positions,yaw[...,None],vel),-1) if initial_motion is None else initial_motion
 state=torch.cat((motion,motion[...,3:6]),-1)[:,None].expand(b,c,e,9).reshape(b*c,e,9).double().clone()
 force=((candidate[...,:n,6:8]*1500+500)/2).reshape(b*c,n,2).double()
 path=[]
 for k in range(steps):
  defenders=physical_step(state[:,:n],force,torch.zeros_like(state[:,:n,3:6]))
  enemy=state[:,n:].clone();enemy[...,:3]+=.2*enemy[...,6:9]
  state=torch.cat((defenders,enemy),1);path.append(state[...,:2].float())
 return torch.stack(path,-2).reshape(b,c,e,steps,2)

@torch.no_grad()
def fast_factorize(actor,packet):
 source=actor.source(packet);preferred=source[...,2:3].expand(-1,-1,2);grid=actuator_candidate_grid(preferred)
 b,n,k,_=grid.shape
 # Derive observed position, heading and public velocity exactly as original CI.
 dynamics=actor.prediction_motion(packet).to(source)
 # prediction_motion returns px,py,theta,groundvx,groundvy,groundr; pad public velocity.
 if dynamics.shape[-1]!=6:raise ValueError(dynamics.shape)
 state=torch.cat((dynamics,dynamics[...,3:6]),-1)[:,:,None].expand(b,n,k,9).reshape(b*n*k,9).double()
 cmd=grid*source[:,:,:2,None].transpose(-1,-2) if False else grid*source[:,:,:2][:,:,None]+(1-grid)*packet['node'][...,11:13][:,:,None]
 force=((cmd*1500+500)/2).reshape(b*n*k,2).double()
 paths=[]
 for step in range(16):
  state=physical_step(state,force,torch.zeros_like(state[...,3:6]));paths.append(state[...,:2].float())
 path=torch.stack(paths,-2).reshape(b,n,k,16,2)
 dd=torch.linalg.vector_norm(path[:,:,None,:,None]-path[:,None,:,None,:],dim=-1).amin(-1)
 risk=(7-dd).clamp_min(0).square()
 _,order=public_return_context(packet,True,2);batch=torch.arange(b)
 sorted_grid=grid.gather(1,order[...,None,None].expand_as(grid));center=preferred.gather(1,order[...,None].expand_as(preferred))
 risk=risk[batch[:,None,None],order[:,:,None],order[:,None,:]]
 unary=((sorted_grid-center[:,:,None])/.4).square().mean(-1)
 dup=(sorted_grid[:,:,:,None]-sorted_grid[:,:,None,:]).abs().amax(-1)<1e-6
 dup=(dup&torch.tril(torch.ones(k,k,dtype=torch.bool),-1)).any(-1);unary=unary.masked_fill(dup,float('inf'))
 return dict(source=source,grid=grid,grid_sorted=sorted_grid,unary=unary,risk=risk,order=order,paths=path.repeat_interleave(2,dim=-2),weight=32.)
