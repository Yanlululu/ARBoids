"""VRX transport adapter for the frozen committed-feedback joint controller."""
import argparse
import json
from pathlib import Path
import sys
import time

import run_gate_contribution as source
from run_gate_contribution_sync_start import SynchronizedStartTrial
from feedback_joint_control import FeedbackJointController, Measurement
from source_arboids import sha256

np,torch=source.np,source.torch
MODE='feedback_joint'
BLOCK_STEPS=10
ENGINE='reference'


class FeedbackTrial(SynchronizedStartTrial):
    def __init__(self,args):
        super().__init__(args)
        self.feedback_warmup_seconds=0.
        if MODE=='feedback_joint':
            started=time.monotonic()
            rule_class=FeedbackJointController
            if ENGINE=='vectorized':
                from feedback_joint_fast import FastFeedbackJointController
                rule_class=FastFeedbackJointController
            self.rule=rule_class(args.num_robots-1,self.policy,BLOCK_STEPS,
                                 velocity_lag=0.,attacker_agility=2.25)
            n=args.num_robots-1
            states=np.zeros((n,6))
            states[:,0]=np.arange(n)*20.
            attacker=np.array([120.,30.,np.pi,0.,0.,0.])
            measurement=self.measurement_from_states(states,attacker,0.)
            self.rule.control(measurement)
            self.rule.reset()
            self.feedback_warmup_seconds=time.monotonic()-started

    def policy(self,observations):
        with torch.no_grad():
            raw,_=self.actor(torch.tensor(observations,dtype=torch.float,device=self.device),True,False)
        return raw.cpu().numpy()

    def measurement_from_states(self,defenders,attacker,elapsed):
        boids,states=source.AUTHOR.Boids_navi_control(defenders[:,:2],defenders[:,3:5],
                                                    defenders[:,2],attacker[:2])
        observations=self.get_observations(defenders[:,:2],defenders[:,2],attacker[:2],
                                            attacker[3:5],states,boids)
        return Measurement(defenders,attacker,observations,boids,float(elapsed),self.defend_r,self.total_time)

    def control(self,elapsed):
        if MODE=='cbf':
            return super().control(elapsed)
        started=time.monotonic()
        limits=np.array([-500.,1000.])*self.agility
        actions=np.zeros((self.num_robots,2))
        actions[0]=source.AUTHOR.APF_navi_control(self.curr_pos[0],np.zeros(2),self.curr_pos[1:],
                                                self.curr_phi[0],*limits)
        states=np.column_stack((self.curr_pos,self.curr_phi,self.curr_vel,self.yaw_rates))
        measurement=self.measurement_from_states(states[1:].copy(),states[0].copy(),elapsed)
        inference_started=time.monotonic()
        actions[1:],info=self.rule.control(measurement)
        inference_seconds=time.monotonic()-inference_started
        if not all(np.isfinite(a).all() for a in (actions,states)):
            raise FloatingPointError('Non-finite feedback controller input/output.')
        physical=actions[:,::-1]
        for publisher,thrust in zip(self.publishers,physical.flat):
            publisher.publish(source.Float64(data=float(thrust)))
        predicted=info['predicted_next_state']
        self.rows.append(dict(timestamp=float(elapsed),AttPos=self.curr_pos[0].copy(),
            AttPhi=self.curr_phi[0].copy(),AttVel=self.curr_vel[0].copy(),AttAct=actions[0].copy(),
            DefPos=self.curr_pos[1:].flatten().copy(),DefPhi=self.curr_phi[1:].copy(),
            DefVel=self.curr_vel[1:].flatten().copy(),DefAct=actions[1:].flatten().copy(),
            PhysicalThrust=physical.flatten().copy(),GateTheta=np.full(self.num_robots-1,np.nan),
            SimulationStamp=float(np.min(self.last_stamp)),GateWarning=bool(info['warning']),
            CBFViolation=float(info['cbf_constraint_violation']),
            CBFSaturation=float(info.get('cbf_control_saturation',0.)),
            Planned=bool(info['planned']),CandidatePolicies=info['candidate_policies'],
            PlanTemplate=info['template'],
            PredictedNextDefPos=(np.full((self.num_robots-1)*2,np.nan) if predicted is None
                                 else predicted[:,:2].flatten()),
            ControlWallSeconds=time.monotonic()-started,PolicyWallSeconds=inference_seconds,
            FeedbackSkewSeconds=float(np.ptp(self.last_stamp))))


def main():
    global MODE,BLOCK_STEPS,ENGINE
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--method',choices=['feedback_joint','cbf'],default='feedback_joint')
    parser.add_argument('--frozen-method',type=Path,required=True)
    parser.add_argument('--prediction-engine',choices=['reference','vectorized'],default='reference')
    extra,remaining=parser.parse_known_args()
    frozen=json.loads(extra.frozen_method.read_text(encoding='utf-8'))
    MODE=extra.method
    ENGINE=extra.prediction_engine
    BLOCK_STEPS=int(frozen['selected']['block_steps'])
    root=Path(__file__).resolve().parents[1]
    for name,expected in frozen['code'].items():
        if sha256(root/'train'/name)!=expected:
            raise RuntimeError('Frozen numerical controller changed: '+name)
    files=[Path(__file__),Path(__file__).with_name('run_gate_contribution_sync_start.py'),
           Path(source.runner.__file__),root/'train/feedback_joint_control.py']
    if ENGINE=='vectorized':
        fast_path=root/'train/feedback_joint_fast.py'
        audit=json.loads((extra.frozen_method.parent/'acceleration_summary.json').read_text())
        if not audit['passed'] or audit['fast_code_sha256']!=sha256(fast_path):
            raise RuntimeError('Vectorized predictor has not passed equivalence checks.')
        files.append(fast_path)
    hashes={p.name:sha256(p) for p in files}
    # Both methods use the same original source constructor and CBF warmup.
    source.SourceTrial=FeedbackTrial
    sys.argv=[sys.argv[0],'--method','cbf',*remaining]
    status=source.main()
    if source.ACTIVE_ARGS is not None:
        args=source.ACTIVE_ARGS
        path=args.output_dir.resolve()/args.run_id/'result.json'
        result=json.loads(path.read_text())
        unchanged=hashes=={p.name:sha256(p) for p in files}
        result.update(method=MODE,block_steps=BLOCK_STEPS,prediction_engine=ENGINE,synchronized_model_start=True,
            feedback_adapter_code=hashes,feedback_code_unchanged=unchanged,
            numerical_checkpoint_sha256=frozen['checkpoint_sha256'],
            velocity_model='Public ground velocity; zero numerical-cache correction',
            prediction_model='Original numerical WAMV nominal model; original VRX APF/physics execute externally',
            control_contract='Commit policy parameters for 2 s; recompute actor/Boids/joint physical QP every 0.2 s')
        if not unchanged:
            result.update(passed=False,error='Feedback adapter changed during trial.')
        path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        status=0 if result['passed'] else 1
        print(f'[FEEDBACK VERIFIED] {MODE} passed={result["passed"]}',flush=True)
    return status


if __name__=='__main__':
    raise SystemExit(main())
