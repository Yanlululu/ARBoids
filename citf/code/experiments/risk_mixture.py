"""Development controller with a registered physical/learned risk mixture."""
import argparse
from pathlib import Path
import sys

import torch
from torch import nn

from citf.risk import pair_indices
from citf.contracts import canonical_hash, file_hash


class RiskMixture(nn.Module):
    def __init__(self, model, alpha):
        super().__init__(); self.model=model; self.alpha=alpha; self.cfg=model.cfg

    def encode_history(self,b):
        return self.model.encode_history(b)

    def forward(self,b,history_cache=None):
        out=self.model(b,history_cache=history_cache)
        nom=b['nominal'];i,j=pair_indices(nom.shape[2]-1,nom.device)
        distance=(nom[:,:,i]-nom[:,:,j]).norm(dim=-1).masked_fill(~b['known_time_valid'][:,None,None,:],torch.inf).amin(-1)
        physical=(self.cfg.clearance-distance).clamp_min(0).square()
        out['risk']=self.alpha*out['risk']+(1-self.alpha)*physical
        out['mean_distance']=self.alpha*out['mean_distance']+(1-self.alpha)*distance
        out['det_risk']=(self.cfg.clearance-out['mean_distance']).clamp_min(0).square()
        return out


def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument('--mixture-alpha',type=float,required=True)
    ap.add_argument('--checkpoint',required=True)
    args, _=ap.parse_known_args()
    if not 0<=args.mixture_alpha<=1:
        raise ValueError('The physical/learned mixture must be a probability')
    # Keep the ordinary checkpoint option for the evaluation parser.
    index=sys.argv.index('--mixture-alpha');del sys.argv[index:index+2]
    import citf.collect as collection
    import citf.evaluate as evaluation
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    implementation={'mixture':file_hash(__file__)}
    if checkpoint['config']['revision']=='CITF-stopping-graph-v1':
        from experiments.stopping_graph import StoppingConfig,StoppingPredictor
        collection.Config=StoppingConfig;collection.Predictor=StoppingPredictor
        implementation['stopping_graph']=file_hash(Path(__file__).with_name('stopping_graph.py'))
    original_loader=collection.load_behavior
    def load(path,source,world,device):
        model=original_loader(path,source,world,device)
        return RiskMixture(model,args.mixture_alpha).to(device).eval()
    original_identity=evaluation.runtime_identity
    def identity(*positional,**kwargs):
        result=original_identity(*positional,**kwargs)
        result['method']='CITF-RiskMixture-development'
        result['controller']['learned_risk_probability']=args.mixture_alpha
        result['controller']['risk_distribution']='nominal point mass plus learned rectified Gaussian; trace mu/sigma describe the learned component'
        result['runtime_code']=canonical_hash({'core':result['runtime_code'],**implementation})
        return result
    evaluation.load_behavior=load;evaluation.runtime_identity=identity
    evaluation.main()


if __name__=='__main__':
    main()
