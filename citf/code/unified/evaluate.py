"""Complete-task evaluation with a fixed checkpoint and explicit scenario bank."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import torch

from .environment import TaskConfig
from .networks import CITF, ModelConfig
from .baselines import PublicBaseline
from .train import evaluate, write_json
from . import IMPLEMENTATION_VERSION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument('--out', required=True)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--scene-seed', type=int, default=73001)
    parser.add_argument('--policy-seed', type=int, default=83001)
    parser.add_argument('--defenders', type=int)
    parser.add_argument('--agility', type=float)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stage', choices=['development','formal'], default='development')
    parser.add_argument('--thrust-delta', type=float, help='Boids task constraint; learned policies retain their checkpoint value.')
    parser.add_argument('--collision-penalty', type=float, help='Boids task reward metadata; trained policies retain their checkpoint value.')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('high')
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f'{output} already contains an evaluation; keep each result under its own name.')
    model, method, task = None, 'boids', TaskConfig(thrust_delta=300. if args.thrust_delta is None else args.thrust_delta,
                                                  collision_penalty=50. if args.collision_penalty is None else args.collision_penalty)
    checkpoint_metadata = None
    checkpoint_compatibility = None
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        source_version = payload.get('implementation_version')
        compatible_r17 = (source_version == 'unified-teammate-response-20261107-r17'
                          and IMPLEMENTATION_VERSION == 'unified-nonlocal-response-20261107-r18'
                          and payload.get('task', {}).get('defenders') == 3
                          and 'method' not in payload and args.stage == 'development')
        if source_version != IMPLEMENTATION_VERSION and not compatible_r17:
            raise ValueError('Use the checkpoint source or a registered compatible three-defender development model.')
        if compatible_r17:
            checkpoint_compatibility = ('r17 three-defender training; r18 preserves that computation '
                                        'and adds nonlocal adverse responses for larger fleets')
        if args.stage == 'formal':
            if (payload['arguments']['stage'] != 'formal' or payload['transitions'] != 1000000
                    or payload['arguments']['steps'] != 1000000
                    or payload['arguments']['seed'] not in (2101, 2102, 2103, 2104, 2105)):
                raise ValueError('Formal tables use the final fixed-budget checkpoint of a formal run.')
            if args.scene_seed in (payload['arguments']['eval_seed'], payload['arguments']['seed']+100000):
                raise ValueError('Formal test scenes must be separate from training and development validation.')
        task = TaskConfig(**payload['task'])
        if args.thrust_delta is not None and args.thrust_delta != task.thrust_delta:
            raise ValueError('A learned policy must retain its training actuator constraint during evaluation.')
        if args.collision_penalty is not None and args.collision_penalty != task.collision_penalty:
            raise ValueError('A learned policy retains its recorded training reward in evaluation metadata.')
        if 'method' in payload:
            method = payload['method']
            model = PublicBaseline(method, payload['width'], task.thrust_delta)
        else:
            method = 'citf'
            model = CITF(ModelConfig(**payload['config']))
            if not model.cfg.action_interaction:
                method += '-no-action-interaction'
            if model.cfg.feedback == 0:
                method += '-no-generator-feedback'
        model.to(args.device).load_state_dict(payload['model'])
        model.eval()
        checkpoint_metadata = dict(transitions=payload['transitions'], seed=payload['arguments']['seed'],
                                   implementation_version=payload['implementation_version'])
        del payload
    if args.defenders is not None:
        task = replace(task, defenders=args.defenders)
    if args.agility is not None:
        task = replace(task, agility=args.agility)
    if args.stage == 'formal':
        expected_policy_seed = 84001 if model is None else 84001+checkpoint_metadata['seed']-2101
        main = args.scene_seed == 74001 and task.defenders == 3 and task.agility == 2.
        agility = (args.scene_seed == 74004 and task.defenders == 3
                   and task.agility in (1.5, 1.75, 2., 2.25, 2.5, 2.75, 3.))
        team = args.scene_seed == 74005 and task.defenders in range(2, 8) and task.agility == 2.
        if args.count != 100 or args.policy_seed != expected_policy_seed or not (main or agility or team):
            raise ValueError('Use the registered formal scenario bank, policy seed, task and 100 episodes.')
        if method.startswith('citf-no-') and not main:
            raise ValueError('The registered two ablations are evaluated only in the main condition.')
    result = evaluate(model, task, args.count, args.scene_seed, args.policy_seed, args.device)
    result.update(method=method, task=asdict(task), checkpoint=args.checkpoint,
                  checkpoint_metadata=checkpoint_metadata,
                  checkpoint_compatibility=checkpoint_compatibility,
                  inference_model_config=asdict(model.cfg) if isinstance(model, CITF) else None,
                  scene_seed=args.scene_seed, policy_seed=args.policy_seed, research_stage=args.stage,
                  code_directory=str(Path(__file__).resolve().parents[1]))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print({key:value for key,value in result.items() if key != 'rows'}, flush=True)


if __name__ == '__main__':
    main()
