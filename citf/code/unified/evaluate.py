"""Complete-task evaluation with a fixed checkpoint and explicit scenario bank."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import torch

from .environment import TaskConfig
from .networks import CITF, ModelConfig
from .baselines import PublicBaseline
from .train import evaluate, write_json
from .protocol import (load_registry, validate_checkpoint, checkpoint_identity, method_of,
                       evaluation_identity, attach_identity, validate_result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument('--registry', help='Frozen registry required for formal evaluation.')
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
    if args.stage == 'formal' and not args.registry:
        raise ValueError('Formal evaluation requires the common study registry.')
    registry = load_registry(args.registry) if args.registry else None
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('high')
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f'{output} already contains an evaluation; keep each result under its own name.')
    model, method, task = None, 'boids', TaskConfig(thrust_delta=300. if args.thrust_delta is None else args.thrust_delta,
                                                  collision_penalty=50. if args.collision_penalty is None else args.collision_penalty)
    checkpoint_metadata = None
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        method = method_of(payload['arguments'])
        if args.stage == 'formal':
            validate_checkpoint(payload, registry, method, payload['arguments']['seed'], 1000000)
        else:
            validate_checkpoint(payload)
        checkpoint_metadata = checkpoint_identity(args.checkpoint, payload)
        task = TaskConfig(**payload['task'])
        if args.thrust_delta is not None and args.thrust_delta != task.thrust_delta:
            raise ValueError('A learned policy must retain its training actuator constraint during evaluation.')
        if args.collision_penalty is not None and args.collision_penalty != task.collision_penalty:
            raise ValueError('A learned policy retains its recorded training reward in evaluation metadata.')
        if 'method' in payload:
            method = payload['method']
            model = PublicBaseline(method, payload['width'], task.thrust_delta)
        else:
            model = CITF(ModelConfig(**payload['config']))
        model.to(args.device).load_state_dict(payload['model'])
        model.eval()
        training_seed = payload['arguments']['seed']
        del payload
    if args.defenders is not None:
        task = replace(task, defenders=args.defenders)
    if args.agility is not None:
        task = replace(task, agility=args.agility)
    identity = None
    if args.stage == 'formal':
        identity = evaluation_identity(registry, method, None if model is None else training_seed,
                                       checkpoint_metadata, asdict(task), args.scene_seed, args.policy_seed, args.count)
    result = evaluate(model, task, args.count, args.scene_seed, args.policy_seed, args.device)
    result.update(method=method, task=asdict(task), checkpoint=args.checkpoint,
                  checkpoint_metadata=checkpoint_metadata,
                  inference_model_config=asdict(model.cfg) if isinstance(model, CITF) else None,
                  scene_seed=args.scene_seed, policy_seed=args.policy_seed, research_stage=args.stage,
                  code_directory=str(Path(__file__).resolve().parents[1]))
    if identity is not None:
        attach_identity(result, identity)
        validate_result(result, identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print({key:value for key,value in result.items() if key != 'rows'}, flush=True)


if __name__ == '__main__':
    main()
