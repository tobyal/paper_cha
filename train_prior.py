"""Optional geometry-prior pretraining; complete surface targets are used HERE only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ssr.support import SurfaceSupportLearner
from ssr.geometry import normalize, patch_layout, directed_distance, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, help='JSON train list; samples (or cache) NPZ has p [N,3] and gt [M,3]')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--initialization', choices=['pretrained', 'random'], default='pretrained')
    p.add_argument('--checkpoint', help='Optional complete support-prior checkpoint')
    p.add_argument('--decoder', choices=['query', 'direct'], default='query')
    p.add_argument('--no-deform', action='store_true')
    p.add_argument('--points', type=int, default=1024)
    p.add_argument('--seed', type=int, default=21)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = Path(args.manifest).resolve()
    cases = json.loads(manifest.read_text())['train']
    if not cases:
        raise ValueError('Empty training split')
    learner = SurfaceSupportLearner(initialization=args.initialization, checkpoint=args.checkpoint,
                                    decoder=args.decoder, deform=not args.no_deform).cuda().train()
    optimizer = torch.optim.Adam(learner.parameters(), lr=args.lr)
    best, stale, history = float('inf'), 0, []
    write_json(output / 'config.json', {'args': vars(args), 'initial': learner.provenance, 'training_cases': len(cases),
                                      'validation_used': False, 'test_used': False,
                                      'prior_scope': 'complete encoding + intermediate representation + decoding',
                                      'objective': 'surface CD plus original kernel regularization; no field feedback'})
    for epoch in range(1, args.epochs + 1):
        total = 0.
        for i in rng.permutation(len(cases)):
            path = Path(cases[i].get('samples') or cases[i]['cache'])
            if not path.is_absolute():
                path = manifest.parent / path
            with np.load(path) as data:
                sparse = data['p']
                if len(sparse) < args.points:
                    raise ValueError(f'{path} has {len(sparse)} observations, needs {args.points}; prepare the input cache first')
                sparse = sparse[rng.choice(len(sparse), args.points, replace=False)]
                pts, center, scale = normalize(sparse)
                gt = (data['gt'] - center) / scale
            gt = gt[rng.choice(len(gt), min(8192, len(gt)), replace=False)]
            # Shared rotation augmentation; target and sparse input stay aligned.
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            points = torch.tensor(pts @ rotation, dtype=torch.float32, device='cuda')
            surface = torch.tensor(gt @ rotation, dtype=torch.float32, device='cuda')
            prediction = learner(patch_layout(points))
            support = prediction['points']
            support = support[torch.randperm(len(support), device='cuda')[:4096]]
            loss = .5 * (directed_distance(support, surface) + directed_distance(surface, support)) + .001 * prediction['regularizer']
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in learner.parameters()):
                raise FloatingPointError('Nonfinite prior training objective/gradient')
            torch.nn.utils.clip_grad_norm_(learner.parameters(), 10.)
            optimizer.step()
            total += float(loss.detach())
        mean = total / len(cases)
        history.append({'epoch': epoch, 'loss': mean})
        write_json(output / 'training.json', history)
        torch.save({'backbone': learner.backbone.state_dict(), 'support_network': learner.state_dict(),
                    'support_config': {'decoder': args.decoder, 'deform': not args.no_deform},
                    'epoch': epoch, 'optimizer': optimizer.state_dict()}, output / 'last.pt')
        print(history[-1], flush=True)
        if mean < best * .997:
            best, stale = mean, 0
        else:
            stale += 1
        if epoch >= 10 and stale >= args.patience:
            break
    write_json(output / 'status.json', {'status': 'complete', 'epochs': epoch,
                                      'stop': 'training_loss_plateau' if epoch < args.epochs else 'epoch_budget'})


if __name__ == '__main__':
    main()
