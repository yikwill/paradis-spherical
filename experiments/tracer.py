"""Figure 3 reconstruction on the full 721 x 1440 sphere.

Both models start randomly and minimize area-weighted MSE for one fixed 68-degree
transport. Optional angular continuation is a separate, explicitly labelled run.
"""
import argparse
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn
import torch.nn.functional as F

from experiments.common import compact_config
from model.padding import GeoCyclicPadding
from model.paradis import Paradis


def tracer_fields(nlat=721, nlon=1440, angle=68):
    lat = torch.linspace(-math.pi/2, math.pi/2, nlat, dtype=torch.float64)
    lon = torch.arange(nlon, dtype=torch.float64)*2*math.pi/nlon
    phi, lam = torch.meshgrid(lat, lon, indexing="ij")
    xyz = torch.stack([phi.cos()*lam.cos(), phi.cos()*lam.sin(), phi.sin()])
    bearing, rotation = math.pi/4, math.radians(angle)
    center = torch.tensor([math.cos(rotation), math.sin(rotation)*math.sin(bearing),
                           math.sin(rotation)*math.cos(bearing)], dtype=torch.float64)
    def gaussian(center):
        distance = torch.acos(torch.einsum("chw,c->hw",xyz,center).clamp(-1,1))
        return torch.exp(-.5*(distance/math.radians(5))**2).float()[None,None]
    initial = gaussian(torch.tensor([1.,0.,0.],dtype=torch.float64))
    target = gaussian(center)
    # Exact areas of the latitude cells, including half-cells at the poles.
    edges = torch.cat([lat[:1], (lat[:-1]+lat[1:])/2, lat[-1:]])
    weights = edges.sin().diff()
    weights = (weights/weights.mean()).float()[None,None,:,None]
    return initial,target,phi.float(),lam.float(),weights


def conv_block(a,b):
    return nn.Sequential(GeoCyclicPadding(1),nn.Conv2d(a,b,3),nn.SiLU(),
                         GeoCyclicPadding(1),nn.Conv2d(b,b,3),nn.SiLU())


class UNet(nn.Module):
    """Depth-three U-Net, top width four; reconstructed from the paper description."""
    def __init__(self):
        super().__init__()
        widths=[4,8,16,32]
        self.encoders=nn.ModuleList([conv_block(1 if i==0 else widths[i-1],w) for i,w in enumerate(widths)])
        self.decoders=nn.ModuleList([conv_block(widths[i+1]+widths[i],widths[i]) for i in (2,1,0)])
        self.output=nn.Conv2d(4,1,1)

    def forward(self,x):
        skips=[]
        for i,block in enumerate(self.encoders):
            if i:
                x=F.avg_pool2d(x,2)
            x=block(x)
            skips.append(x)
        for block,skip in zip(self.decoders,reversed(skips[:-1])):
            x=F.interpolate(x,size=skip.shape[-2:],mode="bilinear",align_corners=True)
            x=block(torch.cat([x,skip],1))
        return self.output(x)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",choices=["paradis","unet"],required=True)
    parser.add_argument("--steps",type=int,default=500)
    parser.add_argument("--seconds",type=float,default=7200)
    parser.add_argument("--threads",type=int,default=8)
    parser.add_argument("--continuation-steps",type=int,default=0)
    parser.add_argument("--identity-steps",type=int,default=0)
    parser.add_argument("--output",default="/home/yikwill/paradis-data/tracer")
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    initial,target,lat,lon,weights=tracer_fields()
    if args.model=="paradis":
        cfg=compact_config("disco",8,8)
        cfg.model.coarsening_factor=2
        cfg.dataset.n_time_inputs=1
        cfg.features.input.constants=["sin_latitude"]
        dm=SimpleNamespace(dataset=SimpleNamespace(num_in_dyn_features=1,num_in_static_features=1),
                           num_common_features=1,num_out_features=1)
        model=Paradis(dm,cfg,lat,lon)
        inputs=torch.cat([initial,lat.sin()[None,None]],1)
    else:
        model=UNet()
        inputs=initial
        cfg=None
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=0)
    directory=Path(args.output)
    directory.mkdir(parents=True,exist_ok=True)
    label=args.model+('-continuation' if args.continuation_steps else '')
    if args.identity_steps:
        if not args.continuation_steps:
            raise ValueError('Identity warmup requires angular continuation')
        label+=f'-identity{args.identity_steps}'
    start=time.perf_counter()
    history=[]
    for step in range(args.steps):
        angle=68*min(1,max(0,step+1-args.identity_steps)/args.continuation_steps) if args.continuation_steps else 68
        current=tracer_fields(angle=angle)[1] if angle!=68 else target
        optimizer.zero_grad(set_to_none=True)
        prediction=model(inputs)
        loss=((prediction-current).square()*weights).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite tracer loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        elapsed=time.perf_counter()-start
        row=dict(step=step+1,angle=angle,loss=loss.item(),seconds=elapsed)
        history.append(row)
        if step%10==0:
            print(label,json.dumps(row),flush=True)
        if elapsed>=args.seconds:
            break
    with torch.inference_mode():
        prediction=model(inputs)
        mse=((prediction-target).square()*weights).mean().item()
        zero_mse=(target.square()*weights).mean().item()
        mass_error=((prediction*weights).sum()/(target*weights).sum()-1).item()
    result=dict(model=label,grid=[721,1440],parameters=sum(p.numel() for p in model.parameters()),
                steps=len(history),elapsed_seconds=time.perf_counter()-start,area_weighted_mse=mse,
                zero_predictor_mse=zero_mse,relative_mse=mse/zero_mse,relative_mass_error=mass_error,
                prediction_min=prediction.min().item(),prediction_max=prediction.max().item(),
                history=history,arguments=vars(args),seed=42,
                assumptions=dict(amplitude=1,sigma_degrees=5,initial_center=[0,0],bearing_degrees=45,rotation_degrees=68),
                protocol="fixed target" if not args.continuation_steps else "angular continuation then fixed target")
    np.savez_compressed(directory/f'{label}.npz',initial=initial.numpy()[0,0],target=target.numpy()[0,0],prediction=prediction.numpy()[0,0])
    torch.save(dict(model=model.state_dict(),config=OmegaConf.to_container(cfg,resolve=True) if cfg else None),directory/f'{label}.pt')
    Path(f'report/results/tracer-{label}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)


if __name__=='__main__':
    main()
