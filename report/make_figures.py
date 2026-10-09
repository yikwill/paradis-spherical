"""Generate report figures and numerical tables from recorded experiment outputs."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'report/results'
FIGURES = ROOT/'report/figures'
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 120, 'savefig.dpi': 180})


def read(name):
    return json.loads((RESULTS/name).read_text())


def tracer(directory):
    target=np.load(directory/'unet-residual.npz')['target'].astype(np.float64)
    lat=np.linspace(-np.pi/2,np.pi/2,target.shape[0])
    edges=np.r_[lat[0],(lat[:-1]+lat[1:])/2,lat[-1]]
    weights=np.diff(np.sin(edges))[:,None]
    denominator=weights.sum()*target.shape[1]
    mean=float((target*weights).sum()/denominator)
    zero=float((target**2*weights).sum()/denominator)
    constant=float(((target-mean)**2*weights).sum()/denominator)
    baselines=dict(optimal_constant_amplitude=mean,zero_mse=zero,constant_mse=constant,
                   constant_over_zero=constant/zero)
    (RESULTS/'tracer-baselines.json').write_text(json.dumps(baselines,indent=2)+'\n')
    for suffix in ('', '-continuation', '-continuation-identity300'):
        names = ['unet-residual'+suffix, 'paradis'+suffix]
        if not all((directory/f'{name}.npz').exists() for name in names):
            continue
        fields = [np.load(directory/f'{name}.npz') for name in names]
        metrics = [read(f'tracer-{name}.json') for name in names]
        maps = [fields[0]['initial'], fields[0]['target'], fields[0]['prediction'], fields[1]['prediction']]
        labels = ['Initial Gaussian', 'Analytic target: 68°',
                  f'Residual U-Net: relative MSE {metrics[0]["relative_mse"]:.3f}',
                  f'Full compact PARADIS: relative MSE {metrics[1]["relative_mse"]:.3f}']
        nlat,nlon = maps[0].shape
        latitude = np.linspace(-np.pi/2, np.pi/2, nlat)
        longitude = (np.arange(nlon+1)-nlon//2)*2*np.pi/nlon
        lat_edges = np.r_[-np.pi/2, (latitude[:-1]+latitude[1:])/2, np.pi/2]
        lon_edges = np.r_[-np.pi, (longitude[:-1]+longitude[1:])/2, np.pi]
        fig,axes = plt.subplots(2,2,figsize=(10.4,6.3),subplot_kw={'projection':'mollweide'},layout='constrained')
        for ax,values,title in zip(axes.flat,maps,labels):
            rolled = np.roll(values,nlon//2,axis=1)
            periodic = np.concatenate([rolled, rolled[:, :1]], axis=1)
            mesh = ax.pcolormesh(lon_edges,lat_edges,periodic,cmap='viridis',vmin=0,vmax=1,rasterized=True,shading='flat')
            ax.grid(alpha=.3)
            ax.set_title(title,pad=13,fontsize=11)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        fig.colorbar(mesh,ax=axes.ravel().tolist(),shrink=.75,label='Tracer amplitude (shared scale)')
        fig.savefig(FIGURES/f'tracer{suffix}.pdf')
        fig.savefig(FIGURES/f'tracer{suffix}.png')
        plt.close(fig)
    rows=[]
    fig,axes = plt.subplots(1,2,figsize=(10.4,3.7),layout='constrained')
    for path in sorted(RESULTS/f'tracer-{model}{suffix}.json'
                       for model in ('unet', 'unet-residual', 'paradis')
                       for suffix in ('', '-continuation', '-continuation-identity300')):
        result = json.loads(path.read_text())
        history=result['history']
        label=result['model'].replace('unet-residual','U-Net(R)').replace('unet','U-Net(P)').replace('paradis','PARADIS')
        label=label.replace('-continuation-identity300',' + warmup').replace('-continuation',' + ramp')
        axes[0].semilogy([r['step'] for r in history],[r['loss'] for r in history],label=label)
        axes[1].plot([r['step'] for r in history],[r['angle'] for r in history],label=label)
        rows.append(f"{label} & {result['parameters']:,} & {result['steps']} & {history[-1]['angle']:.1f} & {result['area_weighted_mse']:.3g} & {result['relative_mse']:.3f} & {result['elapsed_seconds']/60:.1f} \\\\")
    axes[0].set(xlabel='Optimizer update',ylabel='Area-weighted training MSE')
    axes[1].set(xlabel='Optimizer update',ylabel='Training displacement (degrees)')
    axes[0].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(FIGURES/'tracer-learning.pdf')
    plt.close(fig)
    (RESULTS/'tracer-table.tex').write_text('\n'.join(rows)+'\n')
    return baselines


def weather():
    rows=[]
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.7),layout='constrained')
    finals={}
    for name,color in [('conv2d','#176b9a'),('disco','#be5b23')]:
        source=ROOT/'logs/weather'/name
        for filename in ('metrics.jsonl','provenance.json'):
            shutil.copy2(source/filename, RESULTS/f'weather-{name}-{filename}')
        metrics=[json.loads(line) for line in (source/'metrics.jsonl').read_text().splitlines()]
        val=[r for r in metrics if 'val_loss' in r]
        for ax,x,label in zip(axes,('step','elapsed_seconds'),('Optimizer update','Elapsed training/validation time (minutes)')):
            xs=[r[x]/(60 if x=='elapsed_seconds' else 1) for r in val]
            ax.plot(xs,[r['val_loss'] for r in val],marker='o',label=name,color=color)
            ax.set(xlabel=label,ylabel='Validation aggregate loss')
            ax.grid(alpha=.2)
        progress=read(f'weather-{name}-progress.json')
        finals[name]=dict(final_loss=val[-1]['val_loss'],best_loss=progress['best_loss'],updates=progress['step'],fit_wall_seconds=progress['fit_wall_seconds'])
        rows.append(f"{name} & {progress['step']:,} & {val[-1]['val_loss']:.5f} & {progress['best_loss']:.5f} & {progress['fit_wall_seconds']/60:.1f} \\\\")
    axes[0].legend()
    fig.savefig(FIGURES/'weather-learning.pdf')
    plt.close(fig)
    ratio=finals['disco']['final_loss']/finals['conv2d']['final_loss']
    finals.update(final_loss_ratio=ratio,descriptive_limit=1.1,within_descriptive_limit=ratio<=1.1)
    (RESULTS/'weather-summary.json').write_text(json.dumps(finals,indent=2)+'\n')
    (RESULTS/'weather-table.tex').write_text('\n'.join(rows)+'\n')
    return finals


def benchmark():
    result=read('benchmark.json')
    rows=[]
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.4),layout='constrained')
    for ax,workload in zip(axes,('inference','training')):
        samples=result['samples_seconds'][workload]
        names=list(samples)
        ax.boxplot([samples[n] for n in names],tick_labels=names,showfliers=True)
        ax.set(title=workload.capitalize(),ylabel='Seconds per call')
        ax.grid(axis='y',alpha=.2)
        for name in names:
            upper=result['summary'][workload]['disco_over_upstream']['one_sided_95_upper'] if name=='disco' else None
            rows.append(f"{workload} & {name} & {np.median(samples[name]):.4f} & {f'{upper:.3f}' if upper else '---'} \\\\")
    fig.savefig(FIGURES/'latency.pdf')
    plt.close(fig)
    (RESULTS/'benchmark-table.tex').write_text('\n'.join(rows)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracer-dir', type=Path, default=Path('/home/yikwill/paradis-data/tracer'))
    args = parser.parse_args()
    baselines=tracer(args.tracer_dir)
    scores=weather()
    perf=benchmark()
    macros={
        'WeatherRatio': f"{scores['final_loss_ratio']:.3f}",
        'WeatherComparison': 'meets' if scores['within_descriptive_limit'] else 'does not meet',
        'InferenceBound': f"{perf['summary']['inference']['disco_over_upstream']['one_sided_95_upper']:.3f}",
        'TrainingBound': f"{perf['summary']['training']['disco_over_upstream']['one_sided_95_upper']:.3f}",
        'SpeedGate': 'passes' if all(perf['summary'][workload]['disco_over_upstream']['no_slowdown']
                                  for workload in ('inference', 'training')) else 'fails',
        'ConstantRatio': f"{baselines['constant_over_zero']:.5f}",
        'ConstantAmplitude': f"{baselines['optimal_constant_amplitude']:.5f}",
    }
    data=read('data-benchmark.json')
    macros.update(DataUpstream=f"{1000*data['upstream']['median_sample_seconds']:.1f}",
                  DataOptimized=f"{1000*data['optimized']['median_sample_seconds']:.1f}",
                  DataSpeedup=f"{data['upstream']['median_sample_seconds']/data['optimized']['median_sample_seconds']:.2f}")
    kernels=[]
    for row in read('grouped-cpu-final.json'):
        batch,channels,height,width=row['shape']
        old,new=row['old_and_folded_seconds']
        kernels.append(f"{height}$\\times${width}, C={channels} & {1000*np.median(old):.3f} & {1000*np.median(new):.3f} & {row['median_speedup']:.2f} \\\\")
    (RESULTS/'kernel-table.tex').write_text('\n'.join(kernels)+'\n')
    memory=[]
    for name in ('upstream','conv2d','disco'):
        row=read(f'memory-{name}.json')
        memory.append(f"{name} & {row['imported_peak_rss_kib']/1024**2:.2f} & {row['peak_rss_kib']/1024**2:.2f} \\\\")
    (RESULTS/'memory-table.tex').write_text('\n'.join(memory)+'\n')
    rmse=read('weather-physical-rmse.json')['scores']
    quality=[]
    for name,values in [('persistence',rmse['conv2d']['persistence_rmse']),
                        ('conv2d',rmse['conv2d']['rmse']),('disco',rmse['disco']['rmse'])]:
        quality.append(f"{name} & {values[0]:.2f} & {values[1]:.3f} \\\\")
    (RESULTS/'weather-physical-table.tex').write_text('\n'.join(quality)+'\n')
    (RESULTS/'numbers.tex').write_text(''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()))


if __name__=='__main__':
    main()
