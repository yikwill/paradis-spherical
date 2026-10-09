#!/usr/bin/env bash
set -euo pipefail
paradis_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
paradis_root=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
cd -- "$paradis_repo"
micromamba create -y -r "$paradis_root" -n paradis-spherical -f "$paradis_repo/environment/conda-linux-64.lock"
paradis_prefix="$paradis_root/envs/paradis-spherical"
mkdir -p "$paradis_prefix/etc/conda/activate.d" "$paradis_prefix/etc/conda/deactivate.d"
cp "$paradis_repo/environment/activate.sh" "$paradis_prefix/etc/conda/activate.d/paradis.sh"
cp "$paradis_repo/environment/deactivate.sh" "$paradis_prefix/etc/conda/deactivate.d/paradis.sh"
micromamba run -r "$paradis_root" -n paradis-spherical python -m pip install -r "$paradis_repo/environment/pip-cpu.lock" --index-url https://download.pytorch.org/whl/cpu
paradis_harmonics="$paradis_root/src/torch-harmonics"
if [[ ! -d "$paradis_harmonics/.git" ]]; then
    git clone https://github.com/NVIDIA/torch-harmonics.git "$paradis_harmonics"
fi
git -C "$paradis_harmonics" checkout --detach 6e7648d4435109c8a4ae747a9cd279b6ae3a30ba
micromamba run -r "$paradis_root" -n paradis-spherical env MAX_JOBS=8 TORCH_HARMONICS_ENABLE_OPENMP=1 TORCH_HARMONICS_NATIVE_CPU_ARCH=1 python -m pip install --no-deps --no-build-isolation "$paradis_harmonics"
micromamba run -r "$paradis_root" -n paradis-spherical python -m tests.check_model
