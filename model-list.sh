#!/bin/sh

set -eu

if [ -n "${HF_HUB_CACHE:-}" ]; then
    cache_dir=$HF_HUB_CACHE
elif [ -n "${HF_HOME:-}" ]; then
    cache_dir=$HF_HOME/hub
elif [ -n "${XDG_CACHE_HOME:-}" ]; then
    cache_dir=$XDG_CACHE_HOME/huggingface/hub
else
    cache_dir=$HOME/.cache/huggingface/hub
fi

[ -d "$cache_dir" ] || exit 0

for model_dir in "$cache_dir"/models--mlx-community--*; do
    [ -d "$model_dir/snapshots" ] || continue
    model_name=${model_dir##*/models--mlx-community--}
    printf 'mlx-community/%s\n' "$model_name"
done | LC_ALL=C sort
