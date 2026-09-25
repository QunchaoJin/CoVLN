#!/bin/bash
# Co-VLN (MapGPT backbone) on R2R val unseen with prior-based episode pairing.
# Set OPENAI_API_KEY (and OPENAI_BASE_URL for an OpenAI-compatible endpoint) before running,
# e.g. `set -a; source .env; set +a`.

DATA_ROOT=./datasets
IMG_ROOT=/path/to/RGB_Observations   # pre-rendered observation images, see README
outdir=${DATA_ROOT}/exprs_map/qwen3vl_prior

flag="--root_dir ${DATA_ROOT}
      --img_root ${IMG_ROOT}
      --split val_unseen
      --pairing prior
      --output_dir ${outdir}
      --max_action_len 15
      --save_pred
      --stop_after 3
      --llm qwen3-vl-32b-instruct
      --response_format json
      --max_tokens 3000
      "

# Add e.g. `--end 5` to run only the first 5 pairs for a quick test.
python vln/main_gpt.py $flag
