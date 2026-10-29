# Sourced by run_gpu.sh / run_cpu.sh: setup_env.sh <gpu | cpu>
# SHOWERLM_ENV (the env= submit variable): an activate script, a venv folder or a conda env folder.
# Empty: the job keeps the environment of the shell it was submitted from (getenv).
need=$1
env_path=${SHOWERLM_ENV:-}
if [[ -n "$env_path" ]]; then
    set +u                                          # activate scripts may read unset variables
    if [[ -f "$env_path" ]]; then
        source "$env_path"
    elif [[ -f "$env_path/bin/activate" ]]; then
        source "$env_path/bin/activate"
    elif [[ -d "$env_path/conda-meta" || -x "$env_path/bin/python" ]]; then
        export PATH="$env_path/bin:$PATH"
    else
        echo "ERROR: env=$env_path is not an activate script, a venv or a conda env" >&2
        exit 3
    fi
    set -u
    echo "environment: $env_path"
else
    echo "environment: inherited from the submitting shell (no env= given)"
fi
if ! command -v python > /dev/null; then
    echo "ERROR: no python on PATH" >&2
    exit 3
fi
echo "python: $(command -v python)"
python - "$need" <<'PY' || exit 3
import sys
print("python", sys.version.split()[0])
try:
    import torch, numpy, yaml  # noqa: F401
except ImportError as e:
    sys.exit(f"ERROR: {e}; install the packages of requirements.txt in this environment")
print("torch", torch.__version__)
if sys.argv[1] == "gpu":
    if not torch.cuda.is_available():
        sys.exit("ERROR: GPU job but torch.cuda.is_available() is False (torch without CUDA, or no GPU)")
    print("cuda device:", torch.cuda.get_device_name(0))
PY
