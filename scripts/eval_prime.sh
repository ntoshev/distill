#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/eval_prime.sh --environment ENV --num-examples N --adapter PATH [options]

Runs the same Prime/verifiers evaluation twice: once with the base model and
once with PATH loaded as an mlx-lm adapter. The two result sets are compared
by example ID. Without --output-dir, intermediate results are temporary and
only the pretty summaries and diff remain on the terminal.

Required:
  --environment, -e ENV       Verifiers/Prime environment, e.g. gsm8k
  --num-examples, -n N        Number of evaluation examples
  --adapter PATH               mlx-lm adapter directory to compare

Options:
  --output-dir, -o PATH        Save results under PATH/{base,with-adapter}
  --env-dir-path PATH          Directory containing local environments
  --model MODEL                MLX model (default: mlx-community/Qwen3.5-4B-6bit)
  --port PORT                  Local MLX server port (default: 18080)
  --max-concurrent N           Maximum concurrent requests (default: 32)
  --max-tokens N               Completion token limit (default: 1024)
  --temperature T              Sampling temperature (default: 0)
  --rollouts-per-example N     Rollouts per example (default: 1)
  --disable-env-server         Do not start a Prime environment server
  --help, -h                   Show this help

Examples:
  scripts/eval_prime.sh \
    --environment gsm8k \
    --num-examples 16 \
    --adapter outputs/fifa-first-5

  scripts/eval_prime.sh \
    --environment gsm8k \
    --num-examples 16 \
    --adapter outputs/fifa-first-5 \
    --output-dir outputs/prime-gsm8k/fifa-vs-base-16

  scripts/eval_prime.sh \
    --environment gsm8k-slice \
    --env-dir-path environments \
    --num-examples 16 \
    --adapter outputs/fifa-first-5
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

require_value() {
    if [[ $# -lt 2 || -z ${2:-} ]]; then
        die "${1} requires a value"
    fi
}

is_positive_integer() {
    [[ $1 =~ ^[1-9][0-9]*$ ]]
}

environment_id=''
num_examples=''
adapter_path=''
output_dir=''
env_dir_path=''
model='mlx-community/Qwen3.5-4B-6bit'
port=18080
max_concurrent=32
max_tokens=1024
temperature=0
rollouts_per_example=1
disable_env_server=0

while (($# > 0)); do
    case $1 in
        --environment|-e)
            require_value "$@"
            environment_id=$2
            shift 2
            ;;
        --num-examples|-n)
            require_value "$@"
            num_examples=$2
            shift 2
            ;;
        --adapter)
            require_value "$@"
            adapter_path=$2
            shift 2
            ;;
        --output-dir|-o)
            require_value "$@"
            output_dir=$2
            shift 2
            ;;
        --env-dir-path)
            require_value "$@"
            env_dir_path=$2
            shift 2
            ;;
        --model)
            require_value "$@"
            model=$2
            shift 2
            ;;
        --port)
            require_value "$@"
            port=$2
            shift 2
            ;;
        --max-concurrent)
            require_value "$@"
            max_concurrent=$2
            shift 2
            ;;
        --max-tokens)
            require_value "$@"
            max_tokens=$2
            shift 2
            ;;
        --temperature)
            require_value "$@"
            temperature=$2
            shift 2
            ;;
        --rollouts-per-example)
            require_value "$@"
            rollouts_per_example=$2
            shift 2
            ;;
        --disable-env-server)
            disable_env_server=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1 (use --help)"
            ;;
    esac
done

[[ -n $environment_id ]] || die "--environment is required"
[[ -n $num_examples ]] || die "--num-examples is required"
[[ -n $adapter_path ]] || die "--adapter is required"
is_positive_integer "$num_examples" || die "--num-examples must be a positive integer"
is_positive_integer "$max_concurrent" || die "--max-concurrent must be a positive integer"
is_positive_integer "$max_tokens" || die "--max-tokens must be a positive integer"
is_positive_integer "$rollouts_per_example" || die "--rollouts-per-example must be a positive integer"
[[ $port =~ ^[0-9]+$ ]] && ((port >= 1 && port <= 65535)) || die "--port must be between 1 and 65535"

command -v python >/dev/null 2>&1 || die "python is required"
command -v prime >/dev/null 2>&1 || die "prime is required"

[[ -d $adapter_path ]] || die "adapter directory not found: $adapter_path"
[[ -f $adapter_path/adapters.safetensors ]] || die "adapter weights not found: $adapter_path/adapters.safetensors"
[[ -f $adapter_path/adapter_config.json ]] || die "adapter config not found: $adapter_path/adapter_config.json"

if [[ -n $env_dir_path ]]; then
    [[ -d $env_dir_path ]] || die "environment directory not found: $env_dir_path"
fi

temporary_output=0
if [[ -n $output_dir ]]; then
    results_root=$output_dir
    mkdir -p "$results_root"
else
    results_root=$(mktemp -d "${TMPDIR:-/tmp}/distil-eval-prime-results.XXXXXX")
    temporary_output=1
fi
base_output=$results_root/base
adapter_output=$results_root/with-adapter
mkdir -p "$base_output" "$adapter_output"

server_log_dir=$(mktemp -d "${TMPDIR:-/tmp}/distil-eval-prime-server.XXXXXX")
server_pid=''
server_log=''
evaluation_results=''

cleanup() {
    local exit_status=$?

    if [[ -n $server_pid ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi

    if ((exit_status != 0)) && [[ -n $server_log ]] && [[ -s $server_log ]]; then
        printf '\nMLX server log:\n' >&2
        tail -n 80 "$server_log" >&2 || true
    fi

    if ((temporary_output == 1)); then
        rm -rf "$results_root"
    fi
    rm -rf "$server_log_dir"
    exit "$exit_status"
}
trap cleanup EXIT

server_args=(
    -m mlx_lm server
    --model "$model"
    --host 127.0.0.1
    --port "$port"
    --max-tokens "$max_tokens"
    --chat-template-args '{"enable_thinking":false}'
    --log-level WARNING
)

export MLX_LM_API_KEY=${MLX_LM_API_KEY:-local}
export PRIME_DISABLE_VERSION_CHECK=${PRIME_DISABLE_VERSION_CHECK:-1}

stop_server() {
    if [[ -n $server_pid ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=''
    server_log=''
}

start_server() {
    local label=$1
    server_log=$server_log_dir/${label}.log
    local -a command_args=("${server_args[@]}")
    if [[ $label == with-adapter ]]; then
        command_args+=(--adapter-path "$adapter_path")
    fi

    python "${command_args[@]}" >"$server_log" 2>&1 &
    server_pid=$!

    local server_ready=0
    for _ in {1..240}; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            die "MLX $label server exited before opening port $port"
        fi
        if python - "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=0.5):
    pass
PY
        then
            server_ready=1
            break
        fi
        sleep 0.25
    done
    ((server_ready == 1)) || die "MLX $label server did not open port $port within 60 seconds"
}

latest_results_file() {
    python - "$1" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
files = list(root.rglob("results.jsonl"))
if not files:
    raise SystemExit(f"no results.jsonl found under {root}")
print(max(files, key=lambda path: path.stat().st_mtime))
PY
}

run_evaluation() {
    local label=$1
    local run_output=$2
    local -a prime_args=(
        eval run "$environment_id"
        --model "$model"
        --api-base-url "http://127.0.0.1:$port/v1"
        --api-key-var MLX_LM_API_KEY
        --api-client-type openai_chat_completions
        --num-examples "$num_examples"
        --rollouts-per-example "$rollouts_per_example"
        --max-concurrent "$max_concurrent"
        --max-tokens "$max_tokens"
        --temperature "$temperature"
        --output-dir "$run_output"
        --save-results
        --abbreviated-summary
        --skip-upload
    )
    if [[ -n $env_dir_path ]]; then
        prime_args+=(--env-dir-path "$env_dir_path")
    fi
    if ((disable_env_server == 1)); then
        prime_args+=(--disable-env-server)
    fi
    if [[ ! -t 1 ]]; then
        prime_args+=(--disable-tui)
    fi

    printf '\n\033[1;35m%s\033[0m\n' "$label"
    prime "${prime_args[@]}"
    evaluation_results=$(latest_results_file "$run_output")
}

printf '\n\033[1;36mPrime / verifiers paired evaluation\033[0m\n'
printf '  environment: %s\n' "$environment_id"
printf '  model:       %s\n' "$model"
printf '  adapter:     %s\n' "$adapter_path"
printf '  examples:    %s\n' "$num_examples"
printf '  concurrency:  %s\n\n' "$max_concurrent"

start_server base
run_evaluation 'Base model (without adapter)' "$base_output"
base_results=$evaluation_results
stop_server

start_server with-adapter
run_evaluation 'Adapter model' "$adapter_output"
adapter_results=$evaluation_results
stop_server

python - "$base_results" "$adapter_results" "$output_dir" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

base_path = Path(sys.argv[1])
adapter_path = Path(sys.argv[2])
saved_root = sys.argv[3]


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def completion_text(row):
    completion = row.get("completion", "")
    if isinstance(completion, list):
        return "".join(
            message.get("content", "")
            for message in completion
            if message.get("role") == "assistant"
        )
    return str(completion)


def keyed(rows):
    counts = Counter()
    result = {}
    for index, row in enumerate(rows):
        example_id = row.get("example_id", f"row-{index}")
        occurrence = counts[example_id]
        counts[example_id] += 1
        result[(str(example_id), occurrence)] = row
    return result


def reward(row):
    value = row.get("reward")
    return float(value) if value is not None else None


def summary(rows):
    rewards = [reward(row) for row in rows]
    return {
        "examples": len(rows),
        "correct": sum(value == 1.0 for value in rewards),
        "reward": sum(value or 0.0 for value in rewards) / len(rows) if rows else 0.0,
        "truncated": sum(bool(row.get("is_truncated")) for row in rows),
        "errors": sum(bool(row.get("error")) for row in rows),
    }


base_rows = load(base_path)
adapter_rows = load(adapter_path)
base = keyed(base_rows)
adapter = keyed(adapter_rows)
common = sorted(set(base) & set(adapter))

base_summary = summary(base_rows)
adapter_summary = summary(adapter_rows)
print("\n\033[1;36mComparison\033[0m")
print("┌──────────────────┬──────────┬──────────┬───────────┬───────────┐")
print("│ Variant          │ Correct  │ Accuracy │ Truncated │ Errors    │")
print("├──────────────────┼──────────┼──────────┼───────────┼───────────┤")
for label, values in (("Base", base_summary), ("With adapter", adapter_summary)):
    accuracy = values["correct"] / values["examples"] if values["examples"] else 0.0
    print(
        f"│ {label:<16} │ {values['correct']:>3}/{values['examples']:<4} │ "
        f"{accuracy:>7.2%} │ {values['truncated']:>9} │ {values['errors']:>9} │"
    )
print("└──────────────────┴──────────┴──────────┴───────────┴───────────┘")

reward_changes = []
completion_changes = []
truncation_changes = []
for key in common:
    base_row = base[key]
    adapter_row = adapter[key]
    base_reward = reward(base_row)
    adapter_reward = reward(adapter_row)
    if base_reward != adapter_reward:
        reward_changes.append((key[0], base_reward, adapter_reward))
    if completion_text(base_row) != completion_text(adapter_row):
        completion_changes.append(key[0])
    if bool(base_row.get("is_truncated")) != bool(adapter_row.get("is_truncated")):
        truncation_changes.append(key[0])

print("\n\033[1;36mDiff\033[0m")
print(f"  paired examples:       {len(common)}")
print(f"  completion changes:    {len(completion_changes)}")
print(f"  reward changes:        {len(reward_changes)}")
print(f"  truncation changes:    {len(truncation_changes)}")
if reward_changes:
    print("\n  Reward changes by example:")
    for example_id, base_reward, adapter_reward in reward_changes:
        print(f"    {example_id}: base={base_reward:g} adapter={adapter_reward:g}")
else:
    print("  No reward changes; both variants have the same correctness results.")
if completion_changes:
    preview = ", ".join(completion_changes[:20])
    suffix = " ..." if len(completion_changes) > 20 else ""
    print(f"  Changed completion IDs: {preview}{suffix}")
if saved_root:
    print(f"\n\033[1;32mResults saved under {saved_root}\033[0m")
PY
