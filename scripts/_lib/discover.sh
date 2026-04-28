# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# Shared helpers for the wrapper scripts under scripts/. Centralises the
# discovery of the newest populated run_dir under simulator_logs/ so each
# wrapper reduces to: source this file + one function call + exec.
#
# Exposed functions:
#
#   resolve_run_dir "$@"
#     Inspects $1 and sets RUN_DIR + CONSUMED_ARGS.
#       - If $1 is an existing directory and not a flag, RUN_DIR=$1 and
#         CONSUMED_ARGS=1 (the caller must `shift` it off $@).
#       - Otherwise falls back to find_newest_run; RUN_DIR=<newest>,
#         CONSUMED_ARGS=0 and prints "[info] using newest populated run: …"
#         to stderr.
#     Returns 0 on success, 1 with "[err] …" on stderr when no run exists.
#
#   resolve_log_files "$@"
#     Splits positionals into LOG_FILES (existing files) and EXTRA_ARGS
#     (flags or the remainder once a flag is seen). If LOG_FILES is empty
#     after parsing, falls back to every *.csv / *.mcap under the newest
#     run's cpp/ and py/ (excluding *_metrics.csv and *_segments.csv).
#     Returns 0 on success, 1 when nothing plottable is resolvable.
#
#   resolve_plot_target "$@"
#     Dispatches between run-dir mode and file-list mode:
#       - If $1 is an existing file: MODE=files; runs resolve_log_files.
#       - If $1 is an existing directory: MODE=run; RUN_DIR=$1; the rest
#         goes into EXTRA_ARGS.
#       - Otherwise (flag or empty $1): MODE=run; RUN_DIR=find_newest_run;
#         the untouched args go into EXTRA_ARGS.
#     Returns 0 on success, 1 with "[err] …" on stderr when nothing
#     resolvable is found.

# Load find_newest_run when not already available.
if ! declare -F find_newest_run > /dev/null; then
  _DISCOVER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  # shellcheck source=./newest_run.sh
  source "${_DISCOVER_DIR}/newest_run.sh"
  unset _DISCOVER_DIR
fi

resolve_run_dir() {
  RUN_DIR=""
  CONSUMED_ARGS=0
  local first="${1:-}"
  if [[ -n "${first}" && "${first}" != -* && -d "${first}" ]]; then
    RUN_DIR="${first}"
    CONSUMED_ARGS=1
    return 0
  fi
  RUN_DIR="$(find_newest_run || true)"
  if [[ -z "${RUN_DIR}" ]]; then
    echo "[err] no run_dir given and no populated directory found under simulator_logs/." >&2
    return 1
  fi
  echo "[info] using newest populated run: ${RUN_DIR}" >&2
  return 0
}

resolve_log_files() {
  LOG_FILES=()
  EXTRA_ARGS=()
  local seen_flag=0
  while (($#)); do
    if [[ "${seen_flag}" -eq 0 && "$1" != -* && -f "$1" ]]; then
      LOG_FILES+=("$1")
    else
      [[ "$1" == -* ]] && seen_flag=1
      EXTRA_ARGS+=("$1")
    fi
    shift
  done

  if [[ "${#LOG_FILES[@]}" -eq 0 ]]; then
    local run_dir
    run_dir="$(find_newest_run || true)"
    if [[ -z "${run_dir}" ]]; then
      echo "[err] no files given and no populated run under simulator_logs/." >&2
      return 1
    fi
    echo "[info] using newest populated run: ${run_dir}" >&2
    local d
    for d in "${run_dir}/cpp" "${run_dir}/py"; do
      [[ -d "${d}" ]] || continue
      while IFS= read -r -d '' f; do
        case "${f}" in
          *_metrics.csv|*_segments.csv) continue ;;
        esac
        LOG_FILES+=("${f}")
      done < <(find "${d}" -maxdepth 1 -type f \( -name '*.csv' -o -name '*.mcap' \) -print0 2>/dev/null | sort -z)
    done
  fi

  if [[ "${#LOG_FILES[@]}" -eq 0 ]]; then
    echo "[err] no plottable files found." >&2
    return 1
  fi
  return 0
}

resolve_plot_target() {
  MODE=""
  RUN_DIR=""
  LOG_FILES=()
  EXTRA_ARGS=()
  local first="${1:-}"

  if [[ -n "${first}" && "${first}" != -* && -f "${first}" ]]; then
    MODE="files"
    resolve_log_files "$@"
    return $?
  fi

  MODE="run"
  if [[ -n "${first}" && "${first}" != -* && -d "${first}" ]]; then
    RUN_DIR="${first}"
    shift
  else
    RUN_DIR="$(find_newest_run || true)"
    if [[ -z "${RUN_DIR}" ]]; then
      echo "[err] no run_dir given and no populated directory found under simulator_logs/." >&2
      return 1
    fi
    echo "[info] using newest populated run: ${RUN_DIR}" >&2
  fi
  EXTRA_ARGS=("$@")
  return 0
}
