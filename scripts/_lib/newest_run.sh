# Copyright 2025 Universidad Politécnica de Madrid
# SPDX-License-Identifier: BSD-3-Clause
#
# find_newest_run: print the newest directory under simulator_logs/ that
# contains at least one CSV under cpp/ or py/. Empty / stale run_dirs are
# skipped so post-processing wrappers do not pick them up accidentally.

find_newest_run() {
  local root="simulator_logs"
  if [[ ! -d "${root}" ]]; then
    return 1
  fi
  local dir
  while IFS= read -r dir; do
    if compgen -G "${dir}/cpp/*.csv" > /dev/null \
       || compgen -G "${dir}/py/*.csv"  > /dev/null; then
      printf '%s' "${dir}"
      return 0
    fi
  done < <(ls -1dt "${root}"/*/ 2>/dev/null | sed 's:/$::')
  return 1
}
