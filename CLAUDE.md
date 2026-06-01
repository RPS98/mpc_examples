# CLAUDE.md

Guía para Claude sobre este repositorio. Úsala para orientarte rápido antes
de buscar en el árbol. Mantén este fichero breve: los detalles viven en el
código y en los README por carpeta.

## Propósito

Showcase unificado de **4 controladores outer-loop × 5 generadores de
referencia**, ejecutados sobre `mav_simulator` (quadrotor con INDI + modelo
de cuerpo rígido). Las combinaciones se lanzan desde **dos entry-points C++
(`position_examples`, `trajectory_examples`) y sus espejos Python
(`examples_py/runs/run_position_examples.py` y `run_trajectory_examples.py`)**
leyendo `sim_config.runs[]` de `configs/simulation/config_example.yaml`.
Cada caso emite por defecto un MCAP unificado (`output_format: mcap`)
servido por `mav_flight_review`, con topics para ground-truth, referencias,
comandos, IMU, motores y metadatos del run.

El objetivo es doble:
1. Comparar PID geométrico vs MPC (posición) vs SSA-MPC (posición con
   steady-state aumentado) vs MPC (trayectoria) sobre el mismo vehículo
   simulado y los mismos generadores.
2. Dar una interfaz base limpia para enchufar nuevos controladores o
   generadores sin tocar la infraestructura.

## Stack

- **C++17**, CMake ≥ 3.16, yaml-cpp, Eigen.
- **Python 3** (ROS 2 Humble target) con bindings pybind11 publicados bajo
  `build/python/`.
- **acados** para los tres MPCs. `position` y `trajectory` viven en
  `libs/acados_{position,trajectory}_mpc/`. `ssa_position` se consume
  in-place desde el repo externo `RPS98/ssa_position_mpc` (vendoreado en
  `../thirdparty_libs/ssa_position_mpc/acados_ssa_position_mpc/`) — su
  CMakeLists se monta con `add_subdirectory` out-of-tree desde
  `libs/CMakeLists.txt`.
- **mav_simulator** para la dinámica (500 Hz INDI + 1000 Hz modelo físico).
- **mav_flight_review** para logging MCAP/CSV + dashboard de métricas.

## Layout

```
examples_cpp/
├── include/
│   ├── framework/                  # IController, ITrajectoryGenerator,
│   │                               # WaypointsSimulator, WaypointScheduler,
│   │                               # DelayBuffer, factories, parallel_runner,
│   │                               # unified_mcap_logger.
│   ├── controllers/                # pid_position_geometric, pid_trajectory_geometric,
│   │                               # mpc_position, ssa_position_mpc, mpc_trajectory
│   ├── generators/                 # waypoint_reference, jerk_limited, gcopter,
│   │                               # dynamic_trajectory_generator, mav_traj_gen
│   └── utils/                      # example_config_utils, utils
├── src/                            # implementaciones + run_*_examples.cpp
└── tests/                          # gtest por adaptador + por framework module

examples_py/
├── examples_py/                    # mirror puro Python de examples_cpp
│   ├── framework/                  # IController, ITrajectoryGenerator, ...
│   ├── controllers/                # 4 controllers
│   ├── generators/                 # 5 generators
│   └── runs/                       # run_position_examples.py + run_trajectory_examples.py
└── tests/                          # pytest suites paralelas a examples_cpp/tests

configs/
├── controllers/                    # config_pid.yaml (position scope),
│                                   # config_pid_trajectory.yaml,
│                                   # config_mpc.yaml, config_mpc_trajectory.yaml,
│                                   # config_ssa_position_mpc.yaml
├── generators/                     # config_{waypoints,jerk_limited,gcopter,dynamic,mav_traj_gen}.yaml
└── simulation/                     # config_example.yaml + config_simulator.yaml

libs/                               # acados-generated solvers
                                    # acados_{position,trajectory}_mpc/ are in-tree;
                                    # acados_ssa_position_mpc is added via add_subdirectory
                                    # from ../thirdparty_libs/ssa_position_mpc/.
scripts/
├── run_all.sh                      # Lanza el unificado C++/Python + métricas + dashboard
├── single/                         # Un script por (controller × generator) habilitado
├── compute_metrics.sh              # Wrapper sobre mav_flight_review.compute_metrics
└── plot.sh                         # Wrapper sobre mav_flight_review.plotter

thirdparty/                         # 7 submódulos (ver lista abajo)
simulator_logs/                     # Salida de ejecuciones (gitignored), estructura:
                                    #   <run_id>/{cpp,py}/<ctrl>_<gen>.{mcap,csv}
                                    #   <run_id>/metrics/...
                                    #   <run_id>/plots/...
```

## Submódulos (`thirdparty/`)

Siete submódulos. Mantener la lista en sync con `.gitmodules`.

- `mav_simulator` — simulador (rigid-body + INDI), bindings `mavpy.*`.
- `mpc` — controllers MPC posición y trayectoria (acados).
- `ssa_position_mpc` — controller MPC posición con steady-state aumentado
  (acados, formulación "MPC for tracking"). No es submódulo de
  `mav_examples`; el repo `RPS98/ssa_position_mpc` se vendora en el
  workspace padre bajo `../thirdparty_libs/ssa_position_mpc/` y este
  proyecto lo consume in-place (módulo Python `ssa_position_mpc_acados`
  vía `PYTHONPATH` en `libs/generate_acados.sh`; lib C++
  `acados_ssa_position_mpc` vía `add_subdirectory` en `libs/CMakeLists.txt`).
- `dynamic_trajectory_generator` — polynomial dynamic trajectory (replan async).
- `gcopter_lib` — GCOPTER + safe-flight corridor optimiser.
- `trajectory_generator_jerk_limited` — S-curve jerk-limited generator.
- `mav_trajectory_generation_lib` — facade sin ROS sobre ETH-ASL
  `mav_trajectory_generation` (polinomial grado 10).
- `mav_flight_review` — logger MCAP/CSV + viewer/dashboard.

## Interfaces base

Documentadas en detalle en los headers. Resumen:

- **`IController`** ([examples_cpp/include/framework/controller_base.hpp](examples_cpp/include/framework/controller_base.hpp)):
  `compute(state, references, dt) -> ControlCommand`, `initialize()`,
  `requiredReferenceFields()`, `referenceHorizonSize()`, `name()`.
- **`ITrajectoryGenerator`** ([trajectory_generator_base.hpp](examples_cpp/include/framework/trajectory_generator_base.hpp), **API p2p**):
  `initialize(initial_state, example_cfg)`,
  `onWaypointChanged(next_waypoint, state, t_start)` — replanifica sólo al
  cambiar de waypoint; `update(t, state)`; `evaluate(t) -> ReferenceSample`;
  `providedReferenceFields()`.
- **`WaypointScheduler`** decide el cambio de waypoint **por tiempo**:
  `t_switch[i] = t_switch[i-1] + distance/(max_speed*scheduler_speed_factor) + settle_margin_s`.
  Garantiza que las combinaciones reciben el cambio en el mismo `t_sim`.
  El factor (≤ 1) modela que ningún generador real sostiene `max_speed`
  durante todo el segmento (gcopter es campana, jerk_limited tiene rampas,
  el carrot de waypoints satura el cascade ligeramente por debajo).
- **`DelayBuffer<T>`** modela latencia de cómputo (controller + generator).
  Modo `measured` (wall-clock real por iteración) o `fixed` (YAML-forzado).
- **`ReferenceField`** bitmask: `kPosition | kVelocity | kAcceleration`. La
  compatibilidad controller↔generator se chequea en `WaypointsSimulator`
  (warning si el generador no cubre todo lo que el controlador pide).
- **`ExampleConfig`** (parseada de `configs/simulation/config_example.yaml`):
  `sim_time`, `model_dt`, `controller_dt`, `mpc_dt`, `pid_dt`, `max_speed`,
  `hover_time`, `path_facing`, `settle_margin_s`, `scheduler_speed_factor`,
  `controller_delay_mode`,
  `controller_delay_fixed_s`, `generator_delay_mode`,
  `generator_delay_fixed_s`, `output_format`, `parallel`, `silent`,
  `benchmark`, `waypoints[]`, `runs[]`.
- **`ControllerKeys` / `GeneratorKeys`**:
  `pid | mpc_position | ssa_position_mpc | mpc_trajectory` × `waypoints |
  jerk_limited | gcopter | dynamic | mav_traj_gen`. La clave `pid` se
  despacha en la factory según el scope: `position_examples` →
  `PidPositionGeometricController` (cascada pos→vel→acc, sólo consume
  `kPosition`); `trajectory_examples` → `PidTrajectoryGeometricController`
  (paralelo pos+vel→acc con feedforward de aceleración via
  `pid_controllers::TrajectoryController`, consume `kPosition | kVelocity
  | kAcceleration`). Cada uno tiene su YAML (`config_pid.yaml` vs
  `config_pid_trajectory.yaml`) con sólo los parámetros que de verdad
  usa. `mpc_position` y `ssa_position_mpc` son ambos `kPosition`-only y
  comparten el mismo set de waypoints; coexisten en el mismo binario
  gracias a que el SSA solver vive en namespace `acados_ssa_mpc` (evita
  colisión ODR con `acados_mpc::MPC` del position MPC baseline).
- **Scope por binario**: `position_examples` recoge runs cuyo generador es
  `waypoints` (controllers `pid | mpc_position | ssa_position_mpc`);
  `trajectory_examples` recoge `gcopter | jerk_limited | dynamic |
  mav_traj_gen` (controllers `pid | mpc_trajectory`). Cualquier run
  habilitado fuera del scope se salta con un log-info.
- **`runs[]` con configs explícitos**: cada entrada declara los cuatro
  campos (`controller`, `generator`, `enabled`, `controller_config`,
  `generator_config`) en block style. Las factorías siguen teniendo
  defaults internos, pero el YAML los sobreescribe siempre — no hay
  defaults ocultos en el catálogo.

## Contratos importantes

- **`max_speed` es una sola fuente de verdad**: vive en
  `sim_config.max_speed` de `configs/simulation/config_example.yaml`. Los
  generadores NO deben declarar `max_speed` en su YAML — lo reciben vía
  `example_cfg` en `initialize()`. Gcopter fija `drone_limits.max_velocity`
  desde ahí también.
- **Frecuencias**: 100 Hz outer loop (controller+generator), 500 Hz INDI,
  1000 Hz dinámica. Configurables en `config_example.yaml` pero el código
  asume `model_dt ≤ controller_dt ≤ {mpc_dt, pid_dt}`.
- **Duración de la simulación**: el bucle corre hasta
  `min(mission_end + hover_time, sim_time)`, donde `mission_end` es el
  último `t_switch` del `WaypointScheduler`. `sim_time` es por tanto un
  tope duro (validado `> 0`) que puede truncar la fase de hover o incluso
  la propia misión si se fija demasiado pequeño.
- **Singles independientes de `enabled`**: cuando el binario recibe
  **ambos** `--only-controller=X` y `--only-generator=Y` (patrón de
  `scripts/single/*.sh`), el entry matching de `runs[]` se ejecuta
  aunque tenga `enabled: false`. Los `*_config` se siguen leyendo del
  catálogo. Si el combo (X, Y) no existe en `runs[]` se aborta con un
  mensaje claro pidiendo añadirlo. Con un solo `--only-*` (o ninguno)
  el flag `enabled` se respeta — caso de `run_all.sh`. Implementado
  en [run_position_examples.cpp](examples_cpp/src/run_position_examples.cpp),
  [run_trajectory_examples.cpp](examples_cpp/src/run_trajectory_examples.cpp)
  y [_runner.py](examples_py/examples_py/runs/_runner.py).
- **Telemetría**: MCAP ROS 2-compatible por defecto
  (`sim_config.output_format: mcap`); CSV si `output_format: csv`. Topics
  emitidos por `unified_mcap_logger` cubren ground-truth, references,
  comandos outer, IMU, motores y metadata; `mav_flight_review` los
  consume para métricas y dashboard.
- **Unidades**: SI. Posiciones en m (ENU), velocidades m/s, orientaciones en
  cuaterniones `[w, x, y, z]`, tiempos en segundos.
- **acados codegen**: `bash libs/generate_acados.sh` (lo invoca `build.sh`
  automáticamente cuando faltan los `.so`). Los directorios
  `libs/acados_{position,trajectory}_mpc/mpc_generated_code/` y
  `../thirdparty_libs/ssa_position_mpc/acados_ssa_position_mpc/mpc_generated_code/`
  son output, no editar a mano.

## Workflows habituales

```bash
# Build desde cero (tras clonar)
git submodule update --init --recursive
bash build.sh         # acados codegen (si hace falta) + cmake build → build/

# Sanity run (los runs habilitados en config_example.yaml → runs[])
./build/examples_cpp/position_examples \
  -c configs/simulation/config_example.yaml \
  -s configs/simulation/config_simulator.yaml
# Produce simulator_logs/<run_id>/cpp/{pid,mpc_position,ssa_position_mpc}_waypoints.mcap

# Comparativa completa (todos los runs habilitados, C++ + Python)
./scripts/run_all.sh --lang=both

# Caso individual (los más comunes)
./scripts/single/pid_waypoints_cpp.sh
./scripts/single/mpc_trajectory_mav_traj_gen_py.sh

# Tests
ctest --test-dir build --output-on-failure
ctest --test-dir build --output-on-failure -E pytest    # solo gtest
ctest --test-dir build --output-on-failure -L pytest    # solo pytest
```

## Decisiones activas / preferencias del usuario

- README principal en inglés; CLAUDE.md en español; comunicación con el
  usuario en español, código y comentarios en inglés.
- Cuando un fichero queda huérfano, mover a `old/` o `~/tmp_project/` en
  lugar de borrarlo; preferir `trash` antes que `rm -rf` para que sea
  recuperable.
- **No bumpear submódulos** a HEAD remoto salvo petición explícita; se
  mantiene el commit fijado.
- Commits sin trailer `Co-Authored-By`. Nunca commitear sin que el usuario
  lo pida.

## Gotchas

- `configs/generators/config_dynamic.yaml` es un **placeholder con sólo
  comentarios**. El adaptador `dynamic` sólo comprueba existencia del
  fichero; no lo parsea como mapping.
- **Dos binarios C++** (`position_examples`, `trajectory_examples`) y **dos
  scripts Python** (`run_position_examples.py`, `run_trajectory_examples.py`).
  Los binarios legacy `mpc_examples_run_<ctrl>_<gen>` y los subdirs
  `examples/<ctrl>_<gen>/` se han retirado.
- Los mirrors Python se publican bajo `build/python/` vía hard-links /
  symlinks creados por CMake. Editar la fuente y el build se actualiza
  automáticamente. `run_all.sh` y `scripts/single/*` añaden ese path a
  `PYTHONPATH`.
- `simulator_logs/` está en `.gitignore`. Las salidas se estructuran por
  `run_id` (`YYYYmmdd_HHMMSS`); el unificado imprime el `run_id` por stdout,
  y `--output-dir` lo puede sobreescribir.
- Los generadores `gcopter`, `dynamic` y `mav_traj_gen`, al operar p2p con
  sólo 2 waypoints (replan en cada cambio), pierden la optimización global
  multi-waypoint. Es el precio de la comparativa justa. Los tiempos de
  cómputo picando en los `t_switch` del scheduler son la evidencia visible
  del replan.
- **`gcopter` es min-jerk by design — NO produce plateau a `max_speed`**.
  El perfil de velocidad de
  [thirdparty/gcopter_lib](thirdparty/gcopter_lib/) es siempre una
  campana suave que toca `max_speed` en su cresta y baja: NUNCA un
  crucero plano sostenido. La cost function en
  [thirdparty/gcopter_lib/GCOPTER/gcopter/include/gcopter/gcopter.hpp:472-510](thirdparty/gcopter_lib/GCOPTER/gcopter/include/gcopter/gcopter.hpp#L472-L510)
  es
  `cost = MINCO_energy(jerk²) + Σ penalty(v, ω, θ, thrust, pos) + rho*Σtimes`,
  con `rho = optimization.time_weight`. Verificado empíricamente
  (`/tmp/gcopter_distance_sweep/` durante la investigación de
  2026-04-28):
    - **Aumentar la distancia entre waypoints no genera plateau**:
      sweep D ∈ {10, 20, 30} m con `max_speed=1` mantiene `t>0.95·v_max`
      en ~3 s/10 m, fracción `t95/hop_dur` plana en ~0.20-0.26.
      Además `gcopter` elige duraciones largas que para D≥20 ni
      siquiera alcanzan el waypoint dentro de `sim_time` (px_reached
      cae a 0.94→0.79→0.72).
    - **Subir `time_weight`** (sweep `tw ∈ {20, 100, 250, 500}` con
      D=10 y D=30) **tampoco**: cambios de <8% en `t>0.99`, forma
      idéntica a 3 decimales. Las penalizaciones de
      `body_rate_weight`, `tilt_weight` y `thrust_weight` (todas
      ≥1e+4) dominan sobre `rho` y limitan la pendiente de la rampa.
      Y existe una zona inestable (`tw=100` con D=10) donde el
      `PidTrajectoryGeometricController` diverge (rmse=NaN).
  Conclusión: si lo que se necesita es crucero plano a `v_max`, usar
  `jerk_limited` (S-curve con plateau explícito); `gcopter` se
  mantiene en sus valores actuales (`time_weight=20`,
  `velocity_weight=1e+4`) por ser el sweet-spot trackeable por ambos
  controladores. **No retunear esperando plateau**.
- **Delay `measured`** introduce no-determinismo en las métricas wall-clock
  entre ejecuciones. Para CI/tests, forzar `controller_delay_mode: fixed`
  con `controller_delay_fixed_s: 0.0` (y análogo para el generator).
  `loadTestSimConfig()` ya lo hace.
- **Modo paralelo (`sim_config.parallel: true` o flag `--parallel`)**
  lanza un worker por caso habilitado: `std::thread` en C++, subproceso
  (`ProcessPoolExecutor`) en Python. Ambos entry-points fuerzan
  internamente `silent=true` en el config que pasa a los workers para que
  las barras `\r` no se intercalen. Cuando `enabled_in_scope > núcleos`
  los hilos compiten y los `*_compute_time_us` se inflan; combinar paralelo
  con `*_delay_mode: measured` también contamina la métrica (warning
  emitido al arrancar). Para timing reproducible en paralelo, usar
  `*_delay_mode: fixed`.
- `thirdparty/mpc` es un único submódulo que provee position MPC y
  trajectory MPC (no son dos submódulos separados, a pesar de que el
  `.gitmodules` histórico lo sugiriera).
- `mpcc` se eliminó de `.gitmodules`. El contenido del último commit
  está en `~/tmp_project/mpcc/` por si se necesita reactivar.
- Algunos pytest internos de submódulos pueden fallar (e.g.
  `pytest_mav_flight_review` por imports de nombres antiguos,
  `pytest_mav_trajectory_generation_lib` por SEGFAULT en
  `test_spline_throws_when_invalid`). Son issues upstream — no de los
  adaptadores ni del showcase.
- **Tests del framework C++ que enlazan factories_trajectory**
  (`test_factories_trajectory` y `test_waypoints_simulator`) usan un
  `main()` propio con `std::_Exit(rc)` para evitar un crash benigno en los
  destructores globales (acados + dynamic-trajectory worker thread + sim).
  Toda la lógica de gtest pasa antes del `_Exit`; CTest reporta los tests
  individuales como verdes.
- **`trajectory_examples` (binario)** termina con `std::_Exit(0)` por la
  misma razón: los destructores globales (acados, dynamic-trajectory
  worker, mav_simulator) se pisan al salir y producen un `double free or
  corruption` benigno. Los datos del run y los loggers MCAP/CSV ya están
  flusheados antes de ese punto (el destructor de cada `WaypointsSimulator`
  cierra su logger dentro de `runCase`). El binario `position_examples` no
  necesita el workaround porque no carga el adaptador dinámico.
- **`dynamic` adapter (C++ y Python)**: el adaptador recrea la instancia
  subyacente de `dynamic_traj_generator::DynamicTrajectory` en cada
  `onWaypointChanged`/`on_waypoint_changed` y evalúa en tiempo relativo
  al inicio del segmento (`t - t_segment_start_`). Es el único uso
  defendible de la librería en modo p2p: si se reutiliza la instancia,
  el origen de tiempo interno (`global_time_last_trajectory_generated`)
  arrastra el `last_global_time_evaluated` del segmento previo y la
  librería stitch-ea sobre la trayectoria anterior, dejando la
  referencia clampada al final del segmento previo (v=0) en cuanto
  llega el segundo waypoint.
- **Singles**: un script por cada (controller × generator) válido en
  C++ y Python (incluye los del scope `position` con `mpc_position` y
  `ssa_position_mpc`, los del scope `trajectory` con `mpc_trajectory` y
  `pid`, y las variantes `*_moving_path_*` del experimento continuo).
  Cada single ejecuta el binario, llama a `compute_metrics`, imprime el
  `print_summary` por terminal y lanza el plotter de `mav_flight_review`
  (todo desde `scripts/_lib/env.sh :: post_run_review`).
- **Python `run_with_filter`** devuelve siempre `0` (igual que el binario
  C++). Los fallos por caso se reflejan en la tabla de resumen pero no
  abortan al wrapper bajo `set -euo pipefail`.
- **Colisión de símbolos `mav_trajectory_generation` (Linux)**: el
  facade [thirdparty/mav_trajectory_generation_lib/](thirdparty/mav_trajectory_generation_lib/)
  enlaza estáticamente la upstream ETH-ASL
  `mav_trajectory_generation_core`, y el subpaquete vendoreado por
  `dynamic_trajectory_generator` también expone esos mismos símbolos
  desde su propia `libmav_trajectory_generation.so` con un
  `estimateSegmentTimes` parcheado (`d/v + v/a` en vez de la heurística
  `magic_fabian_constant` upstream). Sin precaución el loader colapsa
  ambas copias en una sola y el facade acaba llamando al
  `estimateSegmentTimes` de dyngen, generando segmentos ~½ de duración
  y picos de `v` >> `max_speed` — síntoma observado en C++ pero no en
  Python (los bindings cargan el facade aislado). El
  [CMakeLists.txt](thirdparty/mav_trajectory_generation_lib/CMakeLists.txt)
  del facade aplica `LINKER:--exclude-libs,ALL` (sólo en Linux) para
  ocultar los símbolos heredados de la archive estática y romper la
  colisión. El mismo parche **no** se aplica a
  `dynamic_trajectory_generator` porque su API pública filtra cabeceras
  del upstream (no es Pimpl) y ocultar los símbolos rompería el link de
  sus adaptadores.
- **`mav_traj_gen` alineado con `dynamic`**:
  [configs/generators/config_mav_traj_gen.yaml](configs/generators/config_mav_traj_gen.yaml)
  fija `derivative_to_optimize: 2` (ACCELERATION) y `a_max: 9.81` para
  coincidir con los valores hard-codeados de
  `dynamic_trajectory_generator` (`derivative_to_optimize_` =
  `ACCELERATION` y `MAV_MAX_ACCEL = 1.0 * 9.81` en
  `dynamic_trajectory.hpp`). De este modo ambos generadores polinómicos
  comparten objetivo de optimización y límite de aceleración, y son
  comparables sin sesgo de tuning.
