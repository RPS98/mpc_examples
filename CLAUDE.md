# CLAUDE.md

Guía para Claude sobre este repositorio. Úsala para orientarte rápido antes
de buscar en el árbol. Mantén este fichero breve: los detalles viven en el
código y en los README por carpeta.

## Propósito

Showcase unificado de **3 controladores outer-loop × 4 generadores de
referencia = 12 combinaciones**, ejecutadas sobre `mav_simulator` (quadrotor
con INDI + modelo de cuerpo rígido). Todas las combinaciones se lanzan desde
**un único entry-point C++ (`mpc_examples_run`) y su espejo Python
(`examples/unified/run_example.py`)** leyendo `sim_config.runs[]` de
`configs/simulation/config_example.yaml`. Cada caso produce un CSV unificado
de 45 columnas (con tiempos de cómputo y nombres del caso) y alimenta los
scripts de métricas y el dashboard comparativo.

El objetivo es doble:
1. Comparar PID geométrico vs MPC (posición) vs MPC (trayectoria) sobre el
   mismo vehículo simulado.
2. Dar una interfaz base limpia para que se puedan enchufar controladores o
   generadores nuevos sin tocar la infraestructura.

## Stack

- **C++17**, CMake, yaml-cpp, Eigen.
- **Python 3** (ROS 2 Humble target) con bindings pybind11 publicados bajo
  `build/python/`.
- **acados** para los dos MPCs (código autogenerado en
  `examples/acados_{position,trajectory}_mpc/`).
- **mav_simulator** para la dinámica (500 Hz INDI + 1000 Hz modelo físico).

## Layout

```
examples/
├── framework/                  # IController, ITrajectoryGenerator, ExampleConfig,
│   │                           # WaypointScheduler, DelayBuffer, UnifiedCsvLogger,
│   │                           # factories, WaypointsSimulator. C++ headers + mirror Python.
│   └── python/                 # Mirror puro Python del framework
├── adapters/
│   ├── controllers/            # pid_geometric, mpc_position, mpc_trajectory
│   ├── trajectory_generators/  # waypoint_reference, jerk_limited, gcopter, dynamic
│   └── python/                 # Mirrors puros en Python de todo lo anterior
├── unified/                    # run_example.cpp + run_example.py (único entry-point)
├── acados_{position,trajectory}_mpc/  # Código acados generado (NO editar a mano)
└── utils/                      # compute_metrics.py, plot_results.py, *.hpp helpers

configs/
├── controllers/                # config_pid.yaml, config_mpc.yaml, config_mpc_trajectory.yaml
├── generators/                 # config_{waypoints,jerk_limited,gcopter,dynamic}.yaml
├── simulation/                 # config_example.yaml (scenario+runs[]+delays+max_speed),
│                               # config_simulator.yaml (drone params)
└── solver_definitions/         # solver_definition_mpc_{position,trajectory}.yaml (acados codegen)

scripts/
├── run_all.sh                  # Lanza el unificado C++/Python + compute_metrics + dashboard
└── dashboard.py                # Figura comparativa multi-panel

thirdparty/                     # 5 submódulos: mav_simulator, mpc, gcopter_lib,
                                # dynamic_trajectory_generator, trajectory_generator_jerk_limited
old/                            # Ficheros huérfanos pendientes de revisión del usuario
                                # (incluye los 12 subdirs legacy examples/<ctrl>_<gen>/)
simulator_logs/                 # Salida de ejecuciones (gitignored), estructura:
                                #   <run_id>/{cpp,py}/<ctrl>_<gen>.csv
                                #   <run_id>/metrics/summary.csv
                                #   <run_id>/plots/dashboard.png
```

## Interfaces base

Documentadas en detalle en `examples/README.md`. Resumen:

- **`IController`** (`examples/framework/include/framework/controller_base.hpp`):
  `compute(state, references, dt) -> ControlCommand`, `initialize()`,
  `requiredReferenceFields()`, `referenceHorizonSize()`, `name()`.
- **`ITrajectoryGenerator`** (`trajectory_generator_base.hpp`, **API p2p**):
  `initialize(initial_state, example_cfg)`,
  `onWaypointChanged(next_waypoint, state, t_start)` — replanifica sólo al
  cambiar de waypoint; `update(t, state)`; `evaluate(t) -> ReferenceSample`;
  `providedReferenceFields()`.
- **`WaypointScheduler`** decide el cambio de waypoint **por tiempo**:
  `t_switch[i] = t_switch[i-1] + distance/max_speed + settle_margin_s`.
  Garantiza que los 12 casos reciben el cambio en el mismo `t_sim`.
- **`DelayBuffer<T>`** modela latencia de cómputo (controller + generator).
  Modo `measured` (wall-clock real por iteración) o `fixed` (YAML-forzado).
- **`ReferenceField`** bitmask: `kPosition | kVelocity | kAcceleration`. La
  compatibilidad controller↔generator se chequea en `WaypointsSimulator`
  (warning si el generador no cubre todo lo que el controlador pide).
- **`ExampleConfig`** (parseada de `configs/simulation/config_example.yaml`):
  `sim_time`, `model_dt`, `controller_dt`, `mpc_dt`, `pid_dt`, `max_speed`,
  `hover_time`, `path_facing`, `settle_margin_s`, `controller_delay_mode`,
  `controller_delay_fixed_s`, `generator_delay_mode`,
  `generator_delay_fixed_s`, `waypoints[]`, `runs[]`.

## Contratos importantes

- **`max_speed` es una sola fuente de verdad**: vive en
  `sim_config.max_speed` de `configs/simulation/config_example.yaml`. Los
  generadores NO deben declarar `max_speed` en su YAML — lo reciben vía
  `example_cfg` en `initialize()`. Gcopter fija `drone_limits.max_velocity`
  desde ahí también.
- **Frecuencias**: 100 Hz outer loop (controller+generator), 500 Hz INDI,
  1000 Hz dinámica. Son configurables en `config_example.yaml` pero el
  código asume que `model_dt ≤ controller_dt ≤ {mpc_dt, pid_dt}`.
- **CSV schema**: 45 columnas (definidas en `framework/unified_csv_logger`).
  Incluye `controller_name`, `generator_name`,
  `controller_compute_time_us`, `generator_update_time_us`,
  `generator_eval_time_us`, `controller_delay_applied_us`,
  `generator_delay_applied_us`, `waypoint_index`, `hover_active`,
  `max_speed`. El logger emite un bloque de comentarios `# controller: …`
  antes del header. Fuente única de verdad: la constante `_COLUMN_HEADER` en
  el logger (Python) y la struct `LogRow` en C++.
- **Unidades**: SI. Posiciones en m (ENU), velocidades m/s, orientaciones en
  cuaterniones `[w, x, y, z]`, tiempos en segundos.
- **`generate.sh` es one-shot**: solo hay que rerun cuando cambian los YAMLs
  en `configs/solver_definitions/`. Los directorios
  `examples/acados_{position,trajectory}_mpc/mpc_generated_code/` son output,
  no editar.

## Workflows habituales

```bash
# Build desde cero (tras clonar)
git submodule update --init --recursive
bash generate.sh      # acados codegen, solo la primera vez
bash build.sh         # cmake build → build/

# Sanity run (un solo caso habilitado en config_example.yaml → runs[])
./build/examples/mpc_examples_run \
  -c configs/simulation/config_example.yaml \
  -s configs/simulation/config_simulator.yaml
# Produce simulator_logs/<run_id>/cpp/pid_waypoints.csv con 45 columnas

# Comparativa completa (todos los runs con enabled: true)
./scripts/run_all.sh --lang=both         # Lanza C++ y Python con el mismo run_id
# ↳ genera además metrics/summary.csv y plots/dashboard.png

# Métricas / dashboard sobre un directorio de run existente
python3 examples/utils/compute_metrics.py --run-dir simulator_logs/<run_id>
python3 scripts/dashboard.py              --run-dir simulator_logs/<run_id>

# Tests
ctest --test-dir build --output-on-failure
```

## Decisiones activas / preferencias del usuario

- README principal y `examples/README.md` en **inglés**; el resto de
  comunicación con el usuario en **español**.
- Cuando un fichero queda huérfano, **mover a `old/`** (no borrar) para que
  el usuario revise antes de eliminarlo.
- **No bumpear submódulos** a HEAD remoto salvo petición explícita; se
  mantiene el commit fijado.
- Commits sin trailer `Co-Authored-By`. Nunca commitear sin que el usuario
  lo pida.

## Gotchas

- `configs/generators/config_dynamic.yaml` es un **placeholder con solo
  comentarios**. El adaptador `dynamic` sólo comprueba existencia del
  fichero; no lo parsea como mapping.
- **Un único binario C++** (`./build/examples/mpc_examples_run`) y **un
  único script Python** (`examples/unified/run_example.py`). Los 12
  binarios legacy `mpc_examples_run_<ctrl>_<gen>` y sus subdirs
  `examples/<ctrl>_<gen>/` se han movido a `old/`.
- Los mirrors Python se publican bajo `build/python/` vía **hard-links**
  creados por CMake; editar la fuente y el build se actualiza
  automáticamente sin rebuild. `run_all.sh` añade ese path a `PYTHONPATH`.
- `simulator_logs/` está en `.gitignore`. Las salidas se estructuran por
  `run_id` (`YYYYmmdd_HHMMSS`); el unificado imprime el `run_id` elegido
  por stdout, y `--output-dir` lo puede sobreescribir.
- Los generadores `gcopter` y `dynamic`, al operar en p2p con sólo 2
  waypoints, pierden la optimización global multi-waypoint. Esto está
  asumido — es el precio de la comparativa justa. Los tiempos de cómputo
  picando en los `t_switch` del scheduler son la evidencia visible del
  replan.
- **Delay `measured`** introduce no-determinismo en las métricas wall-clock
  entre ejecuciones. Para CI/tests, forzar `controller_delay_mode: fixed`
  con `controller_delay_fixed_s: 0.0` (y análogo para el generator).
- **Modo paralelo (`sim_config.parallel: true` o flag `--parallel`)**
  lanza un worker por caso habilitado: `std::thread` en C++,
  subproceso (`ProcessPoolExecutor`) en Python. Ambos entry-points
  fuerzan internamente `silent=true` en el config que pasa a los
  workers para que las barras `\r` no se intercalen. Cuando
  `enabled_in_scope > núcleos` los hilos compiten y los
  `*_compute_time_us` se inflan; combinar paralelo con
  `*_delay_mode: measured` también contamina la métrica (warning
  emitido al arrancar). Para timing reproducible en paralelo, usar
  `*_delay_mode: fixed`.
- `thirdparty/mpc` es un único submódulo que provee position MPC y
  trajectory MPC (no son dos submódulos separados, a pesar de que el
  `.gitmodules` histórico lo sugiriera).
