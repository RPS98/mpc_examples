#!/usr/bin/env python3
"""Debugger rápido (1D) de referencias por etapa del MPC.

- Sin argparse.
- Tres casos explícitos en main.
- Un figure por caso.
- Subplot superior: posición de referencia en X.
- Subplot inferior: velocidad implícita entre etapas.
"""

import numpy as np
import matplotlib.pyplot as plt


def progressive_references_1d(x0: float, x_goal: float, N: int, tf: float, v_ref: float) -> tuple[np.ndarray, float]:
    """Misma lógica de referencias por etapa que el MPC, pero en 1D (eje X)."""
    dt_h = tf / N
    refs = np.zeros(N + 1, dtype=float)

    dx = x_goal - x0
    L = abs(dx)

    if L < 1e-9:
        refs[:] = x_goal
        return refs, dt_h

    direction = 1.0 if dx >= 0.0 else -1.0
    for k in range(N + 1):
        s_k = min((k + 1) * v_ref * dt_h, L)
        refs[k] = x0 + direction * s_k

    return refs, dt_h


def reference_velocity_profile_1d(x0: float, refs: np.ndarray, dt_h: float) -> np.ndarray:
    """Velocidad implícita entre referencias consecutivas."""
    v = np.zeros_like(refs)
    prev = x0
    for k in range(refs.size):
        v[k] = (refs[k] - prev) / dt_h
        prev = refs[k]
    return v


def first_saturation_stage(refs: np.ndarray, x_goal: float) -> int | None:
    idx = np.where(np.isclose(refs, x_goal, atol=1e-9))[0]
    if idx.size == 0:
        return None
    return int(idx[0])


def run_case(case_name: str, x0: float, x_goal: float, N: int, tf: float, v_ref: float) -> None:
    refs, dt_h = progressive_references_1d(x0=x0, x_goal=x_goal, N=N, tf=tf, v_ref=v_ref)
    v_refs = reference_velocity_profile_1d(x0=x0, refs=refs, dt_h=dt_h)

    k = np.arange(N + 1)
    horizon_dist = (N + 1) * dt_h * v_ref
    sat_k = first_saturation_stage(refs, x_goal)

    print(f"\n=== {case_name} ===")
    print(f"x0={x0:.4f}, x_goal={x_goal:.4f}, N={N}, tf={tf:.4f}, dt_h={dt_h:.4f}, v_ref={v_ref:.4f}")
    print(f"distancia objetivo = {abs(x_goal - x0):.4f} m")
    print(f"distancia máxima horizonte = {horizon_dist:.4f} m")
    if sat_k is None:
        print("saturación: NO (no llega al goal dentro del horizonte)")
    else:
        print(f"saturación: SÍ, desde k={sat_k} (se queda quieta en x_goal)")

    print("refs primeras 6:", np.array2string(refs[:6], precision=4, suppress_small=True))
    print("refs últimas 6 :", np.array2string(refs[-6:], precision=4, suppress_small=True))
    print("v primeras 6   :", np.array2string(v_refs[:6], precision=4, suppress_small=True))
    print("v últimas 6    :", np.array2string(v_refs[-6:], precision=4, suppress_small=True))

    fig, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Posición de referencia (arriba)
    ax_pos.plot(k, refs, "o-", lw=1.5, ms=4, label="x_ref[k]")
    ax_pos.axhline(x_goal, color="tab:red", ls="--", lw=1.2, label="x_goal")
    ax_pos.axhline(x0, color="tab:green", ls=":", lw=1.2, label="x0")
    if sat_k is not None:
        ax_pos.axvline(sat_k, color="tab:orange", ls="--", lw=1.2, label=f"sat@k={sat_k}")
    ax_pos.set_ylabel("Posición X [m]")
    ax_pos.grid(True, alpha=0.3)
    ax_pos.legend(loc="best")
    ax_pos.set_title(
        f"{case_name} | x0={x0:.2f} -> x_goal={x_goal:.2f} | N={N}, tf={tf:.2f}, v_ref={v_ref:.2f}"
    )

    # Velocidad implícita (abajo)
    ax_vel.step(k, v_refs, where="post", lw=1.7, label="v_ref_implícita[k]")
    ax_vel.axhline(0.0, color="black", lw=0.9)
    if sat_k is not None:
        ax_vel.axvline(sat_k, color="tab:orange", ls="--", lw=1.2, label=f"sat@k={sat_k}")
    ax_vel.set_xlabel("Etapa k")
    ax_vel.set_ylabel("Velocidad X [m/s]")
    ax_vel.grid(True, alpha=0.3)
    ax_vel.legend(loc="best")

    fig.tight_layout()


def main() -> None:
    # Caso 1: fuera del horizonte (no alcanza el objetivo en N+1 referencias)
    run_case(
        case_name="fuera_horizonte",
        x0=0.0,
        x_goal=10.0,
        N=30,
        tf=2.0,
        v_ref=1.0,
    )

    # Caso 2: dentro del horizonte (alcanza pronto y luego se queda quieta en x_goal)
    run_case(
        case_name="dentro_horizonte",
        x0=0.0,
        x_goal=0.8,
        N=30,
        tf=2.0,
        v_ref=1.0,
    )

    # Caso 3: cerca del límite del horizonte (alcanza casi al final)
    run_case(
        case_name="cerca_limite_horizonte",
        x0=0.0,
        x_goal=2.025,
        N=30,
        tf=2.0,
        v_ref=1.0,
    )

    # Muestra los 3 figures (uno por caso)
    plt.show()


if __name__ == "__main__":
    main()
