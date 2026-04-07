"""
Rocket Altitude Simulator
=========================
Simulates the vertical flight of a single-stage rocket including:
  • Variable mass (constant mass-flow rate during burn)
  • Thrust with altitude-dependent pressure correction
  • Aerodynamic drag with ISA atmosphere model
  • Gravity

Usage:
    python rocket_sim.py                      # interactive — prompts for rocket params
    python rocket_sim.py --example            # skip prompts, use built-in demo rocket
    python rocket_sim.py --flight data.csv    # overlay Arduino flight data on the plot
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# ── Physical constants ────────────────────────────────────────────────────────

G_0 = 9.80665  # standard gravitational acceleration  [m/s²]
R_AIR = 287.058  # specific gas constant for dry air   [J/(kg·K)]
T_SL = 288.15  # sea-level ISA temperature             [K]
P_SL = 101_325.0  # sea-level ISA pressure             [Pa]
LAPSE_RATE = 0.0065  # tropospheric temperature lapse   [K/m]
TROPO_CEIL = 11_000.0  # tropopause altitude            [m]


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class RocketConfig:
    """Define every parameter needed for a single-stage vertical flight."""

    m_propellant: float  # propellant mass                           [kg]
    m_dry: float  # structural (dry) mass                            [kg]
    A_throat: float  # nozzle throat area                            [m²]
    A_exit: float  # nozzle exit area                                [m²]
    v_exhaust: float  # effective exhaust velocity (sea-level ref.)  [m/s]
    t_burn: float  # burn duration                                   [s]
    C_d: float  # drag coefficient                                   [–]
    A_ref: float | None = None  # aerodynamic reference area         [m²]

    def __post_init__(self):
        if self.A_ref is None:
            self.A_ref = self.A_exit
        self.m_0: float = self.m_propellant + self.m_dry
        self.mdot: float = self.m_propellant / self.t_burn
        self.expansion_ratio: float = self.A_exit / self.A_throat


# ── Atmosphere model (ISA troposphere + stratosphere) ─────────────────────────


def atmosphere(h: float) -> tuple[float, float, float]:
    """Return (temperature [K], pressure [Pa], density [kg/m³]) at altitude *h* [m].

    Uses the International Standard Atmosphere for the troposphere (0–11 km)
    and lower stratosphere (11–20 km).
    """
    h = max(h, 0.0)

    if h <= TROPO_CEIL:
        T = T_SL - LAPSE_RATE * h
        p = P_SL * (T / T_SL) ** (G_0 / (LAPSE_RATE * R_AIR))
    else:
        # stratosphere: isothermal at 216.65 K
        T = 216.65
        p_tropo = P_SL * (T / T_SL) ** (G_0 / (LAPSE_RATE * R_AIR))
        p = p_tropo * np.exp(-G_0 * (h - TROPO_CEIL) / (R_AIR * T))

    rho = p / (R_AIR * T)
    return T, p, rho


# ── Force models ──────────────────────────────────────────────────────────────


def compute_thrust(t: float, h: float, cfg: RocketConfig) -> float:
    """Thrust at time *t* and altitude *h*.

    F = mdot·v_e(eff) + (p_SL − p_atm(h))·A_exit
    The second term accounts for the increase in thrust as ambient pressure
    falls with altitude (v_exhaust is referenced to sea-level conditions).
    """
    if t > cfg.t_burn:
        return 0.0
    _, p_atm, _ = atmosphere(h)
    return cfg.mdot * cfg.v_exhaust + (P_SL - p_atm) * cfg.A_exit


# ── ODE system ────────────────────────────────────────────────────────────────


def equations_of_motion(t: float, y: list[float], cfg: RocketConfig) -> list[float]:
    """Right-hand side of  dy/dt = f(t, y)  with  y = [h, v, m]."""
    h, v, m = y

    F_thrust = compute_thrust(t, h, cfg)

    _, _, rho = atmosphere(h)
    F_drag = 0.5 * rho * v * abs(v) * cfg.C_d * cfg.A_ref

    dm_dt = -cfg.mdot if t <= cfg.t_burn else 0.0
    dh_dt = v
    dv_dt = F_thrust / m - G_0 - F_drag / m

    return [dh_dt, dv_dt, dm_dt]


# ── Simulation driver ─────────────────────────────────────────────────────────


def simulate(cfg: RocketConfig, t_max: float = 300.0):
    """Integrate the equations of motion and return the *solve_ivp* solution."""
    y0 = [0.0, 0.0, cfg.m_0]

    def apogee_event(t, y):
        return y[1]

    apogee_event.terminal = False
    apogee_event.direction = -1

    def ground_event(t, y):
        return y[0]

    ground_event.terminal = True
    ground_event.direction = -1

    sol = solve_ivp(
        fun=lambda t, y: equations_of_motion(t, y, cfg),
        t_span=(0, t_max),
        y0=y0,
        method="RK45",
        events=[apogee_event, ground_event],
        max_step=0.1,
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
    )
    return sol


# ── Flight-data loader (for Arduino / BMP280 CSV overlay) ────────────────────


def load_flight_data(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load recorded flight data from CSV.

    Expected format (first row is a header):
        time_ms, altitude_m
    Returns arrays (t_seconds, altitude_metres).
    """
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    t = data[:, 0] / 1000.0
    h = data[:, 1]
    return t, h


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_results(
    sol,
    cfg: RocketConfig,
    flight_data_path: str | Path | None = None,
):
    """Generate a 2×2 dashboard: altitude, velocity, thrust, mass."""
    t_fine = np.linspace(sol.t[0], sol.t[-1], 2000)
    y_fine = sol.sol(t_fine)
    h, v, m = y_fine

    F = np.array([compute_thrust(ti, hi, cfg) for ti, hi in zip(t_fine, h)])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Rocket Flight Simulation", fontsize=14, fontweight="bold")

    # ── Altitude ──────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(t_fine, h, "b-", linewidth=1.5, label="Simulation")
    if flight_data_path is not None:
        t_fd, h_fd = load_flight_data(flight_data_path)
        ax.plot(t_fd, h_fd, "k--", linewidth=1.2, alpha=0.7, label="Flight data")
    ax.axvline(cfg.t_burn, color="r", linestyle="--", alpha=0.4, label="Burnout")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Altitude vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Velocity ──────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(t_fine, v, color="tab:green", linewidth=1.5)
    ax.axvline(cfg.t_burn, color="r", linestyle="--", alpha=0.4, label="Burnout")
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Velocity vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Thrust curve ──────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(t_fine, F, color="tab:red", linewidth=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Thrust [N]")
    ax.set_title("Thrust Curve")
    ax.grid(True, alpha=0.3)

    # ── Mass ──────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(t_fine, m, color="tab:purple", linewidth=1.5)
    ax.axvline(cfg.t_burn, color="r", linestyle="--", alpha=0.4, label="Burnout")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mass [kg]")
    ax.set_title("Mass vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def print_summary(sol, cfg: RocketConfig):
    """Print key flight metrics to the console."""
    t = sol.t
    h, v, m = sol.y

    h_max = np.max(h)
    idx_apogee = np.argmax(h)
    t_apogee = t[idx_apogee]
    v_max = np.max(v)
    t_total = t[-1]
    F_sl = cfg.mdot * cfg.v_exhaust
    tsiolkovsky_dv = cfg.v_exhaust * np.log(cfg.m_0 / cfg.m_dry)
    twr = F_sl / (cfg.m_0 * G_0)

    width = 44
    print(f"\n{'=' * width}")
    print(f"  FLIGHT SUMMARY")
    print(f"{'=' * width}")
    print(f"  Apogee              {h_max:>10.1f} m  ({h_max * 3.28084:.0f} ft)")
    print(f"  Time to apogee      {t_apogee:>10.2f} s")
    print(f"  Max velocity        {v_max:>10.1f} m/s (Mach {v_max / 343:.2f})")
    print(f"  Total flight time   {t_total:>10.2f} s")
    print(f"{'─' * width}")
    print(f"  Thrust (sea level)  {F_sl:>10.1f} N")
    print(f"  Thrust-to-weight    {twr:>10.2f}")
    print(f"  Mass ratio (m₀/mf) {cfg.m_0 / cfg.m_dry:>10.2f}")
    print(f"  Expansion ratio     {cfg.expansion_ratio:>10.2f}")
    print(f"  Ideal Δv (Tsiolkovsky) {tsiolkovsky_dv:>7.1f} m/s")
    print(f"{'=' * width}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────


def build_example_config() -> RocketConfig:
    """A small amateur solid-motor rocket for demonstration."""
    return RocketConfig(
        m_propellant=0.050,  # 50 g propellant (e.g. small APCP grain)
        m_dry=0.150,  # 150 g airframe + motor casing
        A_throat=50e-6,  # 50 mm² throat
        A_exit=200e-6,  # 200 mm² exit  (ε = 4)
        v_exhaust=1500.0,  # ≈ Isp 153 s — typical small APCP motor
        t_burn=1.5,  # 1.5 s burn
        C_d=0.5,  # typical subsonic rocket Cd
        A_ref=4.9e-4,  # ~25 mm body-tube diameter
    )


# ── Interactive input ─────────────────────────────────────────────────────────


def _ask_float(prompt: str, default: float | None = None) -> float:
    """Prompt the user for a float, with an optional default shown in brackets."""
    suffix = f" [{default}]: " if default is not None else ": "
    while True:
        raw = input(prompt + suffix).strip()
        if raw == "" and default is not None:
            return default
        try:
            value = float(raw)
            if value <= 0:
                print("  → Value must be positive. Try again.")
                continue
            return value
        except ValueError:
            print("  → Not a valid number. Try again.")


def interactive_config() -> RocketConfig:
    """Walk the user through entering every rocket parameter."""
    print("\n" + "=" * 52)
    print("  ROCKET PARAMETER INPUT")
    print("  (press Enter to accept the default in brackets)")
    print("=" * 52 + "\n")

    m_prop = _ask_float("  Propellant mass [g]", 50.0) / 1000.0
    m_dry = _ask_float("  Dry mass (no propellant) [g]", 150.0) / 1000.0
    A_throat = _ask_float("  Nozzle throat area [mm²]", 50.0) * 1e-6
    A_exit = _ask_float("  Nozzle exit area [mm²]", 200.0) * 1e-6
    v_exhaust = _ask_float("  Exhaust velocity [m/s]", 1500.0)
    t_burn = _ask_float("  Burn time [s]", 1.5)
    C_d = _ask_float("  Drag coefficient", 0.5)
    d_body = _ask_float("  Body tube diameter [mm]", 25.0) / 1000.0
    A_ref = np.pi * (d_body / 2) ** 2

    cfg = RocketConfig(
        m_propellant=m_prop,
        m_dry=m_dry,
        A_throat=A_throat,
        A_exit=A_exit,
        v_exhaust=v_exhaust,
        t_burn=t_burn,
        C_d=C_d,
        A_ref=A_ref,
    )

    print(f"\n  → Total mass:       {cfg.m_0 * 1000:.1f} g")
    print(f"  → Mass flow rate:   {cfg.mdot * 1000:.2f} g/s")
    print(f"  → Expansion ratio:  {cfg.expansion_ratio:.2f}")
    print(f"  → Sea-level thrust: {cfg.mdot * cfg.v_exhaust:.1f} N")
    twr = (cfg.mdot * cfg.v_exhaust) / (cfg.m_0 * G_0)
    print(f"  → Thrust-to-weight: {twr:.2f}")
    if twr < 1.0:
        print("  ⚠  TWR < 1 — this rocket will not lift off!")
    print()
    return cfg


# ── CLI entry point ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Rocket altitude simulator")
    parser.add_argument(
        "--example",
        action="store_true",
        help="Skip prompts and use built-in example rocket",
    )
    parser.add_argument(
        "--flight",
        type=str,
        default=None,
        help="Path to CSV flight data (time_ms, altitude_m) for overlay",
    )
    args = parser.parse_args()

    if args.example:
        cfg = build_example_config()
        print("Using built-in example rocket.\n")
    else:
        cfg = interactive_config()

    print("Running simulation …")
    sol = simulate(cfg)

    if not sol.success and len(sol.t_events[1]) == 0:
        print(f"⚠  Solver did not converge: {sol.message}")
        return

    print_summary(sol, cfg)
    plot_results(sol, cfg, flight_data_path=args.flight)
    plt.show()


if __name__ == "__main__":
    main()
