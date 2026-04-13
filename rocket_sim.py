"""
Rocket Altitude Simulator
=========================
Simulates the vertical flight of a single-stage rocket including:
  • Real motor thrust curves (built-in presets or RASP .eng import)
  • Variable mass (constant mass-flow rate during burn)
  • Aerodynamic drag with ISA atmosphere model
  • Parachute deployment at apogee
  • Launch rod departure physics
  • Gravity

Usage:
    python rocket_sim.py                          # interactive prompts
    python rocket_sim.py --example                # demo with Estes C6 motor
    python rocket_sim.py --motor C6               # use a preset motor
    python rocket_sim.py --list-motors            # show available presets
    python rocket_sim.py --flight data.csv        # overlay recorded flight data
    python rocket_sim.py --thrust-curve motor.eng # load RASP .eng thrust curve

Built-in motors: A8, B6, C6, D12, E16
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# ── Physical constants ────────────────────────────────────────────────────────

G_0 = 9.80665
R_AIR = 287.058
T_SL = 288.15
P_SL = 101_325.0
LAPSE_RATE = 0.0065
TROPO_CEIL = 11_000.0

# ── Motor database ────────────────────────────────────────────────────────────
# Approximate thrust curves based on published manufacturer data.
# For precise curves, load .eng files from thrustcurve.org.
#
# Units in the database: impulse N·s, masses grams, time seconds, thrust N.

MOTOR_DB = {
    "A8": {
        "name": "Estes A8",
        "total_impulse": 2.50,
        "propellant_mass": 3.12,
        "casing_mass": 13.1,
        "burn_time": 0.32,
        "peak_thrust": 9.7,
        "delays": [3, 5],
        "thrust_curve": [
            (0.000, 0.0), (0.010, 5.0), (0.025, 9.7), (0.050, 9.8),
            (0.100, 9.5), (0.150, 8.8), (0.200, 8.0), (0.250, 6.5),
            (0.290, 3.0), (0.320, 0.0),
        ],
    },
    "B6": {
        "name": "Estes B6",
        "total_impulse": 5.00,
        "propellant_mass": 5.60,
        "casing_mass": 13.1,
        "burn_time": 0.83,
        "peak_thrust": 12.1,
        "delays": [4, 6],
        "thrust_curve": [
            (0.000, 0.0), (0.015, 3.0), (0.035, 10.0), (0.060, 12.1),
            (0.100, 11.0), (0.150, 7.5), (0.200, 6.2), (0.350, 5.8),
            (0.500, 5.5), (0.650, 5.2), (0.750, 4.0), (0.800, 2.5),
            (0.825, 1.0), (0.830, 0.0),
        ],
    },
    "C6": {
        "name": "Estes C6",
        "total_impulse": 8.82,
        "propellant_mass": 12.49,
        "casing_mass": 11.6,
        "burn_time": 1.85,
        "peak_thrust": 14.1,
        "delays": [3, 5, 7],
        "thrust_curve": [
            (0.000, 0.0), (0.020, 2.0), (0.040, 8.0), (0.060, 13.5),
            (0.080, 14.1), (0.100, 13.0), (0.150, 8.0), (0.200, 5.8),
            (0.350, 5.2), (0.600, 5.0), (0.900, 4.8), (1.200, 4.5),
            (1.500, 4.2), (1.700, 3.0), (1.800, 1.5), (1.850, 0.0),
        ],
    },
    "D12": {
        "name": "Estes D12",
        "total_impulse": 16.85,
        "propellant_mass": 21.09,
        "casing_mass": 23.1,
        "burn_time": 1.60,
        "peak_thrust": 29.7,
        "delays": [3, 5, 7],
        "thrust_curve": [
            (0.000, 0.0), (0.015, 4.0), (0.030, 14.0), (0.050, 24.0),
            (0.075, 29.7), (0.100, 28.0), (0.130, 20.0), (0.180, 12.0),
            (0.300, 9.5), (0.500, 8.5), (0.800, 8.0), (1.000, 7.5),
            (1.200, 7.0), (1.400, 5.5), (1.550, 2.5), (1.600, 0.0),
        ],
    },
    "E16": {
        "name": "Aerotech E16",
        "total_impulse": 28.45,
        "propellant_mass": 19.0,
        "casing_mass": 33.0,
        "burn_time": 1.82,
        "peak_thrust": 24.0,
        "delays": [4, 6, 8],
        "thrust_curve": [
            (0.000, 0.0), (0.020, 5.0), (0.050, 16.0), (0.080, 22.0),
            (0.100, 24.0), (0.150, 22.0), (0.200, 18.5), (0.350, 16.5),
            (0.600, 15.5), (0.900, 15.0), (1.200, 14.5), (1.400, 13.0),
            (1.600, 10.0), (1.750, 5.0), (1.820, 0.0),
        ],
    },
}


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class RocketConfig:
    """All parameters for a single-stage vertical flight simulation."""

    total_impulse: float  # N·s  (from motor datasheet)
    propellant_mass: float  # kg
    burn_time: float  # s
    motor_casing_mass: float  # kg
    thrust_curve: list[tuple[float, float]] | None = None

    m_structural: float = 0.130  # airframe mass without motor  [kg]
    C_d: float = 0.50
    body_diameter: float = 0.024  # m

    parachute_diameter: float = 0.30  # m  (0 = no chute)
    parachute_cd: float = 1.5

    rod_length: float = 1.0  # m

    def __post_init__(self):
        self.A_ref: float = np.pi * (self.body_diameter / 2) ** 2
        self.m_dry: float = self.m_structural + self.motor_casing_mass
        self.m_0: float = self.m_dry + self.propellant_mass
        self.v_exhaust: float = self.total_impulse / self.propellant_mass
        self.mdot: float = self.propellant_mass / self.burn_time
        self.avg_thrust: float = self.total_impulse / self.burn_time
        self.parachute_area: float = np.pi * (self.parachute_diameter / 2) ** 2

        if self.thrust_curve is not None:
            self._tc_t = np.array([p[0] for p in self.thrust_curve])
            self._tc_F = np.array([p[1] for p in self.thrust_curve])
        else:
            self._tc_t = None
            self._tc_F = None


# ── Atmosphere model (ISA troposphere + stratosphere) ─────────────────────────


def atmosphere(h: float) -> tuple[float, float, float]:
    """Return (temperature K, pressure Pa, density kg/m³) at altitude *h* m."""
    h = max(h, 0.0)
    if h <= TROPO_CEIL:
        T = T_SL - LAPSE_RATE * h
        p = P_SL * (T / T_SL) ** (G_0 / (LAPSE_RATE * R_AIR))
    else:
        T = 216.65
        p_tropo = P_SL * (T / T_SL) ** (G_0 / (LAPSE_RATE * R_AIR))
        p = p_tropo * np.exp(-G_0 * (h - TROPO_CEIL) / (R_AIR * T))
    rho = p / (R_AIR * T)
    return T, p, rho


# ── Thrust (with curve interpolation) ────────────────────────────────────────


def compute_thrust(t: float, cfg: RocketConfig) -> float:
    """Thrust at time *t*, using curve interpolation or constant average."""
    if t > cfg.burn_time:
        return 0.0
    if cfg._tc_t is not None:
        return float(np.interp(t, cfg._tc_t, cfg._tc_F, left=0.0, right=0.0))
    return cfg.avg_thrust


# ── RASP .eng file loader ────────────────────────────────────────────────────


def load_eng_file(filepath: str | Path) -> dict:
    """Parse a RASP .eng thrust curve file.

    Returns dict with: name, total_impulse, propellant_mass_kg,
    casing_mass_kg, burn_time, thrust_curve.
    """
    filepath = Path(filepath)
    lines = filepath.read_text().splitlines()

    header = None
    curve: list[tuple[float, float]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            if curve:
                break
            continue
        if header is None:
            header = stripped.split()
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            curve.append((float(parts[0]), float(parts[1])))

    if header is None or not curve:
        raise ValueError(f"Could not parse {filepath}")

    prop_mass_kg = float(header[4])
    total_mass_kg = float(header[5])

    times = np.array([p[0] for p in curve])
    thrusts = np.array([p[1] for p in curve])
    total_impulse = float(np.trapz(thrusts, times))

    return {
        "name": header[0],
        "total_impulse": total_impulse,
        "propellant_mass_kg": prop_mass_kg,
        "casing_mass_kg": total_mass_kg - prop_mass_kg,
        "burn_time": float(times[-1]),
        "thrust_curve": curve,
    }


# ── ODE system ────────────────────────────────────────────────────────────────


def equations_of_motion(
    t: float, y: list[float], cfg: RocketConfig
) -> list[float]:
    """dy/dt = f(t, y) with y = [h, v, m]."""
    h, v, m = y

    F_thrust = compute_thrust(t, cfg)

    dm_dt = -cfg.mdot if t <= cfg.burn_time else 0.0

    # Pad hold: rocket stays grounded until thrust exceeds weight (launch phase only)
    if h < 0.01 and v <= 0 and t < cfg.burn_time and F_thrust < m * G_0:
        return [0.0, 0.0, dm_dt]

    _, _, rho = atmosphere(h)

    on_rod = h < cfg.rod_length and v >= 0
    chute_open = v < 0 and cfg.parachute_diameter > 0

    if on_rod:
        F_drag = 0.0
    elif chute_open:
        F_drag = 0.5 * rho * v * abs(v) * cfg.parachute_cd * cfg.parachute_area
    else:
        F_drag = 0.5 * rho * v * abs(v) * cfg.C_d * cfg.A_ref

    dh_dt = v
    dv_dt = F_thrust / m - G_0 - F_drag / m

    return [dh_dt, dv_dt, dm_dt]


# ── Simulation driver ─────────────────────────────────────────────────────────


def simulate(cfg: RocketConfig, t_max: float = 300.0):
    """Integrate the equations of motion and return the solve_ivp solution."""
    y0 = [0.0, 0.0, cfg.m_0]

    def apogee_event(t, y):
        return y[1]

    apogee_event.terminal = False
    apogee_event.direction = -1

    def ground_event(t, y):
        return y[0] if t > 0.5 else 1.0

    ground_event.terminal = True
    ground_event.direction = -1

    return solve_ivp(
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


# ── Flight-data loader (Arduino / BMP280 CSV overlay) ────────────────────────


def load_flight_data(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load CSV with header row: time_ms, altitude_m."""
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    return data[:, 0] / 1000.0, data[:, 1]


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_results(sol, cfg: RocketConfig, flight_data_path=None):
    """Generate a 2×2 dashboard: altitude, velocity, thrust, mass."""
    t_fine = np.linspace(sol.t[0], sol.t[-1], 2000)
    y_fine = sol.sol(t_fine)
    h, v, m = y_fine

    F = np.array([compute_thrust(ti, cfg) for ti in t_fine])

    idx_apo = np.argmax(h)
    t_apo = t_fine[idx_apo]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Rocket Flight Simulation", fontsize=14, fontweight="bold")

    # ── Altitude ──────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(t_fine, h, "b-", linewidth=1.5, label="Simulation")
    if flight_data_path is not None:
        t_fd, h_fd = load_flight_data(flight_data_path)
        ax.plot(t_fd, h_fd, "k--", linewidth=1.2, alpha=0.7, label="Flight data")
    ax.axvline(cfg.burn_time, color="r", linestyle="--", alpha=0.4, label="Burnout")
    if cfg.parachute_diameter > 0:
        ax.axvline(t_apo, color="g", linestyle="--", alpha=0.4, label="Chute deploy")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title("Altitude vs Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Velocity ──────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(t_fine, v, color="tab:green", linewidth=1.5)
    ax.axvline(cfg.burn_time, color="r", linestyle="--", alpha=0.4, label="Burnout")
    if cfg.parachute_diameter > 0:
        ax.axvline(t_apo, color="g", linestyle="--", alpha=0.4, label="Chute deploy")
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Velocity vs Time")
    ax.legend(fontsize=8)
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
    ax.plot(t_fine, m * 1000, color="tab:purple", linewidth=1.5)
    ax.axvline(cfg.burn_time, color="r", linestyle="--", alpha=0.4, label="Burnout")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mass [g]")
    ax.set_title("Mass vs Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def print_summary(sol, cfg: RocketConfig):
    """Print key flight metrics to the console."""
    t = sol.t
    h, v, _ = sol.y

    h_max = float(np.max(h))
    idx_apogee = int(np.argmax(h))
    t_apogee = float(t[idx_apogee])
    v_max = float(np.max(v))
    t_total = float(t[-1])
    twr = cfg.avg_thrust / (cfg.m_0 * G_0)
    tsiolkovsky_dv = cfg.v_exhaust * np.log(cfg.m_0 / cfg.m_dry)

    T_sl, _, rho_sl = atmosphere(0)
    a_sl = float(np.sqrt(1.4 * R_AIR * T_sl))

    # Max dynamic pressure
    _, _, rho_arr = zip(*[atmosphere(float(hi)) for hi in h])
    q = 0.5 * np.array(rho_arr) * v**2
    q_max = float(np.max(q))

    # Rod departure velocity
    rod_v = 0.0
    for i, hi in enumerate(h):
        if hi >= cfg.rod_length:
            rod_v = float(v[i])
            break

    # Parachute terminal descent rate
    if cfg.parachute_diameter > 0 and cfg.parachute_area > 0:
        v_descent = float(
            np.sqrt(2 * cfg.m_dry * G_0 / (rho_sl * cfg.parachute_cd * cfg.parachute_area))
        )
    else:
        v_descent = 0.0

    w = 48
    print(f"\n{'=' * w}")
    print("  FLIGHT SUMMARY")
    print(f"{'=' * w}")
    print(f"  Apogee              {h_max:>10.1f} m  ({h_max * 3.28084:.0f} ft)")
    print(f"  Time to apogee      {t_apogee:>10.2f} s")
    print(f"  Max velocity        {v_max:>10.1f} m/s (Mach {v_max / a_sl:.2f})")
    print(f"  Max-Q               {q_max:>10.0f} Pa ({q_max / 1000:.1f} kPa)")
    print(f"  Total flight time   {t_total:>10.2f} s")
    print(f"{'─' * w}")
    print(f"  Total impulse       {cfg.total_impulse:>10.2f} N·s")
    print(f"  Average thrust      {cfg.avg_thrust:>10.1f} N")
    print(f"  Thrust-to-weight    {twr:>10.2f}")
    print(f"  Mass ratio (m₀/mf) {cfg.m_0 / cfg.m_dry:>10.2f}")
    print(f"  Ideal Δv            {tsiolkovsky_dv:>10.1f} m/s")
    print(f"{'─' * w}")
    print(f"  Rod departure vel.  {rod_v:>10.1f} m/s", end="")
    if 0 < rod_v < 15:
        print("  ⚠ unstable (<15 m/s)")
    else:
        print()
    if cfg.parachute_diameter > 0:
        print(f"  Descent rate        {v_descent:>10.1f} m/s (under chute)", end="")
        if v_descent > 7:
            print("  ⚠ fast")
        else:
            print()
    else:
        print(f"  Descent             {'ballistic':>10s} (no chute)")
    print(f"  Drag coefficient    {cfg.C_d:>10.3f}")
    print(f"{'=' * w}\n")


# ── Motor helpers ─────────────────────────────────────────────────────────────


def motor_from_preset(key: str) -> dict:
    """Look up a motor in the built-in database (tolerant of delay suffixes)."""
    clean = key.upper().replace("-", "").replace(" ", "")
    for name in MOTOR_DB:
        if clean.startswith(name.upper()):
            return MOTOR_DB[name]
    raise KeyError(f"Motor '{key}' not found. Available: {', '.join(MOTOR_DB)}")


def list_motors():
    """Print available motor presets."""
    print("\n  Available motors:")
    for key, m in MOTOR_DB.items():
        print(
            f"    {key:4s}  {m['name']:16s}  "
            f"{m['total_impulse']:5.1f} N·s  "
            f"{m['burn_time']:.2f}s burn  "
            f"peak {m['peak_thrust']:.0f} N"
        )
    print()


# ── CLI helpers ───────────────────────────────────────────────────────────────


def _ask_float(prompt: str, default: float | None = None) -> float:
    suffix = f" [{default}]: " if default is not None else ": "
    while True:
        raw = input(prompt + suffix).strip()
        if raw == "" and default is not None:
            return default
        try:
            val = float(raw)
            if val < 0:
                print("  → Value must be non-negative.")
                continue
            return val
        except ValueError:
            print("  → Not a valid number.")


def build_example_config() -> RocketConfig:
    """Demo rocket: Estes C6 motor, 130 g structure, 30 cm chute."""
    m = MOTOR_DB["C6"]
    return RocketConfig(
        total_impulse=m["total_impulse"],
        propellant_mass=m["propellant_mass"] / 1000,
        burn_time=m["burn_time"],
        motor_casing_mass=m["casing_mass"] / 1000,
        thrust_curve=m["thrust_curve"],
        m_structural=0.130,
        C_d=0.50,
        body_diameter=0.024,
        parachute_diameter=0.30,
        parachute_cd=1.5,
        rod_length=1.0,
    )


def interactive_config() -> RocketConfig:
    """Walk the user through parameter entry."""
    print(f"\n{'=' * 52}")
    print("  ROCKET PARAMETER INPUT")
    print("  (press Enter to accept the default in brackets)")
    print(f"{'=' * 52}")

    list_motors()
    motor_key = input("  Motor preset (or 'custom') [C6]: ").strip() or "C6"

    if motor_key.lower() == "custom":
        total_impulse = _ask_float("  Total impulse [N·s]", 8.82)
        prop_mass_g = _ask_float("  Propellant mass [g]", 12.5)
        burn_time = _ask_float("  Burn time [s]", 1.85)
        casing_mass_g = _ask_float("  Motor casing mass [g]", 11.6)
        thrust_curve = None
    else:
        try:
            m = motor_from_preset(motor_key)
        except KeyError as e:
            print(f"  {e}")
            return interactive_config()
        total_impulse = m["total_impulse"]
        prop_mass_g = m["propellant_mass"]
        burn_time = m["burn_time"]
        casing_mass_g = m["casing_mass"]
        thrust_curve = m["thrust_curve"]
        print(
            f"  → {m['name']}: {total_impulse:.1f} N·s, "
            f"{burn_time:.2f}s burn, peak {m['peak_thrust']:.0f} N"
        )

    print()
    m_struct_g = _ask_float("  Structural mass (no motor) [g]", 130.0)
    d_body_mm = _ask_float("  Body tube diameter [mm]", 24.0)
    C_d = _ask_float("  Drag coefficient", 0.50)
    chute_cm = _ask_float("  Parachute diameter [cm] (0 = none)", 30.0)
    rod_m = _ask_float("  Launch rod length [m]", 1.0)

    cfg = RocketConfig(
        total_impulse=total_impulse,
        propellant_mass=prop_mass_g / 1000,
        burn_time=burn_time,
        motor_casing_mass=casing_mass_g / 1000,
        thrust_curve=thrust_curve,
        m_structural=m_struct_g / 1000,
        C_d=C_d,
        body_diameter=d_body_mm / 1000,
        parachute_diameter=chute_cm / 100,
        parachute_cd=1.5,
        rod_length=rod_m,
    )

    twr = cfg.avg_thrust / (cfg.m_0 * G_0)
    print(f"\n  → Liftoff mass:     {cfg.m_0 * 1000:.1f} g")
    print(f"  → Dry mass:         {cfg.m_dry * 1000:.1f} g")
    print(f"  → Avg thrust:       {cfg.avg_thrust:.1f} N")
    print(f"  → Thrust-to-weight: {twr:.2f}")
    if twr < 1.0:
        print("  ⚠  TWR < 1 — this rocket will not lift off!")
    print()
    return cfg


# ── Main entry point ──────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Rocket altitude simulator")
    ap.add_argument("--example", action="store_true", help="Demo with Estes C6")
    ap.add_argument("--motor", type=str, default=None, help="Motor preset name")
    ap.add_argument("--flight", type=str, default=None, help="CSV overlay path")
    ap.add_argument("--thrust-curve", type=str, default=None, help="RASP .eng file")
    ap.add_argument("--list-motors", action="store_true", help="Show presets")
    args = ap.parse_args()

    if args.list_motors:
        list_motors()
        return

    eng_data = None
    if args.thrust_curve:
        eng_data = load_eng_file(args.thrust_curve)
        print(f"Thrust curve loaded from {args.thrust_curve} ({eng_data['name']})")

    if args.example:
        cfg = build_example_config()
        if eng_data:
            cfg.thrust_curve = eng_data["thrust_curve"]
            cfg._tc_t = np.array([p[0] for p in eng_data["thrust_curve"]])
            cfg._tc_F = np.array([p[1] for p in eng_data["thrust_curve"]])
        print("Using demo rocket (Estes C6, 130 g structure, 30 cm chute).\n")

    elif args.motor:
        try:
            m = motor_from_preset(args.motor)
        except KeyError as e:
            print(e)
            list_motors()
            return
        tc = eng_data["thrust_curve"] if eng_data else m["thrust_curve"]
        cfg = RocketConfig(
            total_impulse=m["total_impulse"],
            propellant_mass=m["propellant_mass"] / 1000,
            burn_time=m["burn_time"],
            motor_casing_mass=m["casing_mass"] / 1000,
            thrust_curve=tc,
        )
        print(f"Using {m['name']} motor with default rocket parameters.\n")

    elif eng_data and not args.motor:
        cfg = RocketConfig(
            total_impulse=eng_data["total_impulse"],
            propellant_mass=eng_data["propellant_mass_kg"],
            burn_time=eng_data["burn_time"],
            motor_casing_mass=eng_data["casing_mass_kg"],
            thrust_curve=eng_data["thrust_curve"],
        )
        print(f"Motor '{eng_data['name']}' — all params from .eng file.\n")

    else:
        cfg = interactive_config()
        if eng_data:
            cfg.thrust_curve = eng_data["thrust_curve"]
            cfg._tc_t = np.array([p[0] for p in eng_data["thrust_curve"]])
            cfg._tc_F = np.array([p[1] for p in eng_data["thrust_curve"]])

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
