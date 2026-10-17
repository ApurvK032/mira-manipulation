"""Scripted test scenarios: cubes moved by a hand, slid, teleported, stacked and knocked over.

Every scenario is a continuous function of time, so the renderer can sample sub-frames for motion
blur. Events (releases, teleports, slides) are recorded so the evaluator knows what to measure.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from perception.geometry import HALF, CUBE_SIZE, make_T, rot_y, rot_z

WORK_X, WORK_Y = (-0.18, 0.18), (-0.12, 0.14)
GRASP_Z = HALF + 0.012 + 0.002  # palm center above the cube center


def minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def interp(Ta, Tb, s, lift=0.0):
    p = (1 - s) * Ta[:3, 3] + s * Tb[:3, 3]
    p[2] += lift * 4 * s * (1 - s)
    rv = Rotation.from_matrix(Ta[:3, :3].T @ Tb[:3, :3]).as_rotvec()
    return make_T(Ta[:3, :3] @ Rotation.from_rotvec(rv * s).as_matrix(), p)


def cube_T(x, y, yaw_deg, z=HALF):
    return make_T(rot_z(np.radians(yaw_deg)), (x, y, z))


class Track:
    """Piecewise pose trajectory that holds the last pose between segments. None = not in the scene."""

    def __init__(self, T0):
        self.T0 = T0
        self.segs = []  # (t_start, t_end, fn, end_pose)

    def add(self, ta, tb, fn):
        self.segs.append((ta, tb, fn, fn(tb)))

    def move(self, ta, tb, Tb, lift=0.0, Ta=None):
        Ta = self.pose(ta) if Ta is None else Ta
        self.add(ta, tb, lambda t: interp(Ta, Tb, minjerk((t - ta) / (tb - ta)), lift))

    def set(self, t, T):
        self.add(t, t + 1e-9, lambda _t: T)

    def pose(self, t):
        T = self.T0
        for ta, tb, fn, Tend in self.segs:
            if t < ta:
                break
            T = fn(t) if t < tb else Tend
        return T


@dataclass
class SceneState:
    cubes: dict
    hand: np.ndarray | None
    distractor: np.ndarray | None
    light: float
    tape: np.ndarray | None = None  # flat red square on the table: red like a cube, but not a cube


@dataclass
class Event:
    kind: str  # release | teleport | slide | knock | static
    cube: str
    t: float  # time the cube is at rest and free again (or start of a static window)
    t_end: float = 0.0
    detail: dict = field(default_factory=dict)


class Scenario:
    def __init__(self, name, cubes, duration, seed=0):
        self.name = name
        self.duration = duration
        self.rng = np.random.default_rng(seed)
        self.cubes = {n: Track(T) for n, T in cubes.items()}
        self.hand = Track(None)
        self.distractor = Track(None)
        self.tape = Track(None)
        self.light_fn = lambda t: 1.0
        self.events: list[Event] = []

    # ----- queries used by the renderer -----
    def state(self, t) -> SceneState:
        return SceneState({n: tr.pose(t) for n, tr in self.cubes.items()}, self.hand.pose(t),
                          self.distractor.pose(t), float(self.light_fn(t)), self.tape.pose(t))

    def max_speed(self, t, dt=1e-3) -> float:
        best = 0.0
        for tr in [*self.cubes.values(), self.hand]:
            a, b = tr.pose(t - dt), tr.pose(t + dt)
            if a is not None and b is not None:
                best = max(best, np.linalg.norm(b[:3, 3] - a[:3, 3]) / (2 * dt))
        return best

    # ----- building blocks -----
    def _hand_grasp(self, Tc, approach_yaw):
        # palm above the cube, fingers on two opposite faces: hand yaw = cube yaw + k * 90 deg
        cube_yaw = np.arctan2(Tc[1, 0], Tc[0, 0])
        k = np.round((approach_yaw - cube_yaw) / (np.pi / 2))
        return make_T(rot_z(cube_yaw + k * np.pi / 2), Tc[:3, 3] + (0, 0, GRASP_Z))

    @staticmethod
    def _offset(T, local):
        return make_T(T[:3, :3], T[:3, 3] + T[:3, :3] @ np.asarray(local, float))

    def pick_place(self, name, t0, target, peak_speed=1.0, lift=0.06, approach_yaw=None):
        cube = self.cubes[name]
        Tc0, Tc1 = cube.pose(t0), target
        yaw = self.rng.uniform(-np.pi, np.pi) if approach_yaw is None else approach_yaw
        Hg0 = self._hand_grasp(Tc0, yaw)
        T_cube_hand = np.linalg.inv(Tc0) @ Hg0
        rest0 = self._offset(Hg0, (-0.55, 0, 0.35))
        pre0 = self._offset(Hg0, (0, 0, 0.05))
        dist = np.linalg.norm(Tc1[:3, 3] - Tc0[:3, 3]) + lift
        t_carry = max(0.15, 1.875 * dist / peak_speed)

        t = t0
        self.hand.move(t, t + 0.5, pre0, Ta=rest0); t += 0.5
        self.hand.move(t, t + 0.15, Hg0); t += 0.15
        t += 0.08  # close the hand
        ta = t
        cube.move(ta, ta + t_carry, Tc1, lift=lift)
        self.hand.add(ta, ta + t_carry, lambda tt: cube.pose(tt) @ T_cube_hand)
        t += t_carry
        t_place = t
        Hg1 = Tc1 @ T_cube_hand
        t += 0.08  # open the hand
        self.hand.move(t, t + 0.15, self._offset(Hg1, (0, 0, 0.05))); t += 0.15
        t_free = t
        self.hand.move(t, t + 0.4, self._offset(Hg1, (-0.55, 0, 0.35))); t += 0.4
        self.hand.set(t, None)
        self.events.append(Event("release", name, t_free, t, {"peak_speed": peak_speed, "t_grab": ta, "t_place": t_place}))
        return t

    def slide(self, name, t0, dxy, duration, dyaw_deg=0.0):
        """Cube pushed along the table with no hand in view (clean small-motion test)."""
        cube = self.cubes[name]
        Ta = cube.pose(t0)
        Tb = make_T(rot_z(np.radians(dyaw_deg)) @ Ta[:3, :3], Ta[:3, 3] + (dxy[0], dxy[1], 0))
        cube.move(t0, t0 + duration, Tb)
        self.events.append(Event("slide", name, t0, t0 + duration, {"dxy": tuple(dxy), "dyaw": dyaw_deg}))
        return t0 + duration

    def teleport(self, name, t, target):
        self.cubes[name].set(t, target)
        self.events.append(Event("teleport", name, t, t))
        return t

    def knock_off(self, name, t0, push_yaw):
        """Hand sweeps the (stacked) cube off; it tips over onto the table."""
        cube = self.cubes[name]
        Tc0 = cube.pose(t0)
        d = np.array([np.cos(push_yaw), np.sin(push_yaw), 0.0])
        Hstart = make_T(rot_z(push_yaw), Tc0[:3, 3] - d * (HALF + 0.047) + (0, 0, GRASP_Z - 0.005))
        t = t0
        self.hand.move(t, t + 0.5, Hstart, Ta=self._offset(Hstart, (-0.55, 0, 0.35))); t += 0.5
        Hend = make_T(Hstart[:3, :3], Hstart[:3, 3] + d * 0.07)
        self.hand.move(t, t + 0.14, Hend)
        R0, p0, z0 = Tc0[:3, :3], Tc0[:3, 3].copy(), Tc0[2, 3]
        # tip over one of the cube's own horizontal axes (the one closest to the push), so it lands face-up
        want = np.cross([0, 0, 1.0], d)
        k = int(np.argmax(np.abs(R0.T @ want)))
        axis = R0[:, k] * np.sign(R0[:, k] @ want)
        ta, tb = t + 0.04, t + 0.4

        def fall(tt, ta=ta, tb=tb):
            s = np.clip((tt - ta) / (tb - ta), 0, 1)
            R = Rotation.from_rotvec(axis * (np.pi / 2) * minjerk(s)).as_matrix() @ R0
            p = p0 + d * 0.085 * minjerk(s)
            p[2] = z0 - (z0 - HALF) * s**2
            return make_T(R, p)

        cube.add(ta, tb, fall)
        t = tb
        self.hand.move(t, t + 0.4, self._offset(Hend, (-0.55, 0, 0.35))); t += 0.4
        self.hand.set(t, None)
        self.events.append(Event("knock", name, tb, tb))
        return t

    def wave_hand(self, t0, duration, near):
        """Hand moves around and over the given world points without touching the cubes."""
        t = t0
        pts = [np.array(p, float) for p in near]
        H = None
        while t < t0 + duration:
            p = pts[self.rng.integers(len(pts))] + np.r_[self.rng.uniform(-0.05, 0.05, 2), 0]
            yaw = self.rng.uniform(-np.pi, np.pi)
            Ht = make_T(rot_z(yaw), (p[0], p[1], self.rng.uniform(0.09, 0.16)))
            dur = self.rng.uniform(0.35, 0.8)
            self.hand.move(t, t + dur, Ht, Ta=H if H is not None else self._offset(Ht, (-0.55, 0, 0.35)))
            H = Ht
            t += dur
        self.hand.move(t, t + 0.4, self._offset(H, (-0.55, 0, 0.35))); t += 0.4
        self.hand.set(t, None)
        return t

    def random_free_spot(self, t, exclude=(), min_dist=0.09):
        others = [tr.pose(t)[:3, 3] for n, tr in self.cubes.items() if n not in exclude and tr.pose(t) is not None]
        for _ in range(200):
            x, y = self.rng.uniform(*WORK_X), self.rng.uniform(*WORK_Y)
            if all(np.hypot(x - o[0], y - o[1]) > min_dist for o in others):
                return x, y
        raise RuntimeError("no free spot")


# --------------------------------------------------------------------------------------------
def make_scenarios(seed=0):
    S = []

    s = Scenario("static", {"blue": cube_T(-0.07, 0.03, 20), "red": cube_T(0.08, -0.04, -35)}, 3.0, seed)
    s.events.append(Event("static", "blue", 0.0, 3.0))
    s.events.append(Event("static", "red", 0.0, 3.0))
    S.append(s)

    s = Scenario("small_motion", {"blue": cube_T(-0.05, 0.02, 10), "red": cube_T(0.09, -0.03, -20)}, 7.0, seed + 1)
    t = 1.0
    t = s.slide("blue", t, (0.010, 0.0), 0.5) + 1.0
    t = s.slide("blue", t, (0.0, 0.010), 0.5) + 1.0
    t = s.slide("blue", t, (0.002, 0.0), 0.2) + 1.0
    t = s.slide("red", t, (0.0, 0.0), 0.3, dyaw_deg=5.0) + 1.0
    S.append(s)

    s = Scenario("hand_pick_place", {"blue": cube_T(-0.10, 0.05, 15), "red": cube_T(0.10, -0.05, -10)}, 9.0, seed + 2)
    t = s.pick_place("blue", 0.5, cube_T(0.02, -0.08, 45), peak_speed=1.0) + 0.4
    t = s.pick_place("red", t, cube_T(-0.12, 0.10, 0), peak_speed=1.5) + 0.4
    t = s.pick_place("blue", t, cube_T(0.13, 0.08, -30), peak_speed=2.0) + 0.4
    s.duration = t + 0.5
    S.append(s)

    s = Scenario("fast_snatch", {"blue": cube_T(-0.12, -0.06, 0), "red": cube_T(0.06, 0.08, 30)}, 5.0, seed + 3)
    t = s.pick_place("blue", 0.5, cube_T(0.15, -0.05, 60), peak_speed=3.0, lift=0.10) + 0.4
    t = s.pick_place("red", t, cube_T(-0.14, 0.12, -15), peak_speed=3.0, lift=0.10) + 0.4
    s.duration = t + 0.5
    S.append(s)

    s = Scenario("teleport", {"blue": cube_T(-0.05, 0.0, 0), "red": cube_T(0.1, 0.05, 25)}, 7.0, seed + 4)
    t = 0.8
    for i in range(6):
        n = "blue" if i % 2 == 0 else "red"
        x, y = s.random_free_spot(t, exclude=(n,))
        s.teleport(n, t, cube_T(x, y, s.rng.uniform(-45, 45)))
        t += 1.0
    S.append(s)

    s = Scenario("stack_knock", {"blue": cube_T(-0.10, 0.02, 10), "red": cube_T(0.05, -0.03, -20)}, 8.0, seed + 5)
    red = s.cubes["red"].pose(0)
    t = s.pick_place("blue", 0.5, make_T(red[:3, :3], red[:3, 3] + (0, 0, CUBE_SIZE)), peak_speed=1.0, lift=0.08)
    s.events.append(Event("stacked", "blue", t, t + 1.5, {"on": "red"}))
    t = s.knock_off("blue", t + 1.5, push_yaw=np.radians(170)) + 1.2
    s.duration = t
    S.append(s)

    s = Scenario("clutter_light", {"blue": cube_T(-0.08, 0.03, 20), "red": cube_T(0.07, -0.04, -35),
                                   "green": cube_T(0.13, 0.09, 5), "yellow": cube_T(-0.14, -0.08, 40)}, 10.0, seed + 6)
    s.distractor.set(0.0, make_T(rot_z(0.5), (0.0, 0.10, 0.012)))
    s.tape.set(0.0, make_T(rot_z(0.3), (-0.15, 0.08, 0.0008)))
    s.wave_hand(0.5, 8.5, near=[(-0.08, 0.03, 0), (0.07, -0.04, 0), (0.0, 0.1, 0)])
    s.light_fn = lambda t: 0.35 if 3.0 <= t < 5.0 else (0.75 + 0.25 * np.sin(2 * np.pi * 3 * t) if 6.0 <= t < 7.5 else 1.0)
    for n in ("blue", "red", "green", "yellow"):
        s.events.append(Event("static", n, 0.0, 10.0, {"cluttered": True}))
    S.append(s)

    S.append(random_stress(seed + 7))
    return S


def random_stress(seed, duration=40.0):
    rng = np.random.default_rng(seed)
    names = ["blue", "red", "green", "yellow"]
    s = Scenario("random_stress", {}, duration, seed)
    placed = {}
    for n in names:
        s.cubes[n] = Track(None)
        x, y = s.random_free_spot(0.0)
        s.cubes[n] = Track(cube_T(x, y, rng.uniform(-45, 45)))
        placed[n] = (x, y)
    s.distractor.set(0.0, make_T(rot_z(1.2), (0.0, -0.17, 0.012)))
    s.tape.set(0.0, make_T(rot_z(0.7), (0.17, -0.16, 0.0008)))
    on_top = {}  # top cube -> bottom cube
    t = 0.5
    while t < duration - 2.5:
        r = rng.random()
        free = [n for n in names if n not in on_top.values()]
        if on_top and r < 0.35:
            top = next(iter(on_top))
            t = s.knock_off(top, t, push_yaw=rng.uniform(-np.pi, np.pi))
            del on_top[top]
        elif r < 0.12:
            n = free[rng.integers(len(free))]
            if n in on_top:
                del on_top[n]
            x, y = s.random_free_spot(t, exclude=(n,))
            s.teleport(n, t, cube_T(x, y, rng.uniform(-45, 45)))
        elif r < 0.30 and not on_top:
            top, bottom = rng.choice(free, 2, replace=False)
            B = s.cubes[bottom].pose(t)
            t = s.pick_place(top, t, make_T(B[:3, :3], B[:3, 3] + (0, 0, CUBE_SIZE)), peak_speed=rng.uniform(0.5, 1.5), lift=0.08)
            on_top[top] = bottom
        else:
            n = free[rng.integers(len(free))]
            if n in on_top:
                del on_top[n]
            x, y = s.random_free_spot(t, exclude=(n,))
            t = s.pick_place(n, t, cube_T(x, y, rng.uniform(-45, 45)), peak_speed=rng.uniform(0.3, 2.5),
                             lift=rng.uniform(0.03, 0.10))
        t += rng.uniform(0.3, 1.0)
    s.duration = t + 1.0
    return s
