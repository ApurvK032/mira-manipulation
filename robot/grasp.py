"""Top-down grasp planning for 5 cm cubes seen by the live perception (scripts.web_view)."""

import json
import time
import urllib.request

import mujoco
import numpy as np

from .arm import MODEL
from .camera_frame import CameraToBase
from .ik import IK, gripper_down_rotation

STATS = "http://127.0.0.1:8080/stats.json"
HALF = 0.025  # half the cube edge, m
# straight-down grasp and carry are reachable 18-38 cm from the base at any angle (measured with IK)
REACH = (0.18, 0.36)
TABLE_Z = (-0.01, 0.055)  # centre height band (arm frame) of a cube resting on the table; stacked is about 0.08
CLEARANCE = 0.008  # m the open gripper keeps from any other cube along its path to the grasp


class GripperShape:
    """The open gripper's real shape (Trossen's meshes of the fingers, carriages and wrist) along a joint
    path, for clearance from other cubes. The meshes, not the collision boxes: the fingertips reach 4.5 mm
    past ee_site, and the lowest 30 mm of each finger is 20 mm wide, 35-46 mm out from the centre line."""

    BODIES = ("link_5", "link_6", "carriage_left", "carriage_right")

    def __init__(self, path=MODEL, opening=0.035, stride=3):
        self.m = mujoco.MjModel.from_xml_path(str(path))
        self.d = mujoco.MjData(self.m)
        self.qadr = [self.m.joint(f"joint_{i}").qposadr[0] for i in range(6)]
        self.fingers = [self.m.joint(j).qposadr[0] for j in ("left_carriage_joint", "right_carriage_joint")]
        self.opening = opening
        bodies = {self.m.body(b).id for b in self.BODIES}
        self.geoms = []  # (geom id, vertices in the geom frame)
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] in bodies and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                k = self.m.geom_dataid[g]
                a, n = self.m.mesh_vertadr[k], self.m.mesh_vertnum[k]
                self.geoms.append((g, self.m.mesh_vert[a:a + n:stride].astype(float)))

    def points(self, q):
        self.d.qpos[:] = 0
        self.d.qpos[self.qadr] = q
        self.d.qpos[self.fingers] = self.opening
        mujoco.mj_kinematics(self.m, self.d)
        return np.vstack([V @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g] for g, V in self.geoms])

    def clearance(self, path, boxes, n=12):
        """Smallest distance (m) from the gripper to any cube box (center, R) along joint-space straight
        lines through the configurations in `path` (how SafeArm.move_joints moves)."""
        best = np.inf
        for qa, qb in zip(path[:-1], path[1:]):
            for s in np.linspace(0.0, 1.0, n):
                P = self.points((1 - s) * qa + s * qb)
                for c, R in boxes:
                    out = np.maximum(np.abs((P - c) @ R) - HALF, 0.0)
                    best = min(best, float(np.sqrt((out ** 2).sum(axis=1)).min()))
        return best


def is_still(P, move=(0.002, 0.002, 0.004), bump=0.006):
    """Did a cube rest still over these position samples (about 10 Hz)? Medians of the first, middle and
    last third agree within `move` (per axis: x, y, z) and no sample is `bump` from the median. A partly
    hidden cube's pose jitters about 1 mm per frame, so a raw peak-to-peak test almost never passes; this
    one does 99% of the time and still catches a 2 mm/s slide (logged on the real camera, 2026-10-16).
    Height gets more room: the gripper hovering over a cube shifts its estimated height by about 3 mm."""
    P = np.asarray(P)
    if len(P) < 9:
        return False
    k = len(P) // 3
    meds = [np.median(P[i * k:(i + 1) * k], axis=0) for i in range(3)]
    spread = np.max([np.abs(a - b) for a in meds for b in meds], axis=0)
    return bool(np.all(spread < np.asarray(move)[:P.shape[1]]) and np.abs(P - np.median(P, axis=0)).max() < bump)


def read_stats(timeout=1.0) -> dict:
    """One snapshot of the live viewer: cubes (camera frame), status, operator command, frame time."""
    return json.load(urllib.request.urlopen(STATS, timeout=timeout))


def cubes_in_base(stats: dict, cal: CameraToBase) -> dict:
    """{name: (state, center, R, unhidden fraction)} in the arm frame; center/R None if not posed."""
    out = {}
    for name, c in stats.get("cubes", {}).items():
        if not c.get("pos"):
            out[name] = (c.get("state", "not seen"), None, None, 0.0)
        else:
            out[name] = (c["state"], cal.point(np.array(c["pos"]) / 1000.0), cal.rotation(np.array(c["rot"])),
                         c.get("check", [0.0])[0])
    return out


class RestTracker:
    """Non-blocking 'is this cube resting?': feed it every snapshot; a cube is resting once it has been
    TRACKING, at least min_unhidden in view, and still (is_still) for still_s. Its view must not shrink
    over that time (something such as a hand coming over it)."""

    def __init__(self, still_s=1.0, min_unhidden=0.5):
        self.still_s, self.min_unhidden = still_s, min_unhidden
        self.hist = {}

    def update(self, now, cubes: dict):
        for name, (state, p, R, vis) in cubes.items():
            h = self.hist.setdefault(name, [])
            if state != "TRACKING" or p is None or vis < self.min_unhidden:
                h.clear()
                continue
            h.append((now, p, R, vis))
            while h and now - h[0][0] > self.still_s + 0.3:
                h.pop(0)

    def resting(self, name, now, still_s=None):
        """(center, R, unhidden) if the cube is resting (for still_s, default the tracker's), else None."""
        still_s = self.still_s if still_s is None else still_s
        h = [x for x in self.hist.get(name, []) if now - x[0] <= still_s + 0.3]
        if not h or now - h[0][0] < still_s or now - h[-1][0] > 0.5:
            return None
        P = np.array([x[1] for x in h])
        vis = np.array([x[3] for x in h])
        if not is_still(P) or vis.min() < np.median(vis) - 0.1:
            return None
        return np.median(P, axis=0), h[-1][2], float(np.median(vis))


def read_cube(cal: CameraToBase, name: str):
    """Latest (state, center, R, unhidden fraction) of a cube in the arm frame; center/R None if not posed."""
    c = json.load(urllib.request.urlopen(STATS, timeout=1))["cubes"].get(name, {})
    if not c.get("pos"):
        return c.get("state", "not seen"), None, None, 0.0
    return (c["state"], cal.point(np.array(c["pos"]) / 1000.0), cal.rotation(np.array(c["rot"])),
            c.get("check", [0.0])[0])


def wait_for_cube(cal: CameraToBase, name: str, timeout=180.0, reach=REACH):
    """Return (center, R) in the arm frame once the cube has rested still, unhidden, on the table for 2 s."""
    t0, last_msg, hist = time.time(), 0.0, []
    while time.time() - t0 < timeout:
        state, p, R, vis = read_cube(cal, name)
        now = time.time()
        if state == "TRACKING" and p is not None and vis >= 0.9:
            hist.append((now, p, R))
            hist = [h for h in hist if now - h[0] < 2.0]
            P = np.array([h[1] for h in hist])
            on_table = -0.01 < p[2] < 0.08
            reachable = reach[0] < np.hypot(p[0], p[1]) < reach[1] and p[0] > 0.1
            if now - hist[0][0] > 1.8 and is_still(P) and on_table and reachable:
                return np.median(P, axis=0), hist[-1][2]
            why = "settling"
            if not on_table:
                why = f"not on the table (z {p[2] * 1000:.0f} mm)"
            elif not reachable:
                why = (f"out of reach, {np.hypot(p[0], p[1]) * 100:.0f} cm from the arm base: place it "
                       f"{reach[0] * 100:.0f}-{reach[1] * 100:.0f} cm from the base, in front")
        else:
            hist, why = [], f"{name} cube {state} or partly hidden"
        if now - last_msg > 5:
            print(f"  waiting for the {name} cube: {why}", flush=True)
            last_msg = now
        time.sleep(0.1)
    raise TimeoutError(f"no resting {name} cube within {timeout:.0f} s")


def face_yaw(R):
    """Direction (rad) of a horizontal cube axis: fingers along it close on two opposite faces."""
    a = min((R[:, i] for i in range(3)), key=lambda v: abs(v[2]))
    return float(np.arctan2(a[1], a[0]))


def top_face_tilt_deg(R):
    return float(np.degrees(np.arccos(np.clip(max(abs(R[2, i]) for i in range(3)), 0.0, 1.0))))


def plan_top_down(center, R, heights: dict, ik: IK | None = None, obstacles=(), start=None,
                  shape: GripperShape | None = None, clearance=CLEARANCE, seed=None):
    """Joint targets with the gripper pointing straight down over the cube, fingertips at
    center z + height for each named height (m). The finger direction is the symmetric choice (every
    90 deg) that needs the least wrist roll (from `seed` when given) among those whose path (from `start`, then through the heights
    in order) keeps the open gripper `clearance` from every obstacle cube, given as (name, center, R).
    seed: joint angles to start the IK from (e.g. the current pose or the last plan) for a fast re-plan;
    the full multi-seed search runs only if that does not converge.
    Returns (qs, yaw) or raises ValueError if unreachable or no finger direction clears the obstacles."""
    ik = ik or IK()
    if obstacles:
        shape = shape or GripperShape()
    yaw0 = face_yaw(R)
    az = np.arctan2(center[1], center[0])
    seeds = [np.array([az, a, b, w, 0, 0]) for a in (0.4, 0.9, 1.4) for b in (0.4, 1.0, 1.6) for w in (0.5, 1.2)]
    best, blocked = None, None
    for k in range(4):
        R_down = gripper_down_rotation(yaw0 + k * np.pi / 2)
        qs, worst, prev = {}, 0.0, None
        for name, h in heights.items():
            p = np.asarray(center, dtype=float) + [0, 0, h]
            first = [prev] if prev is not None else ([np.asarray(seed, dtype=float)] if seed is not None else seeds)
            q, ep, er = min((ik.solve(p, R_down, sd) for sd in first), key=lambda r: r[1] + r[2] / 100)
            if prev is None and seed is not None and ep * 1000 + er >= 1.0:
                q, ep, er = min((ik.solve(p, R_down, sd) for sd in seeds), key=lambda r: r[1] + r[2] / 100)
            qs[name], prev, worst = q, q, max(worst, ep * 1000 + er)
        if worst >= 1.0:
            continue
        if obstacles:
            path = ([np.asarray(start, dtype=float)] if start is not None else []) + list(qs.values())
            gap, near = min((shape.clearance(path, [(c, Ro)]), n) for n, c, Ro in obstacles)
            if gap < clearance:
                if blocked is None or gap > blocked[0]:
                    blocked = (gap, near)
                continue
        # least wrist roll: from zero for a fresh plan, from the current wrist angle for a re-plan (else two
        # nearly equal choices can alternate between re-plans and the gripper turns 90 deg back and forth)
        # (every solution here is within 1 mm / 1 deg, so the residual only breaks ties; scaled up, it used to
        # outweigh the roll and the wrist could turn 200+ deg between the pick and the place)
        roll = prev[5] - (np.asarray(seed, dtype=float)[5] if seed is not None else 0.0)
        cost = abs(roll) + 0.01 * worst
        if best is None or cost < best[0]:
            best = (cost, qs, yaw0 + k * np.pi / 2)
    if best is None and blocked is not None:
        raise ValueError(f"the open fingers would pass {max(blocked[0], 0) * 1000:.0f} mm from the {blocked[1]} cube "
                         f"whichever way they turn (need {clearance * 1000:.0f} mm): give it a little more room")
    if best is None:
        raise ValueError("cannot reach the cube with the gripper pointing down: move it closer to the centre")
    return best[1], best[2]


def wait_for_cubes(cal: CameraToBase, names, timeout=180.0, min_sep=0.045, reach=REACH,
                   still_s=1.8, notify=None, min_unhidden=0.5):
    """Wait until every named cube rests still (2 s) on the table, within reach and verified by the
    tracker (TRACKING, at least min_unhidden of it in view: the other cube may hide part of it). Closeness
    is left to the grasp planner's finger-clearance check; centres closer than min_sep mean the tracker
    has confused the two. Returns {name: (center, R)} in the arm frame."""
    t0, last_msg, hist = time.time(), 0.0, {n: [] for n in names}
    while time.time() - t0 < timeout:
        now, why, out = time.time(), None, {}
        seen = {n: read_cube(cal, n) for n in names}
        for n in names:
            state, p, R, vis = seen[n]
            if state != "TRACKING" or p is None:
                hist[n], why = [], why or f"{n} cube {state}"
                continue
            if vis < min_unhidden:
                hist[n], why = [], why or f"{n} cube mostly hidden from the camera ({vis:.0%} in view)"
                continue
            hist[n] = [h for h in hist[n] if now - h[0] < still_s + 0.2] + [(now, p, R)]
            P = np.array([h[1] for h in hist[n]])
            under = [m for m in names if m != n and seen[m][1] is not None
                     and np.linalg.norm(seen[m][1][:2] - p[:2]) < 0.035 and seen[m][1][2] < p[2]]
            if under and p[2] >= TABLE_Z[1]:
                why = why or f"{n} cube is stacked on {under[0]}: unstack them"
            elif not TABLE_Z[0] < p[2] < TABLE_Z[1]:
                why = why or f"{n} cube not on the table (z {p[2] * 1000:.0f} mm)"
            elif not (reach[0] < np.hypot(p[0], p[1]) < reach[1] and p[0] > 0.1):
                why = why or (f"{n} cube out of reach, {np.hypot(p[0], p[1]) * 100:.0f} cm from the arm base "
                              f"(place cubes {reach[0] * 100:.0f}-{reach[1] * 100:.0f} cm from it, in front)")
            elif now - hist[n][0][0] < still_s or not is_still(P):
                why = why or f"{n} cube settling"
            else:
                out[n] = (np.median(P, axis=0), hist[n][-1][2])
        if len(out) == len(names):
            c = [out[n][0] for n in names]
            if all(np.linalg.norm(a[:2] - b[:2]) >= min_sep for i, a in enumerate(c) for b in c[i + 1:]):
                return out
            why = f"cube centres closer than {min_sep * 1000:.0f} mm: the tracker has the cubes mixed up"
        if now - last_msg > 5:
            print(f"  waiting: {why}", flush=True)
            if notify:
                notify(why)
            last_msg = now
        time.sleep(0.1)
    raise TimeoutError("cubes not ready")
