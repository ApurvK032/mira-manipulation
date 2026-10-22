"""Check that rl/deploy.py gives the policies exactly what they were trained on.

Runs the chain in the training environment, as rl/eval_chained.py does, but every observation is rebuilt the way the
deployment runner builds it on the arm:
  - the fingertip and gripper angle from FK of the joint angles (robot.ik, not the environment's own data);
  - the cube readings the environment would show (its delayed, noisy history) sent through the live tracker's
    stats format: into the camera frame with the inverse of assets/calibration.yaml, then back with
    robot.grasp.cubes_in_base, and the cube angle taken from the rotation matrix;
  - rl.deploy.grab_obs / stack_obs, normalized with the checkpoints' statistics by rl.deploy.load_chain.
The policies act on those observations. Each step compares them with the environment's own (_obs and
rl.eval_chained.grab_obs) and compares rl.deploy.Commander's fingertip target and gripper angle with the
environment's.

One difference is expected and is not the deployment path's: the environment reads the fingertip from MuJoCo's
site_xpos, which mj_step computes before it integrates, so it is one physics step (2 ms) older than the joint angles.
FK of the joints (what the runner does) gives the current fingertip. So the comparison is made twice: with the
runner's FK fingertip (the real path), and with the environment's own fingertip fed to the same code, which must
match to float32 rounding.

Usage:  python -m rl.check_deploy_obs <grab_model> <place_model> [--episodes 50] [--seed 4242] [--env-kwargs '{...}']
"""

import argparse
import json

import numpy as np

from rl.deploy import CHAIN_KWARGS, Commander, Meas, Reading, cube_tilt, cube_yaw, grab_obs, load_chain, stack_obs
from rl.dry_world import rot_z
from rl.envs.moving_cube import HALF
from rl.envs.stack_cubes import CARRY_Z, StackCubes
from rl.eval_chained import grab_obs as env_grab_obs
from robot.camera_frame import CameraToBase
from robot.grasp import cubes_in_base
from robot.ik import IK


def via_tracker(env, idx, cal):
    """The environment's cube readings at history index idx, through the tracker's stats format and back."""
    Rbc, tbc = cal.T[:3, :3], cal.T[:3, 3]
    stats = {"cubes": {}, "camera_serial": cal.serial}
    for n in ("red", "green"):
        p, yaw = env.hist[n][idx]
        stats["cubes"][n] = {"state": "TRACKING", "pos": (1000 * Rbc.T @ (p - tbc) / cal.k).tolist(),
                             "rot": (Rbc.T @ rot_z(yaw)).tolist(), "check": [1.0, 1.0, 1.0, 0.0]}
    return {n: Reading(p, cube_yaw(R), cube_tilt(R)) for n, (_, p, R, _) in cubes_in_base(stats, cal).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grab_model")
    ap.add_argument("place_model")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--env-kwargs", default=json.dumps(dict(CHAIN_KWARGS, lag_tau=[0.16, 0.16], delay_steps=[5, 5])))
    args = ap.parse_args()
    grab, place = load_chain(args.grab_model, args.place_model)
    env, ik, cal = StackCubes(**json.loads(args.env_kwargs)), IK(), CameraToBase()
    cmd = Commander(ik, np.zeros(6))
    worst = {k: 0.0 for k in ("grab obs, same fingertip", "place obs, same fingertip", "tip target (m)",
                              "gripper angle (rad)", "grab obs, FK fingertip", "place obs, FK fingertip",
                              "FK vs env fingertip (m)", "action, FK fingertip")}
    results, steps, gap = [], 0, {"tip": [], "obs": [], "action": []}  # per step, FK fingertip vs the env's
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        phase = "grab"
        for t in range(env.max_steps):
            if phase == "grab":
                rp, _, _ = env._cube("red")
                if env._red_grasped() and rp[2] > HALF + 0.01:
                    phase = "lift"
            if phase == "lift" and (env._tip()[0][2] > CARRY_Z - 0.005 or not env._red_grasped()):
                phase = "stack"
            q, qd, finger = env.d.qpos[env.qadr].copy(), env.d.qvel[env.dadr].copy(), float(env.d.qpos[env.finger])
            tip, R = ik.fk(q)
            env_tip, env_R = env._tip()
            worst["FK vs env fingertip (m)"] = max(worst["FK vs env fingertip (m)"], float(np.abs(tip - env_tip).max()))
            gap["tip"].append(float(np.abs(tip - env_tip).max()))
            m_fk, m_same = Meas(tip, R, q, qd, finger), Meas(env_tip, env_R, q, qd, finger)
            k = env.delay
            cur, prev = via_tracker(env, -1 - k, cal), via_tracker(env, -2 - k, cal)
            if phase == "grab":
                o_env, build, pol, name = env_grab_obs(env, True), grab_obs, grab, "grab"
            elif phase == "stack":
                o_env, build, pol, name = env._obs(), stack_obs, place, "place"
            if phase == "lift":
                a = np.array([0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
            else:
                o_same, o_fk = build(m_same, cur, prev, env.prev_action), build(m_fk, cur, prev, env.prev_action)
                worst[f"{name} obs, same fingertip"] = max(worst[f"{name} obs, same fingertip"], float(np.abs(o_same - o_env).max()))
                worst[f"{name} obs, FK fingertip"] = max(worst[f"{name} obs, FK fingertip"], float(np.abs(o_fk - o_env).max()))
                a = pol(o_same)
                worst["action, FK fingertip"] = max(worst["action, FK fingertip"], float(np.abs(pol(o_fk) - a).max()))
                gap["obs"].append(float(np.abs(o_fk - o_env).max()))
                gap["action"].append(float(np.abs(pol(o_fk) - a).max()))
            # the deployment runner's target update from the same state, against the environment's
            cmd.tip_target, cmd.yaw_target = env.tip_target.copy(), env.yaw_target
            t_dep, yaw_dep = cmd.target(np.clip(np.asarray(a, dtype=float), -1, 1), env_tip, env_R)
            _, _, done, trunc, info = env.step(a)
            worst["tip target (m)"] = max(worst["tip target (m)"], float(np.abs(t_dep - env.tip_target).max()))
            worst["gripper angle (rad)"] = max(worst["gripper angle (rad)"], abs(yaw_dep - env.yaw_target))
            steps += 1
            if done or trunc:
                results.append("success" if info.get("success") else "knocked" if info.get("knocked")
                               else "out" if info.get("out") else "timeout")
                break
    n = len(results)
    print(f"{steps} steps, {n} episodes from seed {args.seed}: success {results.count('success')}/{n} "
          f"(knocked {results.count('knocked')}, out {results.count('out')}, timeout {results.count('timeout')})")
    print("largest differences, deployment path vs training environment:")
    for k, v in worst.items():
        print(f"  {k:26s} {v:.2e}")
    print("FK fingertip vs the env's (one physics step older), over all steps: median / 99th percentile / max")
    for k, v in gap.items():
        print(f"  {k:8s} {np.median(v):.2e} / {np.percentile(v, 99):.2e} / {np.max(v):.2e}")


if __name__ == "__main__":
    main()
