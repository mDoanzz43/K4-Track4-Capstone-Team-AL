"""PyBullet simulation of a KUKA LBR iiwa (7-DoF) holding a pen.

The simulation runs headless (DIRECT mode, CPU TinyRenderer). Each control
step solves inverse kinematics for the commanded pen-tip pose, sends the joint
targets to position-controlled motors and steps the physics. The pen tip pose
is then read back from forward kinematics: wherever the *actual* tip touches
the paper it leaves ink, so what you see is what the arm really did, tracking
errors included.

Ink is composited onto the rendered frame by projecting the inked 3-D points
with the same camera matrices and masking with the segmentation buffer
(only pixels that belong to the paper receive ink, so the arm correctly
occludes it).
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import pybullet as p
import pybullet_data

from .trajectory import Trajectory, TrajectoryConfig

EE_LINK = 6
PEN_LENGTH = 0.15  # distance from the last link frame to the pen tip along its z axis
INK_COLOR = np.array([122, 42, 28], np.float32)  # BGR, dark navy ink
PLAN_COLOR = np.array([235, 200, 160], np.float32)  # BGR, light blue for the planned path

CAMERAS = {
    # name: (eye, target, up, fov) -- world frame, paper centre at ~(0.62, 0, 0.62)
    "perspective": ([0.05, -0.95, 1.30], [0.56, 0.0, 0.70], [0, 0, 1], 48),
    "top": ([0.95, 0.0, 1.65], [0.62, 0.0, 0.62], [1, 0, 0], 26),
    "side": ([0.55, -1.35, 1.05], [0.42, 0.0, 0.82], [0, 0, 1], 50),
    "front": ([0.95, -0.85, 1.25], [0.55, 0.0, 0.70], [0, 0, 1], 50),
}


@dataclass
class FrameInfo:
    step: int
    total: int
    t: float
    pen_down: bool
    tip: tuple[float, float, float]
    new_ink: list  # list of polylines (world XY) inked since the last frame
    stroke_id: int


class DrawingSim:
    def __init__(
        self,
        cfg: TrajectoryConfig | None = None,
        width: int = 480,
        height: int = 360,
        camera: str = "perspective",
        shadows: bool = False,
    ):
        self.cfg = cfg or TrajectoryConfig()
        self.width, self.height = width, height
        self.shadows = shadows
        self.cid = p.connect(p.DIRECT)
        self._lock = threading.Lock()
        self._build_scene()
        self.set_camera(camera)
        self.ink: list[list[np.ndarray]] = []  # polylines of 3-D points
        self.plan: list[np.ndarray] = []

    # ------------------------------------------------------------------ scene
    def _build_scene(self):
        cid = self.cid
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.setTimeStep(self.cfg.dt, physicsClientId=cid)
        self.plane = p.loadURDF("plane.urdf", physicsClientId=cid)

        pz = self.cfg.paper_z
        cx, cy = self.cfg.paper_center
        # Table: the robot is bolted onto it, the paper lies in front of it.
        top_h = 0.04
        tx0, tx1, ty = -0.30, cx + 0.40, 0.55
        half = [(tx1 - tx0) / 2, ty, top_h / 2]
        tvis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.55, 0.38, 0.24, 1], physicsClientId=cid)
        tcol = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)
        self.table = p.createMultiBody(0, tcol, tvis, [(tx0 + tx1) / 2, 0, pz - 0.001 - top_h / 2], physicsClientId=cid)
        leg_h = pz - top_h
        lvis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, leg_h / 2], rgbaColor=[0.4, 0.27, 0.17, 1], physicsClientId=cid)
        for lx in (tx0 + 0.05, tx1 - 0.05):
            for ly in (-ty + 0.05, ty - 0.05):
                p.createMultiBody(0, -1, lvis, [lx, ly, leg_h / 2], physicsClientId=cid)

        # Paper: thin white sheet.
        s = self.cfg.char_size
        self.paper_half = [s / 2 + 0.04, s / 2 + 0.06, 0.0005]
        pvis = p.createVisualShape(p.GEOM_BOX, halfExtents=self.paper_half, rgbaColor=[0.98, 0.98, 0.96, 1], physicsClientId=cid)
        self.paper = p.createMultiBody(0, -1, pvis, [cx, cy, pz - 0.0005], physicsClientId=cid)

        # Robot mounted on the table top; a second, invisible copy in its own
        # physics client is used to refine IK solutions with forward kinematics.
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, pz], useFixedBase=True, physicsClientId=cid)
        self.ik_cid = p.connect(p.DIRECT)
        self.ik_robot = p.loadURDF(
            pybullet_data.getDataPath() + "/kuka_iiwa/model.urdf", [0, 0, pz], useFixedBase=True, physicsClientId=self.ik_cid
        )
        self.n_joints = p.getNumJoints(self.robot, physicsClientId=cid)
        self.ll, self.ul = [], []
        for j in range(self.n_joints):
            info = p.getJointInfo(self.robot, j, physicsClientId=cid)
            self.ll.append(info[8])
            self.ul.append(info[9])
        self.jr = [u - l for l, u in zip(self.ll, self.ul)]
        # Comfortable "writing" posture used as the IK null-space attractor.
        self.rest = [0.0, 0.45, 0.0, -1.65, 0.0, 1.0, 0.0]
        self.down_orn = p.getQuaternionFromEuler([0, math.pi, 0])

        # Pen: visual only, its pose follows the end effector.
        pen_vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.007, length=PEN_LENGTH - 0.02, rgbaColor=[0.95, 0.75, 0.1, 1], physicsClientId=cid
        )
        tip_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.006, rgbaColor=[0.1, 0.1, 0.35, 1], physicsClientId=cid)
        self.pen = p.createMultiBody(0, -1, pen_vis, [0, 0, 0], physicsClientId=cid)
        self.pen_tip = p.createMultiBody(0, -1, tip_vis, [0, 0, 0], physicsClientId=cid)

        home = [cx - 0.1, cy, pz + self.cfg.lift + 0.06]
        self.q_prev = np.array(self.rest, float)
        q = self.ik(home)
        for j in range(self.n_joints):
            p.resetJointState(self.robot, j, q[j], physicsClientId=cid)
        self.q_prev = np.array(q)
        self._update_pen()

    # ------------------------------------------------------------ kinematics
    def _set_ik_joints(self, q):
        for j in range(self.n_joints):
            p.resetJointState(self.ik_robot, j, q[j], physicsClientId=self.ik_cid)

    def ik(self, tip_pos, refine: int = 3, tol: float = 2e-5) -> list[float]:
        """IK for the pen tip pointing straight down at `tip_pos`.

        PyBullet's damped-least-squares IK with null-space attraction leaves a
        residual of a few millimetres, so the solution is refined: evaluate FK
        on the shadow robot, shift the target by the error and solve again,
        always seeding from the previous solution (keeps the motion continuous).
        """
        rot = np.array(p.getMatrixFromQuaternion(self.down_orn)).reshape(3, 3)
        offset = rot @ np.array([0, 0, PEN_LENGTH])
        tip_pos = np.asarray(tip_pos, float)
        goal = tip_pos - offset
        q = list(self.q_prev)
        for _ in range(refine + 1):
            self._set_ik_joints(q)
            q = list(
                p.calculateInverseKinematics(
                    self.ik_robot,
                    EE_LINK,
                    goal.tolist(),
                    self.down_orn,
                    lowerLimits=self.ll,
                    upperLimits=self.ul,
                    jointRanges=self.jr,
                    restPoses=self.rest,
                    maxNumIterations=50,
                    residualThreshold=1e-6,
                    physicsClientId=self.ik_cid,
                )[: self.n_joints]
            )
            self._set_ik_joints(q)
            ls = p.getLinkState(self.ik_robot, EE_LINK, computeForwardKinematics=True, physicsClientId=self.ik_cid)
            r = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
            err = tip_pos - (np.array(ls[4]) + r @ np.array([0, 0, PEN_LENGTH]))
            if np.linalg.norm(err) < tol:
                break
            goal = goal + err
        return q

    def tip_pose(self):
        ls = p.getLinkState(self.robot, EE_LINK, computeForwardKinematics=True, physicsClientId=self.cid)
        pos, orn = np.array(ls[4]), ls[5]
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        return pos + rot @ np.array([0, 0, PEN_LENGTH]), pos, orn, rot

    def _update_pen(self):
        tip, ee, orn, rot = self.tip_pose()
        mid = ee + rot @ np.array([0, 0, PEN_LENGTH / 2 + 0.005])
        p.resetBasePositionAndOrientation(self.pen, mid.tolist(), orn, physicsClientId=self.cid)
        p.resetBasePositionAndOrientation(self.pen_tip, tip.tolist(), orn, physicsClientId=self.cid)
        return tip

    # ---------------------------------------------------------------- camera
    def set_camera(self, name: str):
        """Select a camera preset (call with `_lock` held while a run is active)."""
        eye, target, up, fov = CAMERAS.get(name, CAMERAS["perspective"])
        self.camera = name if name in CAMERAS else "perspective"
        self.view = p.computeViewMatrix(eye, target, up, physicsClientId=self.cid)
        self.proj = p.computeProjectionMatrixFOV(fov, self.width / self.height, 0.05, 6.0, physicsClientId=self.cid)
        V = np.array(self.view).reshape(4, 4).T
        P = np.array(self.proj).reshape(4, 4).T
        self._pv = P @ V

    def project(self, pts3: np.ndarray) -> np.ndarray:
        h = np.column_stack([pts3, np.ones(len(pts3))]) @ self._pv.T
        ndc = h[:, :3] / h[:, 3:4]
        x = (ndc[:, 0] + 1) * 0.5 * self.width
        y = (1 - ndc[:, 1]) * 0.5 * self.height
        return np.stack([x, y], 1)

    def _ink_px_width(self) -> int:
        c = np.array([[*self.cfg.paper_center, self.cfg.paper_z]])
        a = self.project(c)[0]
        b = self.project(c + [[0.0, 0.002, 0.0]])[0]
        return int(np.clip(round(np.linalg.norm(b - a)), 2, 6))

    def render(self) -> np.ndarray:
        """Render the scene (BGR uint8) with ink composited onto the paper."""
        w, h = self.width, self.height
        _, _, rgba, _, seg = p.getCameraImage(
            w,
            h,
            self.view,
            self.proj,
            renderer=p.ER_TINY_RENDERER,
            shadow=int(self.shadows),
            lightDirection=[0.4, -0.6, 1.0],
            physicsClientId=self.cid,
        )
        frame = cv2.cvtColor(np.reshape(np.asarray(rgba, np.uint8), (h, w, 4))[:, :, :3], cv2.COLOR_RGB2BGR)
        seg = np.reshape(np.asarray(seg), (h, w))
        on_paper = (seg == self.paper).astype(np.float32)

        scale = 4  # supersample the ink layer for smooth lines
        width_px = self._ink_px_width()
        layers = []
        for polylines, color, thick in ((self.plan, PLAN_COLOR, 1), (self.ink, INK_COLOR, width_px)):
            if not polylines:
                continue
            layer = np.zeros((h * scale, w * scale), np.uint8)
            for pl in polylines:
                pl = np.asarray(pl)
                if len(pl) == 0:
                    continue
                px = np.round(self.project(pl) * scale).astype(np.int32)
                if len(px) == 1:
                    cv2.circle(layer, tuple(px[0]), thick * scale // 2, 255, -1)
                else:
                    cv2.polylines(layer, [px], False, 255, thick * scale, cv2.LINE_8)
            alpha = cv2.resize(layer, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            layers.append((alpha * on_paper, color))
        if layers:
            f = frame.astype(np.float32)
            for alpha, color in layers:
                a = alpha[..., None]
                f = f * (1 - a) + color * a
            frame = f.astype(np.uint8)
        return frame

    # -------------------------------------------------------------- run loop
    def plan_from(self, traj: Trajectory) -> list[np.ndarray]:
        """Planned drawing segments projected onto the paper (for display)."""
        plan = []
        for k in np.unique(traj.stroke_id[traj.stroke_id >= 0]):
            seg = traj.points[traj.stroke_id == k]
            pts = np.vstack([seg[::4], seg[-1:]])
            pts[:, 2] = self.cfg.paper_z
            plan.append(pts)
        return plan

    def run(
        self,
        traj: Trajectory,
        on_frame: Callable[[np.ndarray, FrameInfo], None],
        speed: float = 1.0,
        fps: float = 20.0,
        realtime: bool = True,
        stop_event: threading.Event | None = None,
        show_plan: bool = True,
    ) -> dict:
        """Execute a trajectory, calling `on_frame(frame_bgr, info)` for each
        rendered frame (about `fps` per wall-clock second in real-time mode).
        Returns tracking statistics."""
        cid = self.cid
        cfg = self.cfg
        with self._lock:
            self.ink = []
            self.plan = self.plan_from(traj) if show_plan else []
        # Offline mode renders every `steps_per_frame` steps. Real-time mode
        # renders whenever the wall clock reaches the next frame slot; if the
        # CPU cannot keep up, frames are dropped (not the motion) so playback
        # stays in sync with simulated time.
        steps_per_frame = max(1, int(round(speed / (fps * cfg.dt))))
        next_frame = 0.0
        ink_thresh = cfg.paper_z + 0.002
        errors = []
        current = None
        new_ink: list = []
        prev_tip = None
        wall0 = time.perf_counter()
        n = len(traj)
        stopped = False

        joints = list(range(self.n_joints))
        for i in range(n):
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            target = traj.points[i]
            q = np.array(self.ik(target))
            qd = (q - self.q_prev) / cfg.dt
            self.q_prev = q
            p.setJointMotorControlArray(
                self.robot,
                joints,
                p.POSITION_CONTROL,
                targetPositions=q.tolist(),
                targetVelocities=qd.tolist(),
                forces=[320.0] * self.n_joints,
                positionGains=[0.35] * self.n_joints,
                velocityGains=[1.0] * self.n_joints,
                physicsClientId=cid,
            )
            p.stepSimulation(physicsClientId=cid)

            tip, _, _, _ = self.tip_pose()
            if traj.pen_down[i]:
                errors.append(np.linalg.norm(tip[:2] - target[:2]))
            touching = tip[2] <= ink_thresh
            if touching:
                ink_pt = np.array([tip[0], tip[1], cfg.paper_z])
                if current is None:
                    current = [ink_pt]
                    self.ink.append(current)
                    new_ink.append([ink_pt[:2].tolist()])
                elif np.linalg.norm(ink_pt - current[-1]) > 0.0004:
                    current.append(ink_pt)
                    if not new_ink:
                        new_ink.append([])
                    new_ink[-1].append(ink_pt[:2].tolist())
            else:
                current = None
            prev_tip = tip

            last = i == n - 1
            if realtime:
                due = (i + 1) * cfg.dt / speed
                render_now = due >= next_frame or last
                if render_now:
                    ahead = due - (time.perf_counter() - wall0)
                    if ahead > 0:
                        time.sleep(ahead)
            else:
                render_now = i % steps_per_frame == 0 or last
            if render_now:
                with self._lock:
                    self._update_pen()
                    frame = self.render()
                info = FrameInfo(
                    step=i + 1,
                    total=n,
                    t=(i + 1) * cfg.dt,
                    pen_down=bool(touching),
                    tip=tuple(float(v) for v in prev_tip),
                    new_ink=new_ink,
                    stroke_id=int(traj.stroke_id[i]),
                )
                on_frame(frame, info)
                new_ink = [[current[-1][:2].tolist()]] if current is not None else []
                if realtime:
                    next_frame = max(due, time.perf_counter() - wall0) + 1.0 / fps

        err = np.array(errors) if errors else np.zeros(1)
        return {
            "stopped": stopped,
            "sim_time": round(n * cfg.dt, 2),
            "wall_time": round(time.perf_counter() - wall0, 2),
            "tracking_rms_mm": round(float(np.sqrt((err**2).mean())) * 1000, 3),
            "tracking_max_mm": round(float(err.max()) * 1000, 3),
            "ink_polylines": len(self.ink),
            "ink_points": int(sum(len(pl) for pl in self.ink)),
        }

    def close(self):
        for c in (self.cid, self.ik_cid):
            try:
                p.disconnect(c)
            except Exception:
                pass
