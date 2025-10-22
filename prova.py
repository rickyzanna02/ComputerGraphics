#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============== JOINTS (ordine del tuo file, 24) ==============
JOINTS = [
    'Hips','Spine','Spine1','Spine2','Neck','Head',
    'LeftShoulder','LeftArm','LeftForeArm','LeftForeArmRoll','LeftHand',
    'RightShoulder','RightArm','RightForeArm','RightForeArmRoll','RightHand',
    'LeftUpLeg','LeftLeg','LeftFoot','LeftToeBase',
    'RightUpLeg','RightLeg','RightFoot','RightToeBase'
]

# ============== GERARCHIA (parent -> children) ==============
PARENTS = {
    'Hips':        ['Spine','LeftUpLeg','RightUpLeg'],
    'Spine':       ['Spine1'],
    'Spine1':      ['Spine2'],
    'Spine2':      ['Neck','LeftShoulder','RightShoulder'],
    'Neck':        ['Head'],
    'Head':        [],
    'LeftShoulder':['LeftArm'],
    'LeftArm':     ['LeftForeArm'],
    'LeftForeArm': ['LeftForeArmRoll'],
    'LeftForeArmRoll':['LeftHand'],
    'LeftHand':    [],
    'RightShoulder':['RightArm'],
    'RightArm':    ['RightForeArm'],
    'RightForeArm':['RightForeArmRoll'],
    'RightForeArmRoll':['RightHand'],
    'RightHand':   [],
    'LeftUpLeg':   ['LeftLeg'],
    'LeftLeg':     ['LeftFoot'],
    'LeftFoot':    ['LeftToeBase'],
    'LeftToeBase': [],
    'RightUpLeg':  ['RightLeg'],
    'RightLeg':    ['RightFoot'],
    'RightFoot':   ['RightToeBase'],
    'RightToeBase':[]
}

# ============== UTILS ==============
def build_edges():
    name2idx = {n:i for i,n in enumerate(JOINTS)}
    edges = []
    for p, childs in PARENTS.items():
        p_idx = name2idx[p]
        for c in childs:
            edges.append((p_idx, name2idx[c]))
    return edges

def load_all_frames_positions(json_path: Path, scale: float):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for k in data.keys():
        if k.startswith("frame_"):
            suf = k.split("frame_")[-1]
            num = int(suf.lstrip("0") or "0")
            items.append((num, k))
    if not items:
        raise ValueError("Nessuna chiave 'frame_*' trovata nel JSON posizioni.")
    items.sort(key=lambda t: t[0])
    frames = []
    for _, k in items:
        pts = np.array(data[k], dtype=float)
        if pts.shape != (24,3):
            raise ValueError(f"{k}: atteso (24,3), trovato {pts.shape}")
        frames.append(pts * scale)
    return np.stack(frames, axis=0)  # (F,24,3)

def load_all_frames_rotations(json_path: Path):
    """Ritorna array (F,24,4) quaternioni (x,y,z,w), normalizzati."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for k in data.keys():
        if k.startswith("frame_"):
            suf = k.split("frame_")[-1]
            num = int(suf.lstrip("0") or "0")
            items.append((num, k))
    if not items:
        raise ValueError("Nessuna chiave 'frame_*' trovata nel JSON rotazioni.")
    items.sort(key=lambda t: t[0])
    frames_q = []
    for _, k in items:
        qs = np.array(data[k], dtype=float)
        if qs.shape != (24,4):
            raise ValueError(f"{k}: atteso (24,4), trovato {qs.shape}")
        # normalizza
        n = np.linalg.norm(qs, axis=1, keepdims=True)
        n[n == 0] = 1.0
        qs = qs / n
        frames_q.append(qs)
    return np.stack(frames_q, axis=0)  # (F,24,4)

def quat_to_R(q):
    """q = (x,y,z,w). Ritorna matrice 3x3."""
    x,y,z,w = q
    # Assicura normalizzazione
    n = x*x + y*y + z*z + w*w
    if n <= 0.0:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s
    R = np.array([
        [1.0 - (yy + zz),     xy - wz,           xz + wy],
        [xy + wz,             1.0 - (xx + zz),   yz - wx],
        [xz - wy,             yz + wx,           1.0 - (xx + yy)]
    ], dtype=float)
    return R

def rotX_matrix(deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)

def apply_global_rotX(points, Rs, deg):
    """Applica rotazione globale Rx(deg) a posizioni e rotazioni (pre-molt)."""
    if abs(deg) < 1e-9:
        return points, Rs
    Rx = rotX_matrix(deg)
    # posizioni
    pts_rot = points @ Rx.T  # (F,24,3)
    # rotazioni: R_global_new = Rx * R_global
    if Rs is None:
        return pts_rot, None
    F, J, _ = Rs.shape
    Rs_new = np.empty_like(Rs)
    for f in range(F):
        for j in range(J):
            Rs_new[f,j] = Rx @ Rs[f,j]
    return pts_rot, Rs_new

def set_axes_equal_and_limits(ax, xyz_min, xyz_max):
    x_min, y_min, z_min = xyz_min
    x_max, y_max, z_max = xyz_max
    ax.set_xlim([x_min, x_max]); ax.set_ylim([y_min, y_max]); ax.set_zlim([z_min, z_max])
    rng = np.array([x_max-x_min, y_max-y_min, z_max-z_min], dtype=float)
    mr = rng.max()
    cx = (x_min+x_max)/2; cy=(y_min+y_max)/2; cz=(z_min+z_max)/2
    ax.set_xlim(cx-mr/2, cx+mr/2); ax.set_ylim(cy-mr/2, cy+mr/2); ax.set_zlim(cz-mr/2, cz+mr/2)

# ============== MAIN ==============
def main():
    parser = argparse.ArgumentParser(
        description="GIF MoCap con ossa + (opzionale) triad degli assi locali da quaternioni globali."
    )
    parser.add_argument("pos_json", type=Path, help="JSON posizioni (frame_*: 24x3).")
    parser.add_argument("--rot_json", type=Path, default=None, help="JSON rotazioni (frame_*: 24x4, quaternioni x,y,z,w).")
    parser.add_argument("--out", type=Path, default=Path("mocap.gif"), help="Output GIF.")
    parser.add_argument("--fps", type=int, default=100, help="FPS GIF (default 100).")
    parser.add_argument("--scale", type=float, default=1.0, help="Scala posizioni (es. 0.001 per mm→m).")
    parser.add_argument("--elev", type=float, default=20.0, help="Vista elevazione (gradi).")
    parser.add_argument("--azim", type=float, default=-60.0, help="Vista azimut (gradi).")
    parser.add_argument("--dpi", type=int, default=120, help="DPI GIF.")
    parser.add_argument("--point_size", type=int, default=12, help="Dimensione punti joint.")
    parser.add_argument("--show_axes", action="store_true", help="Mostra assi globali (label).")
    parser.add_argument("--axes_scale", type=float, default=0.1, help="Lunghezza assi locali per i triad (in unità dei dati).")
    parser.add_argument("--axes_subset", type=str, default="Hips,Head,LeftHand,RightHand,LeftFoot,RightFoot",
                        help="CSV dei joint per cui disegnare i triad (default: pochi chiave). Usa 'ALL' per tutti.")
    parser.add_argument("--global_rx_deg", type=float, default=0.0, help="Rotazione globale di -/+deg sull'asse X (es. -90 per Blender).")
    args = parser.parse_args()

    # Carica posizioni
    frames = load_all_frames_positions(args.pos_json, scale=args.scale)  # (F,24,3)
    F = frames.shape[0]
    if F == 0:
        raise ValueError("Nessun frame di posizioni caricato.")
    if frames.shape[1:] != (24,3):
        raise ValueError(f"Shape posizioni inattesa: {frames.shape}")

    # Rotazioni opzionali: carica e converti a matrici 3x3 (globali)
    frames_R = None
    triad_indices = []
    if args.rot_json is not None:
        qs = load_all_frames_rotations(args.rot_json)  # (F,24,4)
        if qs.shape[0] != F:
            raise ValueError(f"Frames pos ({F}) e rot ({qs.shape[0]}) non coincidono.")
        # quaternioni -> matrici
        frames_R = np.empty((F, 24, 3, 3), dtype=float)
        for f in range(F):
            for j in range(24):
                frames_R[f,j] = quat_to_R(tuple(qs[f,j]))

        # Selezione subset joint per triad
        if args.axes_subset.strip().upper() == "ALL":
            triad_indices = list(range(24))
        else:
            wanted = [s.strip() for s in args.axes_subset.split(",") if s.strip()]
            name2idx = {n:i for i,n in enumerate(JOINTS)}
            triad_indices = [name2idx[n] for n in wanted if n in name2idx]

    # Applica rotazione globale Rx (se richiesta)
    frames, frames_R = apply_global_rotX(frames, frames_R, args.global_rx_deg)

    # Limiti fissi
    xyz_min = frames.reshape(-1,3).min(axis=0)
    xyz_max = frames.reshape(-1,3).max(axis=0)

    edges = build_edges()

    # Figure
    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=args.elev, azim=args.azim)
    set_axes_equal_and_limits(ax, xyz_min, xyz_max)
    if args.show_axes:
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    else:
        ax.set_axis_off()

    # Oggetti grafici
    pts0 = frames[0]
    scat = ax.scatter(pts0[:,0], pts0[:,1], pts0[:,2], s=args.point_size)

    bone_lines = []
    for (i,j) in edges:
        ln, = ax.plot([pts0[i,0], pts0[j,0]],
                      [pts0[i,1], pts0[j,1]],
                      [pts0[i,2], pts0[j,2]],
                      linewidth=2)
        bone_lines.append(ln)

    # Triad per orientamenti (se disponibili)
    # Per ogni joint selezionato: 3 linee (asse x,y,z locali)
    triad_lines = []
    if frames_R is not None and len(triad_indices) > 0:
        L = args.axes_scale
        for j in triad_indices:
            p = pts0[j]
            R = frames_R[0, j]
            # assi locali unitari
            ex, ey, ez = R[:,0], R[:,1], R[:,2]
            for u in (ex, ey, ez):
                q = p + L*u
                ln, = ax.plot([p[0], q[0]],[p[1], q[1]],[p[2], q[2]], linewidth=1.5)
                triad_lines.append(ln)

    title = ax.set_title(f"Frame 1 / {F}")

    def update(fi):
        pts = frames[fi]
        # punti
        scat._offsets3d = (pts[:,0], pts[:,1], pts[:,2])
        # ossa
        for ln,(i,j) in zip(bone_lines, edges):
            ln.set_data([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]])
            ln.set_3d_properties([pts[i,2], pts[j,2]])
        # triad
        if frames_R is not None and len(triad_indices) > 0:
            L = args.axes_scale
            idx = 0
            for j in triad_indices:
                p = pts[j]
                R = frames_R[fi, j]
                ex, ey, ez = R[:,0], R[:,1], R[:,2]
                for u in (ex, ey, ez):
                    q = p + L*u
                    ln = triad_lines[idx]
                    ln.set_data([p[0], q[0]], [p[1], q[1]])
                    ln.set_3d_properties([p[2], q[2]])
                    idx += 1
        title.set_text(f"Frame {fi+1} / {F}")
        return [scat, *bone_lines, *triad_lines, title]

    anim = FuncAnimation(fig, update, frames=F, interval=1000/args.fps, blit=False)

    # Salva GIF
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=args.fps)
    print(f"Scrivo GIF: {args.out} (fps={args.fps}, dpi={args.dpi}) …")
    anim.save(str(args.out), writer=writer, dpi=args.dpi)
    print("Fatto!")

if __name__ == "__main__":
    main()
