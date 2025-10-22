#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ================== JOINTS (ordine del tuo file) ==================
JOINTS = [
    'Hips','Spine','Spine1','Spine2','Neck','Head',
    'LeftShoulder','LeftArm','LeftForeArm','LeftForeArmRoll','LeftHand',
    'RightShoulder','RightArm','RightForeArm','RightForeArmRoll','RightHand',
    'LeftUpLeg','LeftLeg','LeftFoot','LeftToeBase',
    'RightUpLeg','RightLeg','RightFoot','RightToeBase'
]

# ================== GERARCHIA (parent -> children) ==================
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

# ================== UTILS ==================
def frame_key_variants(i: int):
    """Possibili chiavi del frame: 'frame_1' o 'frame_0001' ecc."""
    return [f"frame_{i}", f"frame_{i:04d}", f"frame_{i:03d}", f"frame_{i:05d}"]

def set_axes_equal_and_limits(ax, xyz_min, xyz_max):
    """Imposta limiti fissi e stessa scala sui tre assi."""
    x_min, y_min, z_min = xyz_min
    x_max, y_max, z_max = xyz_max
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    # equal aspect workaround for 3D
    ranges = np.array([x_max-x_min, y_max-y_min, z_max-z_min], dtype=float)
    max_range = ranges.max()
    x_c = (x_min + x_max) / 2.0
    y_c = (y_min + y_max) / 2.0
    z_c = (z_min + z_max) / 2.0
    ax.set_xlim(x_c - max_range/2, x_c + max_range/2)
    ax.set_ylim(y_c - max_range/2, y_c + max_range/2)
    ax.set_zlim(z_c - max_range/2, z_c + max_range/2)

def build_edges():
    """Costruisce lista di archi (indici) a partire da PARENTS e JOINTS."""
    name2idx = {name: i for i, name in enumerate(JOINTS)}
    edges = []
    for p, childs in PARENTS.items():
        if p not in name2idx:
            raise KeyError(f"Parent '{p}' non presente in JOINTS.")
        p_idx = name2idx[p]
        for c in childs:
            if c not in name2idx:
                raise KeyError(f"Child '{c}' non presente in JOINTS.")
            c_idx = name2idx[c]
            edges.append((p_idx, c_idx))
    return edges

def load_all_frames_positions(json_path: Path, scale: float):
    """Ritorna un array (F, 24, 3) con tutti i frame in ordine crescente."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # trova tutte le chiavi tipo 'frame_xxx' e ordinale per numero
    items = []
    for k in data.keys():
        if k.startswith("frame_"):
            # estrai numero finale robustamente
            suffix = k.split("frame_")[-1]
            try:
                num = int(suffix)
            except ValueError:
                # prova a ripulire zeri a sinistra o altro
                try:
                    num = int(suffix.lstrip("0") or "0")
                except ValueError:
                    continue
            items.append((num, k))
    if not items:
        raise ValueError("Nessuna chiave 'frame_*' trovata nel JSON.")

    items.sort(key=lambda t: t[0])  # ordina per numero di frame
    frames = []
    for _, k in items:
        pts = np.array(data[k], dtype=float)  # (24,3)
        if pts.shape != (24, 3):
            raise ValueError(f"Frame {k}: atteso (24,3), trovato {pts.shape}")
        frames.append(pts * scale)

    return np.stack(frames, axis=0)  # (F,24,3)

# ================== MAIN ==================
def main():
    parser = argparse.ArgumentParser(
        description="Crea una GIF dell'animazione MoCap con ossa secondo la gerarchia indicata."
    )
    parser.add_argument("json_path", type=Path, help="File JSON con posizioni per frame.")
    parser.add_argument("--out", type=Path, default=Path("mocap.gif"), help="Output GIF.")
    parser.add_argument("--fps", type=int, default=100, help="FPS della GIF (default: 100).")
    parser.add_argument("--scale", type=float, default=1.0, help="Scala (es. 0.001 per mm→m).")
    parser.add_argument("--elev", type=float, default=20.0, help="Vista elevazione in gradi.")
    parser.add_argument("--azim", type=float, default=-60.0, help="Vista azimut in gradi.")
    parser.add_argument("--dpi", type=int, default=120, help="DPI della GIF renderizzata.")
    parser.add_argument("--point_size", type=int, default=12, help="Dimensione pallini joints.")
    parser.add_argument("--show_axes", action="store_true", help="Mostra assi e label.")
    args = parser.parse_args()

    # Carica frames -> (F, 24, 3)
    frames = load_all_frames_positions(args.json_path, scale=args.scale)
    F = frames.shape[0]
    if F == 0:
        raise ValueError("Nessun frame caricato.")
    if frames.shape[1:] != (24, 3):
        raise ValueError(f"Shape inattesa: {frames.shape}, atteso (F,24,3).")

    # Calcola limiti globali stabili
    xyz_min = frames.reshape(-1, 3).min(axis=0)
    xyz_max = frames.reshape(-1, 3).max(axis=0)

    # Edge list dalla gerarchia
    edges = build_edges()

    # Setup figura
    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=args.elev, azim=args.azim)
    set_axes_equal_and_limits(ax, xyz_min, xyz_max)

    if args.show_axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    else:
        ax.set_axis_off()

    # Oggetti grafici: scatter + linee per ossa
    # init con primo frame
    pts0 = frames[0]
    scat = ax.scatter(pts0[:, 0], pts0[:, 1], pts0[:, 2], s=args.point_size)

    lines = []
    for (i, j) in edges:
        line, = ax.plot([pts0[i, 0], pts0[j, 0]],
                        [pts0[i, 1], pts0[j, 1]],
                        [pts0[i, 2], pts0[j, 2]],
                        linewidth=2)
        lines.append(line)

    title = ax.set_title(f"Frame 1 / {F}")

    def update(f_idx):
        pts = frames[f_idx]
        # aggiorna scatter
        scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        # aggiorna linee
        for line, (i, j) in zip(lines, edges):
            x = [pts[i, 0], pts[j, 0]]
            y = [pts[i, 1], pts[j, 1]]
            z = [pts[i, 2], pts[j, 2]]
            line.set_data(x, y)
            line.set_3d_properties(z)
        title.set_text(f"Frame {f_idx+1} / {F}")
        return [scat, *lines, title]

    anim = FuncAnimation(fig, update, frames=F, interval=1000/args.fps, blit=False)

    # Salvataggio GIF
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=args.fps)
    print(f"Scrivo GIF in: {args.out} (fps={args.fps}, dpi={args.dpi}) …")
    anim.save(str(args.out), writer=writer, dpi=args.dpi)
    print("Fatto!")

if __name__ == "__main__":
    main()


# python plot_mocap_frame.py ./position_data_filtered.json --out ./mocap_100fps.gif --fps 100 --scale 0.001
