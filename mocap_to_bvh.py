#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, math
from pathlib import Path
import argparse
import numpy as np

JOINTS = [
    'Hips','Spine','Spine1','Spine2','Neck','Head',
    'LeftShoulder','LeftArm','LeftForeArm','LeftForeArmRoll','LeftHand',
    'RightShoulder','RightArm','RightForeArm','RightForeArmRoll','RightHand',
    'LeftUpLeg','LeftLeg','LeftFoot','LeftToeBase',
    'RightUpLeg','RightLeg','RightFoot','RightToeBase'
]
CHILDREN = {
    'Hips':['Spine','LeftUpLeg','RightUpLeg'],
    'Spine':['Spine1'],
    'Spine1':['Spine2'],
    'Spine2':['Neck','LeftShoulder','RightShoulder'],
    'Neck':['Head'],'Head':[],
    'LeftShoulder':['LeftArm'],'LeftArm':['LeftForeArm'],
    'LeftForeArm':['LeftForeArmRoll'],'LeftForeArmRoll':['LeftHand'],'LeftHand':[],
    'RightShoulder':['RightArm'],'RightArm':['RightForeArm'],
    'RightForeArm':['RightForeArmRoll'],'RightForeArmRoll':['RightHand'],'RightHand':[],
    'LeftUpLeg':['LeftLeg'],'LeftLeg':['LeftFoot'],'LeftFoot':['LeftToeBase'],'LeftToeBase':[],
    'RightUpLeg':['RightLeg'],'RightLeg':['RightFoot'],'RightFoot':['RightToeBase'],'RightToeBase':[]
}
PARENT = {c:p for p,cs in CHILDREN.items() for c in cs}
PARENT['Hips'] = None

# ---------- math ----------
def q_norm(q):
    n = np.linalg.norm(q);  return q if n < 1e-20 else (q/n)
def q_mul(a,b):
    ax,ay,az,aw = a; bx,by,bz,bw = b
    return np.array([aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw,
                     aw*bw-ax*bx-ay*by-az*bz], float)
def q_inv(q):
    x,y,z,w = q; n2 = x*x+y*y+z*z+w*w
    return np.array([-x,-y,-z,w], float)/(n2 if n2>1e-20 else 1.0)
def rotX_quat(deg):
    a = math.radians(deg); s = math.sin(a/2); c = math.cos(a/2)
    return np.array([s,0,0,c], float)
def R_from_q(q):
    x,y,z,w = q_norm(q); xx,yy,zz = x*x,y*y,z*z; xy,xz,yz = x*y,x*z,y*z; wx,wy,wz = w*x,w*y,w*z
    return np.array([[1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
                     [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
                     [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]], float)
def euler_ZXY_from_R(R):
    sy = -R[0,1]; sy = max(-1.0, min(1.0, sy))
    x = math.asin(sy); cx = math.cos(x)
    if abs(cx) < 1e-8:
        y = 0.0; z = math.atan2(-R[1,2], R[1,1])
    else:
        y = math.atan2(R[0,2], R[0,0])
        z = math.atan2(R[2,1], R[1,1])
    return math.degrees(z), math.degrees(x), math.degrees(y)
def rot_x_mat(deg):
    a = math.radians(deg); c,s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], float)

# ---------- IO helpers ----------
def read_positions(frame_dict, Rglob, scale):
    arr = np.array(frame_dict, float)  # (24,3)
    arr = (Rglob @ arr.T).T * scale
    return {JOINTS[i]: arr[i] for i in range(len(JOINTS))}

def read_quats(frame_dict, order='xyzw', comp='pre', qglob=None):
    """
    order: 'xyzw' (default) o 'wxyz'
    comp:  'pre' -> q_corr = qglob * q_world   (ruoto sistema Mondo)
           'post'-> q_corr = q_world * qglob   (ruoto locale dei segmenti)
    """
    out = {}
    for i,name in enumerate(JOINTS):
        vals = frame_dict[i]
        if order == 'wxyz':
            w,x,y,z = vals; q = np.array([x,y,z,w], float)
        else:
            x,y,z,w = vals; q = np.array([x,y,z,w], float)
        q = q_norm(q)
        if qglob is not None:
            q = q_mul(qglob, q) if comp=='pre' else q_mul(q, qglob)
        out[name] = q_norm(q)
    return out

# ---------- BVH ----------
def write_bvh(out_path, fps, offsets, endsites, frames_euler, root_positions):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("HIERARCHY\n")
        def emit(j, d=0):
            ind = "  "*d; root = PARENT[j] is None
            f.write(f"{ind}{'ROOT' if root else 'JOINT'} {j}\n{ind}{{\n")
            off = offsets[j]; f.write(f"{ind}  OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}\n")
            if root:
                f.write(f"{ind}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
            else:
                f.write(f"{ind}  CHANNELS 3 Zrotation Xrotation Yrotation\n")
            for c in CHILDREN[j]: emit(c, d+1)
            if len(CHILDREN[j])==0:
                es = endsites[j]
                f.write(f"{ind}  End Site\n{ind}  {{\n")
                f.write(f"{ind}    OFFSET {es[0]:.6f} {es[1]:.6f} {es[2]:.6f}\n")
                f.write(f"{ind}  }}\n")
            f.write(f"{ind}}}\n")
        emit('Hips')
        N = len(frames_euler)
        f.write("MOTION\n")
        f.write(f"Frames: {N}\n")
        f.write(f"Frame Time: {1.0/float(fps):.6f}\n")

        order=[]
        def collect(j):
            order.append((j, PARENT[j] is None))
            for c in CHILDREN[j]: collect(c)
        collect('Hips')

        for i in range(N):
            row=[]
            for j,is_root in order:
                z,x,y = frames_euler[i][j]
                if is_root:
                    px,py,pz = root_positions[i]
                    row += [f"{px:.6f}", f"{py:.6f}", f"{pz:.6f}",
                            f"{z:.6f}", f"{x:.6f}", f"{y:.6f}"]
                else:
                    row += [f"{z:.6f}", f"{x:.6f}", f"{y:.6f}"]
            f.write(" ".join(row)+"\n")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Qualisys (pos+quat global) -> BVH con zero-rot al frame_1.")
    ap.add_argument("--positions", required=True)
    ap.add_argument("--rotations", required=True)
    ap.add_argument("--output", default="out_mocap.bvh")
    ap.add_argument("--fps", type=int, default=100)
    ap.add_argument("--pos-scale", type=float, default=0.001)
    ap.add_argument("--rotate-x-deg", type=float, default=-90.0, help="correzione globale asse X")
    ap.add_argument("--quat-order", choices=["xyzw","wxyz"], default="xyzw")
    ap.add_argument("--quat-compose", choices=["pre","post"], default="pre",
                    help="pre: q'=qglob*q  |  post: q'=q*qglob")
    ap.add_argument("--endsite-scale", type=float, default=0.3)
    ap.add_argument("--flip-x", action="store_true", help="inverti asse X delle POSIZIONI dopo la rot globale")
    ap.add_argument("--flip-z", action="store_true", help="inverti asse Z delle POSIZIONI dopo la rot globale")
    args = ap.parse_args()

    pos_json = json.loads(Path(args.positions).read_text(encoding="utf-8"))
    rot_json = json.loads(Path(args.rotations).read_text(encoding="utf-8"))
    frames = sorted(pos_json.keys(), key=lambda k:int(k.split("_")[-1]))
    if frames != sorted(rot_json.keys(), key=lambda k:int(k.split("_")[-1])):
        raise RuntimeError("Frames in positions e rotations non coincidono.")

    Rglob = rot_x_mat(args.rotate_x_deg)
    qglob = rotX_quat(args.rotate_x_deg)

    def get_pos(k):
        P = np.array(pos_json[k], float)
        P = (Rglob @ P.T).T * args.pos_scale
        if args.flip_x: P[:,0] *= -1
        if args.flip_z: P[:,2] *= -1
        return {JOINTS[i]: P[i] for i in range(24)}

    def get_q_world(k):
        return read_quats(rot_json[k], order=args.quat_order, comp=args.quat_compose, qglob=qglob)

    # rest (frame_1) offsets
    rest_pos = get_pos(frames[0])
    offsets={}
    for j in JOINTS:
        p = PARENT[j]
        offsets[j] = np.array([0,0,0], float) if p is None else (rest_pos[j]-rest_pos[p])

    # end sites
    endsites={}
    for j in JOINTS:
        if len(CHILDREN[j])==0:
            p = PARENT[j]
            if p is None: endsites[j]=np.array([0,0,0],float)
            else:
                v = rest_pos[j]-rest_pos[p]; L = np.linalg.norm(v)
                endsites[j] = (v/L * (L*args.endsite_scale)) if L>1e-8 else np.array([0,0,0],float)
        else:
            endsites[j]=np.array([0,0,0],float)

    # --- compute local (zeroed at rest) ---
    # world quats for frame_1
    world_q_rest = get_q_world(frames[0])

    # local_rest = inv(parent_world_rest) * world_rest
    local_rest={}
    def visit_rest(j):
        p = PARENT[j]
        if p is None:
            local_rest[j] = world_q_rest[j]
        else:
            local_rest[j] = q_norm(q_mul(q_inv(world_q_rest[p]), world_q_rest[j]))
        for c in CHILDREN[j]: visit_rest(c)
    visit_rest('Hips')

    frames_euler=[]
    root_positions=[]
    for k in frames:
        pos = get_pos(k)
        world_q = get_q_world(k)
        root_positions.append(pos['Hips'])

        local_euler={}
        def visit(j):
            p = PARENT[j]
            if p is None:
                local_now = world_q[j]
            else:
                local_now = q_norm(q_mul(q_inv(world_q[p]), world_q[j]))
            # zero-at-rest: remove rest local orientation
            local_rel = q_norm(q_mul(q_inv(local_rest[j]), local_now))
            R = R_from_q(local_rel)
            local_euler[j] = euler_ZXY_from_R(R)
            for c in CHILDREN[j]: visit(c)
        visit('Hips')
        frames_euler.append(local_euler)

    write_bvh(args.output, args.fps, offsets, endsites, frames_euler, root_positions)
    print(f"OK: {args.output}  | frames={len(frames)} fps={args.fps} rotX={args.rotate_x_deg}° order={args.quat_order} comp={args.quat_compose}")
    print("Nota: rotazioni local azzerate al frame_1 (rest). Se ancora storto, prova --quat-compose post, --quat-order wxyz, o --flip-x/--flip-z.")

if __name__ == "__main__":
    main()

#python mocap_to_bvh.py --positions position_data_filtered.json  --rotations rotation_data_filtered.json  --output out_mocap.bvh --fps 100