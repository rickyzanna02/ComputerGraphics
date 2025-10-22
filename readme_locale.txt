//ESPORTA PositionData 3X24X12000 IN JSON

%% Export PositionData (3x24xN) -> JSON { "frame_i": [[x,y,z], ...] }
clear; clc;

matFile = 'Nick_3.mat';             % <-- cambia se necessario
outJson = 'position_data.json';      % file di output

% 1) Carica il .mat
S = load(matFile);

% 2) Estrai PositionData
%    Atteso: 3 x 24 x N (double)
PD = S.Nick_3.Skeletons.PositionData;

% Controlli di forma
if ndims(PD) ~= 3 || size(PD,1) ~= 3
    error('PositionData deve essere 3 x 24 x N. Trovato: %s', mat2str(size(PD)));
end
if size(PD,2) ~= 24
    warning('Seconda dimensione non è 24 (è %d). Procedo comunque.', size(PD,2));
end

numFrames = size(PD, 3);

% 3) Costruisci la struct per il JSON
%    Per ogni frame: PD(:,:,f) è 3x24 -> trasponi per ottenere 24x3
outStruct = struct();
for f = 1:numFrames
    % 3x24 -> (24x3) righe = joint, colonne = [x y z]
    M = squeeze(PD(:,:,f)).';   % 24 x 3
    key = sprintf('frame_%d', f);
    outStruct.(key) = M;        % jsonencode produrrà una lista di liste
end

% 4) Serializza e salva
jsonStr = jsonencode(outStruct, 'PrettyPrint', true);

fid = fopen(outJson, 'w');
assert(fid ~= -1, 'Impossibile aprire il file di output per scrittura.');
fwrite(fid, jsonStr, 'char');
fclose(fid);

fprintf('✅ Esportazione completata: %s\n', outJson);

-------------------------------------------------------------------------------------------------------------------------------------

//ESPORTA ROTATION 4X24X12000 IN JSON
%% Export Nick_3.Skeletons.RotationData (4x24xN) -> JSON { "frame_i": [[q1 q2 q3 q4], ...] }
clear; clc;

matFile = 'Nick_3.mat';          % <-- cambia se necessario
outJson = 'rotation_data.json';   % file di output

% 1) Carica il .mat
S = load(matFile);

% 2) Estrai RotationData (atteso: 4 x 24 x N)
RD = S.Nick_3.Skeletons.RotationData;

% 3) Validazione dimensioni
sz = size(RD);
if ndims(RD) ~= 3 || sz(1) ~= 4 || sz(2) ~= 24
    error('RotationData deve essere 4x24xN. Trovato: %s', mat2str(sz));
end
numFrames = sz(3);

% 4) Costruisci la struct per il JSON
%    Per ogni frame: RD(:,:,f) è 4x24 -> trasponi a 24x4 (una quartina per joint)
outStruct = struct();
for f = 1:numFrames
    M = squeeze(RD(:,:,f)).';     % 24 x 4
    key = sprintf('frame_%d', f); % usa '%04d' per zero padding
    outStruct.(key) = M;
end

% 5) Serializza e salva
jsonStr = jsonencode(outStruct, 'PrettyPrint', true);
fid = fopen(outJson, 'w');
assert(fid ~= -1, 'Impossibile aprire il file di output per scrittura.');
fwrite(fid, jsonStr, 'char');
fclose(fid);

fprintf('✅ Esportazione completata: %s (frames: %d)\n', outJson, numFrames);






///////////////////////////////////////////////////////////////////////

1) eseguire i 2 script sopra su matlab e scaricare i 2 json che vengono creati
2) python filter_frame.py .\position_data.json .\position_data_filtered.json --start 980 --end 1370
   python filter_frame.py .\rotation_data.json .\rotation_data_filtered.json --start 980 --end 1370



















   /////////////////////////////////
    script blender per esportare fbx (non va)
    import json, math
from mathutils import Vector, Quaternion
import bpy
from pathlib import Path

# ========= USER PARAMS =========

POSITIONS_JSON = r"C:\Users\ricky\Desktop\ComputerGraphics\ComputerGraphics\position_data_filtered.json"
ROTATIONS_JSON = r"C:\Users\ricky\Desktop\ComputerGraphics\ComputerGraphics\rotation_data_filtered.json"
OUTPUT_FBX     = r"C:\Users\ricky\Desktop\ComputerGraphics\ComputerGraphics\out_mocap.fbx"

# Scala unità: i tuoi dati paiono in mm -> converti a metri
POS_SCALE = 0.001

# FPS (adatta se necessario)
FPS = 100

# Joint order coerente con i tuoi file
JOINTS = [
    'Hips','Spine','Spine1','Spine2','Neck','Head',
    'LeftShoulder','LeftArm','LeftForeArm','LeftForeArmRoll','LeftHand',
    'RightShoulder','RightArm','RightForeArm','RightForeArmRoll','RightHand',
    'LeftUpLeg','LeftLeg','LeftFoot','LeftToeBase',
    'RightUpLeg','RightLeg','RightFoot','RightToeBase'
]

# Gerarchia standard (parent -> [children])
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

# ========= LOAD DATA =========
with open(POSITIONS_JSON, 'r') as f:
    pos_data = json.load(f)

with open(ROTATIONS_JSON, 'r') as f:
    rot_data = json.load(f)

# Ordina i frame per indice: "frame_1"..."frame_N"
def sorted_frames(d):
    def idx(k): return int(k.split('_')[-1])
    return sorted(d.keys(), key=idx)

frames = sorted_frames(pos_data)
assert frames == sorted_frames(rot_data), "Frames in positions e rotations non combaciano!"

# Sanity check dimensioni
num_j = len(JOINTS)
def check_frame_shape(frame_key):
    assert len(pos_data[frame_key]) == num_j, f"positions: joint count mismatch @ {frame_key}"
    assert len(rot_data[frame_key]) == num_j, f"rotations: joint count mismatch @ {frame_key}"
check_frame_shape(frames[0])

# ========= SCENE PREP =========
# Pulisci scena
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# FPS
bpy.context.scene.render.fps = FPS

# ========= BUILD ARMATURE (REST POSE FROM frame_1) =========
rest = frames[0]
positions_rest = {JOINTS[i]: Vector(pos_data[rest][i]) * POS_SCALE for i in range(num_j)}

# Crea armature
bpy.ops.object.armature_add(enter_editmode=True)
arm_obj = bpy.context.object
arm_obj.name = "MoCapArmature"
arm = arm_obj.data
arm.name = "MoCapArmatureData"

# Rinomina bone di default a Hips
arm.edit_bones[0].name = 'Hips'

# Helper: crea o ottieni bone
def ensure_bone(name):
    if name in arm.edit_bones:
        return arm.edit_bones[name]
    b = arm.edit_bones.new(name)
    return b

# Crea tutte le bones e posiziona head/tail
for j in JOINTS:
    b = ensure_bone(j)
    head = positions_rest[j]
    # Tail: prova a puntare verso un figlio, altrimenti usa un offset lungo Y
    children = PARENTS.get(j, [])
    if children:
        # media dei figli per una buona direzione iniziale
        tail_target = sum((positions_rest[c] for c in children), Vector((0,0,0))) / len(children)
        dir_vec = (tail_target - head)
        if dir_vec.length < 1e-6:
            dir_vec = Vector((0, 0.05, 0))
        tail = head + dir_vec.normalized() * 0.05
    else:
        tail = head + Vector((0, 0.05, 0))  # piccolo osso
    b.head = head
    b.tail = tail

# Applica parenting secondo PARENTS
for parent, childs in PARENTS.items():
    for ch in childs:
        if ch in arm.edit_bones and parent in arm.edit_bones:
            arm.edit_bones[ch].parent = arm.edit_bones[parent]
            arm.edit_bones[ch].use_connect = False  # evitiamo deformazioni forzate

# Esci da Edit Mode
bpy.ops.object.mode_set(mode='OBJECT')

# ========= POSE MODE SETTINGS =========
bpy.context.view_layer.objects.active = arm_obj
bpy.ops.object.mode_set(mode='POSE')

# Rotazioni in quaternione
for j in JOINTS:
    p = arm_obj.pose.bones[j]
    p.rotation_mode = 'QUATERNION'

# ========= ANIMATION (KEYFRAMES) =========
# Funzione: from [x,y,z,w] to Blender Quaternion (w,x,y,z)
def q_from_json(qxyzw):
    x,y,z,w = qxyzw
    return Quaternion((w, x, y, z))

# Imposta range timeline
start = 1
end = start + len(frames) - 1
bpy.context.scene.frame_start = start
bpy.context.scene.frame_end = end

for fi, fkey in enumerate(frames, start=start):
    # Sanity per frame
    check_frame_shape(fkey)
    # Set frame
    bpy.context.scene.frame_set(fi)

    # Root motion: Hips location dalla positions
    hips_idx = JOINTS.index('Hips')
    hips_loc = Vector(pos_data[fkey][hips_idx]) * POS_SCALE
    pb_hips = arm_obj.pose.bones['Hips']
    # Porta l'armatura all'origine: usiamo la location del root come delta rispetto al primo frame
    if fi == start:
        hips_loc0 = hips_loc.copy()
    root_delta = hips_loc - hips_loc0
    arm_obj.location = root_delta  # root motion sull'oggetto armatura

    arm_obj.keyframe_insert(data_path="location", frame=fi)

    # Rotazioni per TUTTE le bones (locali)
    for j_i, jname in enumerate(JOINTS):
        q = q_from_json(rot_data[fkey][j_i])
        pb = arm_obj.pose.bones[jname]
        pb.rotation_quaternion = q
        pb.keyframe_insert(data_path="rotation_quaternion", frame=fi)

# Torna in Object Mode
bpy.ops.object.mode_set(mode='OBJECT')

print("✅ Animazione creata. Pronta per export FBX.")

# ========= (OPZIONALE) EXPORT FBX per Unreal =========
# Consigliato: seleziona solo l'armatura
bpy.ops.object.select_all(action='DESELECT')
arm_obj.select_set(True)
bpy.context.view_layer.objects.active = arm_obj

bpy.ops.export_scene.fbx(
    filepath=OUTPUT_FBX,
    use_selection=True,
    apply_unit_scale=True,
    bake_space_transform=True,
    add_leaf_bones=False,
    armature_nodetype='NULL',   # evita bone extra
    object_types={'ARMATURE'},
    use_armature_deform_only=True,
    bake_anim=True,
    bake_anim_use_all_bones=True,
    bake_anim_force_startend_keying=True,
    bake_anim_simplify_factor=0.0,
    path_mode='AUTO',
    axis_forward='-Z',  # per UE
    axis_up='Y'
)

print(f"📦 FBX esportato: {OUTPUT_FBX}")

