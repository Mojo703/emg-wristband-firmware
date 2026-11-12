"""The band worn on a right forearm, in the user guide's line-art style.

    blender -b --factory-startup \
        --python-expr "import bpy;bpy.ops.preferences.addon_enable(module='bl_ext.blender_org.mpfb')" \
        --python band_worn.py -- <out.png> <pads.json>

The arm is the MPFB human the skier pipeline builds, cut down to the forearm
and hand, and the band is drawn see-through. The pod sits on the back of the
wrist; the black electrode strap runs under it and round the little-finger side
to the palm. Its two modules of 2 x 8 pads face the skin on opposite sides of
the wrist: one under the pod, one under the palm. Writes where each pad lands in
the image and whether it is on the near side, so the page can colour it by
contact.
"""
import bpy, bmesh, json, math, sys
from bpy_extras.object_utils import world_to_camera_view
from bl_ext.blender_org.mpfb.services.humanservice import HumanService
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
OUT, PADS_OUT = argv[0], argv[1]
MM = 0.001

scene = bpy.context.scene
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj)

body = HumanService.create_human()
rig = HumanService.add_builtin_rig(body, "default")
bpy.context.view_layer.update()


def joint(name):
    return rig.matrix_world @ rig.data.bones[name].head_local


elbow = joint("lowerarm01.R")
wrist = joint("wrist.R")
index_knuckle = joint("finger2-1.R")
little_knuckle = joint("finger5-1.R")

axis = (wrist - elbow).normalized()
fingers = ((index_knuckle + little_knuckle) / 2 - wrist).normalized()
across = (index_knuckle - little_knuckle).normalized()
dorsal = fingers.cross(across)
dorsal = (dorsal - axis * dorsal.dot(axis)).normalized()
ulnar = axis.cross(dorsal).normalized()

# Keep the forearm and hand: vertices near the elbow-to-fingertip segment.
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated = body.evaluated_get(depsgraph)
source = evaluated.to_mesh()
forearm_length = (wrist - elbow).length
bm = bmesh.new()
bm.from_mesh(source)
bm.transform(body.matrix_world)
remove = []
for vertex in bm.verts:
    along = (vertex.co - elbow).dot(axis)
    radial = (vertex.co - elbow - axis * along).length
    # Generous radius past the wrist so no cut crosses the hand: a cut edge draws
    # as a stray line, and the thumb reaches well off the forearm's axis.
    limit = 0.075 if along < forearm_length else 0.16
    if along < forearm_length * 0.3 or along > forearm_length * 2.6 or radial > limit:
        remove.append(vertex)
bmesh.ops.delete(bm, geom=remove, context='VERTS')
arm_mesh = bpy.data.meshes.new("arm")
bm.to_mesh(arm_mesh)
bm.free()
evaluated.to_mesh_clear()
arm = bpy.data.objects.new("arm", arm_mesh)
arm_collection = bpy.data.collections.new("arm")
band_collection = bpy.data.collections.new("band")
scene.collection.children.link(arm_collection)
scene.collection.children.link(band_collection)
arm_collection.objects.link(arm)
for modifier_type in ('SUBSURF',):
    arm.modifiers.new("smooth", modifier_type).levels = 1
bpy.data.objects.remove(body)
bpy.data.objects.remove(rig)

# The wrist's cross-section where the band sits, as radius by angle round the
# arm. Angle 0 is the back of the wrist, increasing toward the little finger.
center = wrist - axis * 32 * MM
BINS = 72
radius = [0.0] * BINS
for vertex in arm_mesh.vertices:
    offset = vertex.co - center
    if abs(offset.dot(axis)) > 6 * MM:
        continue
    flat = offset - axis * offset.dot(axis)
    if flat.length > 0.06:
        continue
    angle = math.atan2(flat.dot(ulnar), flat.dot(dorsal)) % (2 * math.pi)
    slot = int(angle / (2 * math.pi) * BINS) % BINS
    radius[slot] = max(radius[slot], flat.length)
known = [r for r in radius if r > 0]
mean_radius = sum(known) / len(known)
for slot in range(BINS):
    if radius[slot] == 0:
        radius[slot] = mean_radius
smoothed = [sum(radius[(slot + k) % BINS] for k in range(-2, 3)) / 5 for slot in range(BINS)]


def radius_at(angle):
    position = (angle % (2 * math.pi)) / (2 * math.pi) * BINS
    low = int(position) % BINS
    high = (low + 1) % BINS
    fraction = position - int(position)
    return smoothed[low] * (1 - fraction) + smoothed[high] * fraction


def round_point(angle, out, along):
    direction = dorsal * math.cos(angle) + ulnar * math.sin(angle)
    return center + axis * along + direction * (radius_at(angle) + out)


def link(obj, collection):
    collection.objects.link(obj)
    return obj


STRAP_HALF = 11 * MM
STRAP_THICK = 2.5 * MM


def ring(name, start, end, steps, half_width, thickness, collection):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    rows = []
    closed = abs(end - start - 2 * math.pi) < 1e-6
    count = steps if closed else steps + 1
    for step in range(count):
        angle = start + (end - start) * step / steps
        rows.append([bm.verts.new(round_point(angle, out, along))
                     for out, along in ((0.5 * MM, -half_width), (0.5 * MM, half_width),
                                        (0.5 * MM + thickness, half_width), (0.5 * MM + thickness, -half_width))])
    for index in range(len(rows) - (0 if closed else 1)):
        a, b = rows[index], rows[(index + 1) % len(rows)]
        for k in range(4):
            bm.faces.new((a[k], a[(k + 1) % 4], b[(k + 1) % 4], b[k]))
    if not closed:
        bm.faces.new(rows[0])
        bm.faces.new(list(reversed(rows[-1])))
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    return link(bpy.data.objects.new(name, mesh), collection)


# The pod's half-width, as an angle round the wrist at the back.
POD_LENGTH, POD_WIDTH, POD_HEIGHT = 50 * MM, 32 * MM, 12 * MM
pod_half_angle = (POD_WIDTH / 2) / radius_at(0)
# Electrode strap from the pod round the little-finger side; the slotted strap
# from the other side of the pod round the thumb side, meeting it at the palm.
# The electrode strap runs under the pod, so its pads sit between the pod and the skin.
ring("electrode_strap", -pod_half_angle * 1.4, math.radians(215), 56, STRAP_HALF, STRAP_THICK, band_collection)
ring("slotted_strap", math.radians(205), 2 * math.pi - pod_half_angle * 1.3, 48, STRAP_HALF, STRAP_THICK * 0.9, band_collection)

# Pod: a bevelled block on the back of the wrist, long axis along the arm.
pod_mesh = bpy.data.meshes.new("pod")
bm = bmesh.new()
bmesh.ops.create_cube(bm, size=1.0)
bmesh.ops.scale(bm, vec=(POD_LENGTH, POD_WIDTH, POD_HEIGHT), verts=bm.verts)
bm.to_mesh(pod_mesh)
bm.free()
pod = link(bpy.data.objects.new("pod", pod_mesh), band_collection)
basis = [axis, ulnar, dorsal]
from mathutils import Matrix
rotation = Matrix((basis[0], basis[1], basis[2])).transposed().to_4x4()
# Sunk 4 mm: a flat pod on a curved wrist rests on its long edges.
# The strap crosses under the pod's hand-side end; the body runs on toward the
# elbow (enclosure CAD), so the pod's centre sits that far up the arm.
POD_TOWARD_ELBOW = POD_LENGTH / 2 - STRAP_HALF - 3 * MM
pod.matrix_world = Matrix.Translation(center - axis * POD_TOWARD_ELBOW + dorsal * (radius_at(0) + STRAP_THICK + POD_HEIGHT / 2 - 4 * MM)) @ rotation
bevel = pod.modifiers.new("bevel", 'BEVEL')
bevel.width = 7 * MM
bevel.segments = 6
bevel.limit_method = 'NONE'

# Pads: two modules of 2 x 8 on the electrode strap's skin side, on opposite
# sides of the wrist: module 0 under the pod, module 1 under the palm side.
# Columns step 5.5 mm of arc about each module's centre.
PITCH = 5.5 * MM
pads = []
for module, middle in enumerate((0.0, math.pi)):
    for column in range(8):
        angle = middle + (column - 3.5) * PITCH / radius_at(middle)
        for row, along in enumerate((-3.6 * MM, 3.6 * MM)):
            pads.append({"channel": module * 8 + column, "row": row, "point": round_point(angle, 0.2 * MM, along)})

# Line art.
material = bpy.data.materials.new("fill")
material.use_nodes = True
nodes = material.node_tree.nodes
for node in list(nodes):
    nodes.remove(node)
# No fill at all: the page shows through, and the lines carry the drawing.
transparent = nodes.new("ShaderNodeBsdfTransparent")
output = nodes.new("ShaderNodeOutputMaterial")
material.node_tree.links.new(transparent.outputs[0], output.inputs[0])
for obj in scene.objects:
    if obj.type == 'MESH':
        obj.data.materials.clear()
        obj.data.materials.append(material)
        if obj.name != "arm":
            for polygon in obj.data.polygons:
                polygon.use_smooth = False

scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 1
scene.cycles.use_denoising = False
scene.view_settings.view_transform = 'Standard'
scene.render.film_transparent = True
scene.render.use_freestyle = True
scene.render.line_thickness_mode = 'ABSOLUTE'
settings = scene.view_layers[0].freestyle_settings
settings.crease_angle = math.radians(132)
settings.use_smoothness = True
for lineset in list(settings.linesets):
    settings.linesets.remove(lineset)


def add_lineset(name, thickness, collection, hidden=False, **flags):
    lineset = settings.linesets.new(name)
    lineset.select_by_collection = True
    lineset.collection = collection
    lineset.select_by_visibility = True
    lineset.visibility = 'HIDDEN' if hidden else 'VISIBLE'
    for key in ("select_silhouette", "select_border", "select_crease", "select_external_contour", "select_contour"):
        setattr(lineset, key, flags.get(key, False))
    style = lineset.linestyle
    style.color = (0.086, 0.137, 0.361)
    style.thickness = thickness
    style.caps = 'ROUND'
    if hidden:
        style.alpha = 0.55
        style.use_dashed_line = True
        style.dash1, style.gap1 = 7, 6
    return lineset


OUTLINES = dict(select_silhouette=True, select_border=True, select_external_contour=True, select_contour=True)
add_lineset("arm", 2.2, arm_collection, **OUTLINES)
add_lineset("band", 3.2, band_collection, **OUTLINES)
add_lineset("band_detail", 1.5, band_collection, select_crease=True)
# The band's far side, seen through the wrist.
add_lineset("band_hidden", 1.8, band_collection, hidden=True, **OUTLINES)

world = scene.world or bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)

# One see-through view. Everything is drawn, so the pads on the far side of the
# wrist show through rather than needing a second view.
camera_data = bpy.data.cameras.new("camera")
camera_data.lens = 50
camera = link(bpy.data.objects.new("camera", camera_data), scene.collection)
scene.camera = camera
# The wearer's own view: from above and behind the wrist, toward the elbow and a
# little toward the thumb, with the hand pointing away up the image.
along, toward_little_finger, above = map(float, (argv[2] if len(argv) > 2 else "-1.0,-0.12,0.65").split(","))
view = (axis * along + ulnar * toward_little_finger + dorsal * above).normalized()
target = center + axis * float(argv[3] if len(argv) > 3 else 0.02)
camera.location = target + view * float(argv[4] if len(argv) > 4 else 0.42)
forward = (target - camera.location).normalized()
up = (axis - forward * axis.dot(forward)).normalized()
right = forward.cross(up)
camera.matrix_world = Matrix.Translation(camera.location) @ Matrix((right, up, -forward)).transposed().to_4x4()
scene.render.resolution_x, scene.render.resolution_y = 1200, 760
scene.render.filepath = OUT

bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
out = []
for pad in pads:
    point = pad["point"]
    projected = world_to_camera_view(scene, camera, point)
    # A pad faces the skin, so the strap always covers it; what decides whether
    # this view shows it is whether its side of the wrist faces the camera.
    outward = (point - center - axis * (point - center).dot(axis)).normalized()
    facing = outward.dot((camera.location - point).normalized())
    out.append({
        "channel": pad["channel"],
        "row": pad["row"],
        "x": round(projected.x * scene.render.resolution_x, 1),
        "y": round((1 - projected.y) * scene.render.resolution_y, 1),
        "near": facing > 0,
    })
with open(PADS_OUT, "w") as handle:
    json.dump({"width": scene.render.resolution_x, "height": scene.render.resolution_y, "pads": out}, handle, indent=1)

bpy.ops.render.render(write_still=True)
print("RENDERED", OUT)
