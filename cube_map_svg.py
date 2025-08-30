import bpy
import math
import mathutils
import os
import re
import json
from pathlib import Path
from datetime import datetime
#=========================
#CONFIG
#=========================

output_dir = "/path/to/cube_svg_captures/"   # Blend-relative path; use absolute if you prefer
base_name  = "freestyle_cube"  # Prefix for outputs
resolution = 2048              # Square render size, e.g., 1024, 2048, 4096
fov_deg    = 90.0              # Cube faces are 90° FOV
#If you want to target a specific camera by name; otherwise active scene camera is used

camera_name = "Camera"  # e.g. "Camera"
#=========================
#HELPER FUNCTIONS
#=========================

def ensure_dir(path):
    apath = bpy.path.abspath(path)
    os.makedirs(apath, exist_ok=True)
    return apath

def rotation_from_forward_up(forward, up):
    """
    Build a rotation (camera local -> world) so that:
    - camera forward (-Z) maps to given 'forward' (world)
    - camera up (+Y) maps to given 'up' (world)
    Returns a mathutils.Quaternion and 3x3 rotation matrix.
    """
    f = mathutils.Vector(forward).normalized()
    u = mathutils.Vector(up).normalized()
    # Right = forward x up
    r = f.cross(u)
    if r.length < 1e-8:
        raise ValueError("forward and up are collinear")
    r.normalize()
    # Re-orthogonalize up to guarantee an orthonormal basis
    u_ortho = r.cross(f)
    # Columns of rotation matrix are world-space images of camera axes:
    # cam X -> r, cam Y -> u_ortho, cam Z -> -f
    R = mathutils.Matrix((r, u_ortho, -f)).transposed()  # put r,u,-f as columns
    q = R.to_quaternion()
    return q, R

def get_camera(scene, camera_name=None):
    if camera_name:
        cam_obj = bpy.data.objects.get(camera_name)
    if cam_obj and cam_obj.type == 'CAMERA':
        return cam_obj
        raise RuntimeError(f"Camera '{camera_name}' not found or not a CAMERA object.")
    if scene.camera and scene.camera.type == 'CAMERA':
        return scene.camera
    # Fallback: first camera in the file
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            return obj
            raise RuntimeError("No camera found in the scene.")

def set_square_resolution(scene, size):
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100

def set_camera_fov_deg(cam_data, fov_deg):
    # Use square resolution so angle_x == angle_y; set generic camera FOV
    cam_data.type = 'PERSP'
    # Let sensor_fit AUTO with square aspect; angle becomes both horiz/vert
    cam_data.sensor_fit = 'AUTO'
    cam_data.angle = math.radians(fov_deg)

def get_intrinsics(scene, cam_obj):
    camd = cam_obj.data
    W = scene.render.resolution_x
    H = scene.render.resolution_y
    fov_x = camd.angle_x
    fov_y = camd.angle_y
    fx = W / (2.0 * math.tan(fov_x * 0.5))
    fy = H / (2.0 * math.tan(fov_y * 0.5))
    cx = W * 0.5
    cy = H * 0.5
    return dict(width=W, height=H, fov_x_rad=float(fov_x), fov_y_rad=float(fov_y), fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))

def try_enable_addon(module_name):
    try:
        if module_name not in bpy.context.preferences.addons:
            bpy.ops.preferences.addon_enable(module=module_name)
            print(f"Enabled add-on: {module_name}")
    except Exception as e:
        print(f"Could not enable add-on {module_name}: {e}")
        
def rename_svg_files(directory_path, dry_run=True):
    """
    Rename SVG files to remove the extra 4-digit suffix that Blender adds
    
    Args:
        directory_path: Path to the directory containing SVG files
        dry_run: If True, only shows what would be renamed (default: True for safety)
    """
    
    directory = Path(directory_path)
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return
    
    # More flexible pattern: match any filename ending with 4 digits + .svg
    # Captures everything before the 4 digits as group 1
    pattern = r'^(.+?)(\d{4})\.svg$'
    
    renamed_count = 0
    files_found = list(directory.glob("*.svg"))
    
    print(f"Found {len(files_found)} SVG files in {directory}")
    
    for file_path in files_found:
        filename = file_path.name
        match = re.match(pattern, filename)
        
        if match:
            # Extract the base name without the 4-digit suffix
            base_name = match.group(1)
            digit_suffix = match.group(2)
            new_filename = f"{base_name}.svg"
            new_path = file_path.parent / new_filename
            
            # Skip if the new filename would be the same (no actual suffix to remove)
            if filename == new_filename:
                continue
                
            # Skip if target file already exists
            if new_path.exists():
                print(f"Warning: Target already exists, skipping: {filename} → {new_filename}")
                continue
            
            if dry_run:
                print(f"Would rename: {filename} → {new_filename} (removing '{digit_suffix}')")
            else:
                try:
                    file_path.rename(new_path)
                    print(f"Renamed: {filename} → {new_filename}")
                    renamed_count += 1
                except Exception as e:
                    print(f"Error renaming {filename}: {e}")
    
    if dry_run:
        matches = [f for f in files_found if re.match(pattern, f.name) and f.name != re.match(pattern, f.name).group(1) + ".svg"]
        print(f"\nDry run complete. Would rename {len(matches)} files.")
        print("Run with dry_run=False to actually rename the files.")
    else:
        print(f"\nRenamed {renamed_count} files successfully.")
#=========================
#MAIN
#=========================

def main():
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer

    # Try to ensure the SVG exporter is enabled (won't error if missing)
    try_enable_addon("render_freestyle_svg")

    # Ensure Freestyle
    scene.render.use_freestyle = True

    # Output dir
    outdir = ensure_dir(output_dir)

    # Camera
    cam_obj = get_camera(scene, camera_name=camera_name)
    camd = cam_obj.data

    # Save original state to restore later
    orig = {
        "res_x": scene.render.resolution_x,
        "res_y": scene.render.resolution_y,
        "res_pct": scene.render.resolution_percentage,
        "filepath": scene.render.filepath,
        "use_freestyle": scene.render.use_freestyle,
        "cam_loc": cam_obj.location.copy(),
        "cam_rot_mode": cam_obj.rotation_mode,
        "cam_rot_euler": cam_obj.rotation_euler.copy(),
        "cam_rot_quat": cam_obj.rotation_quaternion.copy(),
        "cam_angle": getattr(camd, "angle", None),
        "cam_sensor_fit": camd.sensor_fit,
        "engine": scene.render.engine,
    }

    # Recommended engine: Eevee or Cycles—Freestyle works with both in recent Blender
    # Leave current engine as-is.

    # Configure square resolution and 90° FOV
    set_square_resolution(scene, resolution)
    set_camera_fov_deg(camd, fov_deg)

    # Switch to quaternion rotation for reliable face orientations
    cam_obj.rotation_mode = 'QUATERNION'

    # Define 6 cube faces (forward, up). These are conventional and consistent.
    faces = [
        # name, forward, up
        ("+X", ( 1,  0,  0), (0,  1,  0)),
        ("-X", (-1,  0,  0), (0,  1,  0)),
        ("+Y", ( 0,  1,  0), (0,  0, -1)),
        ("-Y", ( 0, -1,  0), (0,  0,  1)),
        ("+Z", ( 0,  0,  1), (0,  1,  0)),
        ("-Z", ( 0,  0, -1), (0,  1,  0)),
    ]

    # Collect session-wide metadata
    meta = {
        "blender_version": bpy.app.version_string,
        "timestamp": datetime.now().isoformat(),
        "output_dir": os.path.abspath(outdir),
        "base_name": base_name,
        "camera_object": cam_obj.name,
        "camera_location_world": list(cam_obj.location),
        "camera_forward_cam": [0, 0, -1],
        "camera_up_cam": [0, 1, 0],
        "notes": "Render 6 Freestyle SVGs for cubemap; rotations are world_from_cam.",
        "faces": [],
    }

    # Intrinsics (same for all faces if square res and same FOV)
    intr = get_intrinsics(scene, cam_obj)
    meta["intrinsics"] = intr

    # Current frame
    frame = scene.frame_current

    for face_name, forward, up in faces:
        # Compute rotation to align camera forward (-Z) to 'forward', and up to 'up'
        q, R = rotation_from_forward_up(forward, up)
        # Apply rotation (location unchanged)
        cam_obj.rotation_quaternion = q

        # Update depsgraph so matrices are current
        bpy.context.view_layer.update()

        # Filepath base for this face (used by render and picked up by SVG exporter)
        filepath_base = os.path.join(outdir, f"{base_name}_{face_name}")
        scene.render.filepath = filepath_base

        print(f"Rendering face {face_name} to base {filepath_base} ...")
        # Render one still frame; write_still=True writes the image to disk.
        bpy.ops.render.render(write_still=True, use_viewport=False)

        # Grab the final camera world rotation matrix after application
        R_wc = cam_obj.matrix_world.to_3x3()
        q_wc = R_wc.to_quaternion()

        face_meta = {
            "name": face_name,
            "forward_world": list(forward),
            "up_world": list(up),
            # Rotation mapping camera local -> world
            "R_world_from_cam_3x3_rowmajor": [list(R_wc[0]), list(R_wc[1]), list(R_wc[2])],
            "R_world_from_cam_quat_wxyz": [float(q_wc.w), float(q_wc.x), float(q_wc.y), float(q_wc.z)],
            # Camera location (not needed for reprojection, but included)
            "camera_location_world": list(cam_obj.location),
            # Output name base (SVG exporter will write alongside)
            "output_base": filepath_base,
            # Expected SVG filename guess (depends on exporter; this is a conventional hint)
            "expected_svg": filepath_base + ".svg",
            # Intrinsics copied for convenience
            "intrinsics": intr,
        }
        meta["faces"].append(face_meta)

    # Write metadata JSON
    json_path = os.path.join(outdir, f"{base_name}_metadata.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata: {json_path}")
    
    # Rewrite file name (hack)
    rename_svg_files(outdir, dry_run=False)

    # Restore original state
    scene.render.resolution_x = orig["res_x"]
    scene.render.resolution_y = orig["res_y"]
    scene.render.resolution_percentage = orig["res_pct"]
    scene.render.filepath = orig["filepath"]
    scene.render.use_freestyle = orig["use_freestyle"]
    cam_obj.location = orig["cam_loc"]
    cam_obj.rotation_mode = orig["cam_rot_mode"]
    if cam_obj.rotation_mode == 'QUATERNION':
        cam_obj.rotation_quaternion = orig["cam_rot_quat"]
    else:
        cam_obj.rotation_euler = orig["cam_rot_euler"]
    if orig["cam_angle"] is not None:
        camd.angle = orig["cam_angle"]
    camd.sensor_fit = orig["cam_sensor_fit"]
    scene.render.engine = orig["engine"]

    print("Done. Camera and render settings restored.")


main()
