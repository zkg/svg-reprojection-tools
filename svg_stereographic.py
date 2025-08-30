import argparse
import math
import os
import re
import sys
import time
import tkinter
from tkinter import filedialog
import xml.etree.ElementTree as ET

# ----------------------------
# Utilities
# ----------------------------

def progress_update(current: int, total: int, prefix='Progress', suffix='', length=50):
    if total <= 0:
        return
    completed = int(length * current // total)
    pct = 100 * (current / float(total))
    bar = "#" * completed + " " * (length - completed)
    print(f"\r{prefix} |{bar}| {pct:6.2f}% {suffix}", end="\r")
    if current >= total:
        print()


def parse_length(value, dpi=96.0):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    try:
        # plain number
        return float(s)
    except ValueError:
        pass
    # Unit parsing
    units = [
        ('px', 1.0),
        ('in', dpi),
        ('cm', dpi / 2.54),
        ('mm', dpi / 25.4),
        ('pt', dpi / 72.0),
        ('pc', dpi / 6.0),
        ('%', None)  # handled separately
    ]
    for u, factor in units:
        if s.endswith(u):
            num = s[:-len(u)].strip()
            try:
                f = float(num)
            except ValueError:
                return None
            if u == '%':
                return f / 100.0
            return f * factor
    # Unknown unit, try to strip non-numeric
    filtered = ''.join(c for c in s if c.isdigit() or c in '.-+eE')
    try:
        return float(filtered)
    except ValueError:
        return None


def parse_viewbox(vb):
    if not vb:
        return None
    try:
        parts = re.split(r'[,\s]+', vb.strip())
        if len(parts) != 4:
            return None
        return tuple(float(p) for p in parts)
    except Exception:
        return None


def parse_style_attr(style_str):
    d = {}
    if not style_str:
        return d
    for part in style_str.split(';'):
        if not part.strip():
            continue
        if ':' in part:
            k, v = part.split(':', 1)
            d[k.strip()] = v.strip()
    return d


def make_style_attr(d):
    if not d:
        return None
    return ';'.join(f'{k}:{v}' for k, v in d.items())


def local_name(tag):
    return tag.split('}')[-1]


# ----------------------------
# 2D Affine transforms (for SVG "transform" attributes)
# ----------------------------

def mat_identity():
    return [1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0]


def mat_mul(a, b):
    # 3x3 matrix multiply (row-major)
    out = [0.0]*9
    out[0] = a[0]*b[0] + a[1]*b[3] + a[2]*b[6]
    out[1] = a[0]*b[1] + a[1]*b[4] + a[2]*b[7]
    out[2] = a[0]*b[2] + a[1]*b[5] + a[2]*b[8]

    out[3] = a[3]*b[0] + a[4]*b[3] + a[5]*b[6]
    out[4] = a[3]*b[1] + a[4]*b[4] + a[5]*b[7]
    out[5] = a[3]*b[2] + a[4]*b[5] + a[5]*b[8]

    out[6] = a[6]*b[0] + a[7]*b[3] + a[8]*b[6]
    out[7] = a[6]*b[1] + a[7]*b[4] + a[8]*b[7]
    out[8] = a[6]*b[2] + a[7]*b[5] + a[8]*b[8]
    return out


def apply_mat(m, x, y):
    nx = m[0]*x + m[1]*y + m[2]
    ny = m[3]*x + m[4]*y + m[5]
    return nx, ny


def parse_transform_attr(tr):
    # Supports translate, scale, rotate, skewX, skewY, matrix
    if not tr:
        return mat_identity()
    m = mat_identity()
    token_re = re.compile(r'(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)')
    for func, args in token_re.findall(tr):
        nums = []
        if args.strip():
            nums = [float(v) for v in re.split(r'[,\s]+', args.strip()) if v]
        if func == 'matrix' and len(nums) == 6:
            a, b, c, d, e, f = nums
            m2 = [a, c, e,
                  b, d, f,
                  0, 0, 1]
        elif func == 'translate':
            tx = nums[0] if len(nums) > 0 else 0.0
            ty = nums[1] if len(nums) > 1 else 0.0
            m2 = [1, 0, tx,
                  0, 1, ty,
                  0, 0, 1]
        elif func == 'scale':
            sx = nums[0] if len(nums) > 0 else 1.0
            sy = nums[1] if len(nums) > 1 else sx
            m2 = [sx, 0, 0,
                  0, sy, 0,
                  0, 0, 1]
        elif func == 'rotate':
            ang = math.radians(nums[0] if nums else 0.0)
            cos_a = math.cos(ang)
            sin_a = math.sin(ang)
            if len(nums) >= 3:
                cx, cy = nums[1], nums[2]
                # translate(-cx, -cy) * rotate * translate(cx, cy)
                m2 = [cos_a, -sin_a, cx - cos_a*cx + sin_a*cy,
                      sin_a,  cos_a, cy - sin_a*cx - cos_a*cy,
                      0,      0,     1]
            else:
                m2 = [cos_a, -sin_a, 0,
                      sin_a,  cos_a, 0,
                      0,      0,     1]
        elif func == 'skewX':
            ang = math.radians(nums[0] if nums else 0.0)
            m2 = [1, math.tan(ang), 0,
                  0, 1,            0,
                  0, 0,            1]
        elif func == 'skewY':
            ang = math.radians(nums[0] if nums else 0.0)
            m2 = [1, 0,            0,
                  math.tan(ang), 1, 0,
                  0, 0,            1]
        else:
            m2 = mat_identity()
        m = mat_mul(m, m2)
    return m


# ----------------------------
# Sampling primitives
# ----------------------------

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def sample_line(x1, y1, x2, y2, samples_per_unit=1.0, min_samples=2, max_samples=200):
    length = math.hypot(x2 - x1, y2 - y1)
    n = max(min_samples, min(max_samples, int(length * samples_per_unit)))
    if n <= 1:
        return [(x1, y1), (x2, y2)]
    pts = []
    for i in range(n + 1):
        t = i / n
        pts.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    return pts


def sample_cubic_bezier(p0, p1, p2, p3, samples_per_unit=1.0, min_samples=8, max_samples=200):
    def bezier(t, a, b, c, d):
        mt = 1 - t
        return (mt**3)*a + 3*(mt**2)*t*b + 3*mt*(t**2)*c + (t**3)*d

    # rough length estimate
    chord = dist(p0, p3)
    polylen = dist(p0, p1) + dist(p1, p2) + dist(p2, p3)
    est_len = (chord + polylen) / 2.0
    n = max(min_samples, min(max_samples, int(est_len * samples_per_unit)))
    pts = []
    for i in range(n + 1):
        t = i / n
        x = bezier(t, p0[0], p1[0], p2[0], p3[0])
        y = bezier(t, p0[1], p1[1], p2[1], p3[1])
        pts.append((x, y))
    return pts


def sample_quadratic_bezier(p0, p1, p2, samples_per_unit=1.0, min_samples=6, max_samples=200):
    def bezier(t, a, b, c):
        mt = 1 - t
        return (mt*mt)*a + 2*mt*t*b + (t*t)*c

    chord = dist(p0, p2)
    polylen = dist(p0, p1) + dist(p1, p2)
    est_len = (chord + polylen) / 2.0
    n = max(min_samples, min(max_samples, int(est_len * samples_per_unit)))
    pts = []
    for i in range(n + 1):
        t = i / n
        x = bezier(t, p0[0], p1[0], p2[0])
        y = bezier(t, p0[1], p1[1], p2[1])
        pts.append((x, y))
    return pts


def sample_arc(p0, rx, ry, x_axis_rotation, large_arc_flag, sweep_flag, p1, samples_per_unit=1.0, min_samples=8, max_samples=300):
    # Based on SVG spec: convert endpoint arc to center parameterization and sample by angle
    # Handle degenerate radii
    if rx == 0 or ry == 0:
        return [p0, p1]

    phi = math.radians(x_axis_rotation % 360.0)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    # Step 1: Compute (x1', y1')
    dx2 = (p0[0] - p1[0]) / 2.0
    dy2 = (p0[1] - p1[1]) / 2.0
    x1p = cos_phi * dx2 + sin_phi * dy2
    y1p = -sin_phi * dx2 + cos_phi * dy2

    # Correct radii
    rx_abs = abs(rx)
    ry_abs = abs(ry)
    lam = (x1p**2) / (rx_abs**2) + (y1p**2) / (ry_abs**2)
    if lam > 1:
        s = math.sqrt(lam)
        rx_abs *= s
        ry_abs *= s

    # Step 2: Compute (cx', cy')
    sign = -1 if large_arc_flag == sweep_flag else 1
    num = (rx_abs**2) * (ry_abs**2) - (rx_abs**2) * (y1p**2) - (ry_abs**2) * (x1p**2)
    den = (rx_abs**2) * (y1p**2) + (ry_abs**2) * (x1p**2)
    if den == 0:
        den = 1e-12
    coef = sign * math.sqrt(max(0.0, num / den))
    cxp = coef * (rx_abs * y1p) / ry_abs
    cyp = coef * (-ry_abs * x1p) / rx_abs

    # Step 3: Compute (cx, cy)
    cx = cos_phi * cxp - sin_phi * cyp + (p0[0] + p1[0]) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (p0[1] + p1[1]) / 2.0

    # Step 4: Compute angles
    def angle(u, v):
        dot = u[0]*v[0] + u[1]*v[1]
        det = u[0]*v[1] - u[1]*v[0]
        ang = math.atan2(det, dot)
        return ang

    ux = (x1p - cxp) / rx_abs
    uy = (y1p - cyp) / ry_abs
    vx = (-x1p - cxp) / rx_abs
    vy = (-y1p - cyp) / ry_abs

    theta1 = angle((1, 0), (ux, uy))
    delta = angle((ux, uy), (vx, vy))
    if not sweep_flag and delta > 0:
        delta -= 2 * math.pi
    elif sweep_flag and delta < 0:
        delta += 2 * math.pi

    # Estimate arc length ~ average radius * angle
    avg_r = (rx_abs + ry_abs) / 2.0
    est_len = abs(delta) * avg_r
    n = max(min_samples, min(max_samples, int(est_len * samples_per_unit)))

    pts = []
    for i in range(n + 1):
        t = i / n
        ang = theta1 + t * delta
        cos_a = math.cos(ang)
        sin_a = math.sin(ang)
        x = cos_phi * (rx_abs * cos_a) - sin_phi * (ry_abs * sin_a) + cx
        y = sin_phi * (rx_abs * cos_a) + cos_phi * (ry_abs * sin_a) + cy
        pts.append((x, y))
    return pts


# ----------------------------
# Path parsing to segments
# ----------------------------

def tokenize_path(d):
    # Returns list of (cmd, [floats...])
    if not d:
        return []
    tokens = []
    pattern = re.compile(r'([MmLlHhVvCcSsQqTtAaZz])|([-+]?\d*\.?\d*(?:[eE][-+]?\d+)?)')
    parts = pattern.findall(d)
    flat = []
    for cmd, num in parts:
        if cmd:
            flat.append(cmd)
        elif num and num not in ['+', '-', '.', '+.', '-.']:
            try:
                flat.append(float(num))
            except ValueError:
                pass
    # Now iterate
    i = 0
    while i < len(flat):
        if isinstance(flat[i], str):
            cmd = flat[i]
            i += 1
        else:
            # implicit repeat of previous command
            if not tokens:
                # malformed path
                break
            cmd = tokens[-1][0]
        # number of parameters per segment for each command
        if cmd in 'Mm':
            n = 2
            coords = []
            while i < len(flat) and not isinstance(flat[i], str):
                coords.append(flat[i]); i += 1
            tokens.append((cmd, coords))
        elif cmd in 'Ll':
            coords = []
            while i < len(flat) and not isinstance(flat[i], str):
                coords.append(flat[i]); i += 1
            tokens.append((cmd, coords))
        elif cmd in 'HhVv':
            coords = []
            while i < len(flat) and not isinstance(flat[i], str):
                coords.append(flat[i]); i += 1
            tokens.append((cmd, coords))
        elif cmd in 'Cc':
            coords = []
            while i < len(flat) and not isinstance(flat[i], str):
                coords.append(flat[i]); i += 1
            tokens.append((cmd, coords))
        elif cmd in 'SsQqTt':
            coords = []
            while i < len(flat) and not isinstance(flat[i], str):
                coords.append(flat[i]); i += 1
            tokens.append((cmd, coords))
        elif cmd in 'Aa':
            coords = []
            while i < len(flat) and not isinstance(flat[i], str):
                coords.append(flat[i]); i += 1
            tokens.append((cmd, coords))
        elif cmd in 'Zz':
            tokens.append((cmd, []))
        else:
            # unknown
            i += 1
    return tokens


def path_to_subpaths(d_attr, samples_per_unit=1.0):
    tokens = tokenize_path(d_attr)
    subpaths = []  # list of (points, closed)
    cur = (0.0, 0.0)
    start = (0.0, 0.0)
    last_cubic_cp = None
    last_quad_cp = None
    current_points = []
    closed_flag_for_current = False

    def flush():
        nonlocal current_points, closed_flag_for_current
        if len(current_points) >= 2:
            subpaths.append((current_points, closed_flag_for_current))
        current_points = []
        closed_flag_for_current = False

    i = 0
    while i < len(tokens):
        cmd, coords = tokens[i]
        i += 1
        if cmd == 'M' or cmd == 'm':
            flush()
            last_cubic_cp = None
            last_quad_cp = None
            idx = 0
            while idx + 1 < len(coords):
                x = coords[idx]; y = coords[idx + 1]
                idx += 2
                if cmd == 'm':
                    cur = (cur[0] + x, cur[1] + y)
                else:
                    cur = (x, y)
                start = cur
                current_points.append(cur)
                # If more pairs follow, they are treated as implicit L
                cmd = 'L' if cmd == 'M' else 'l'
                coords = coords[idx:]
                idx = 0
                if not coords:
                    break
            last_cubic_cp = None
            last_quad_cp = None
        elif cmd == 'L' or cmd == 'l':
            idx = 0
            while idx + 1 < len(coords):
                x = coords[idx]; y = coords[idx + 1]
                idx += 2
                p1 = (cur[0] + x, cur[1] + y) if cmd == 'l' else (x, y)
                seg = sample_line(cur[0], cur[1], p1[0], p1[1], samples_per_unit)
                if current_points:
                    current_points.extend(seg[1:])
                else:
                    current_points.extend(seg)
                cur = p1
            last_cubic_cp = None
            last_quad_cp = None
        elif cmd == 'H' or cmd == 'h':
            for v in coords:
                x = cur[0] + v if cmd == 'h' else v
                p1 = (x, cur[1])
                seg = sample_line(cur[0], cur[1], p1[0], p1[1], samples_per_unit)
                if current_points:
                    current_points.extend(seg[1:])
                else:
                    current_points.extend(seg)
                cur = p1
            last_cubic_cp = None
            last_quad_cp = None
        elif cmd == 'V' or cmd == 'v':
            for v in coords:
                y = cur[1] + v if cmd == 'v' else v
                p1 = (cur[0], y)
                seg = sample_line(cur[0], cur[1], p1[0], p1[1], samples_per_unit)
                if current_points:
                    current_points.extend(seg[1:])
                else:
                    current_points.extend(seg)
                cur = p1
            last_cubic_cp = None
            last_quad_cp = None
        elif cmd == 'C' or cmd == 'c':
            idx = 0
            while idx + 5 < len(coords):
                x1, y1, x2, y2, x, y = coords[idx:idx+6]
                idx += 6
                cp1 = (cur[0] + x1, cur[1] + y1) if cmd == 'c' else (x1, y1)
                cp2 = (cur[0] + x2, cur[1] + y2) if cmd == 'c' else (x2, y2)
                p1 = (cur[0] + x, cur[1] + y) if cmd == 'c' else (x, y)
                pts = sample_cubic_bezier(cur, cp1, cp2, p1, samples_per_unit)
                if current_points:
                    current_points.extend(pts[1:])
                else:
                    current_points.extend(pts)
                cur = p1
                last_cubic_cp = cp2
                last_quad_cp = None
        elif cmd == 'S' or cmd == 's':
            idx = 0
            while idx + 3 < len(coords):
                x2, y2, x, y = coords[idx:idx+4]
                idx += 4
                cp2 = (cur[0] + x2, cur[1] + y2) if cmd == 's' else (x2, y2)
                if last_cubic_cp is not None:
                    cp1 = (2*cur[0] - last_cubic_cp[0], 2*cur[1] - last_cubic_cp[1])
                else:
                    cp1 = cur
                p1 = (cur[0] + x, cur[1] + y) if cmd == 's' else (x, y)
                pts = sample_cubic_bezier(cur, cp1, cp2, p1, samples_per_unit)
                if current_points:
                    current_points.extend(pts[1:])
                else:
                    current_points.extend(pts)
                cur = p1
                last_cubic_cp = cp2
                last_quad_cp = None
        elif cmd == 'Q' or cmd == 'q':
            idx = 0
            while idx + 3 < len(coords):
                x1, y1, x, y = coords[idx:idx+4]
                idx += 4
                cp = (cur[0] + x1, cur[1] + y1) if cmd == 'q' else (x1, y1)
                p1 = (cur[0] + x, cur[1] + y) if cmd == 'q' else (x, y)
                pts = sample_quadratic_bezier(cur, cp, p1, samples_per_unit)
                if current_points:
                    current_points.extend(pts[1:])
                else:
                    current_points.extend(pts)
                cur = p1
                last_quad_cp = cp
                last_cubic_cp = None
        elif cmd == 'T' or cmd == 't':
            idx = 0
            while idx + 1 < len(coords):
                x, y = coords[idx:idx+2]
                idx += 2
                if last_quad_cp is not None:
                    cp = (2*cur[0] - last_quad_cp[0], 2*cur[1] - last_quad_cp[1])
                else:
                    cp = cur
                p1 = (cur[0] + x, cur[1] + y) if cmd == 't' else (x, y)
                pts = sample_quadratic_bezier(cur, cp, p1, samples_per_unit)
                if current_points:
                    current_points.extend(pts[1:])
                else:
                    current_points.extend(pts)
                cur = p1
                last_quad_cp = cp
                last_cubic_cp = None
        elif cmd == 'A' or cmd == 'a':
            idx = 0
            while idx + 6 < len(coords):
                rx, ry, xrot, large_flag, sweep_flag, x, y = coords[idx:idx+7]
                idx += 7
                p1 = (cur[0] + x, cur[1] + y) if cmd == 'a' else (x, y)
                pts = sample_arc(cur, rx, ry, xrot, int(large_flag) != 0, int(sweep_flag) != 0, p1, samples_per_unit)
                if current_points:
                    current_points.extend(pts[1:])
                else:
                    current_points.extend(pts)
                cur = p1
                last_cubic_cp = None
                last_quad_cp = None
        elif cmd == 'Z' or cmd == 'z':
            # close path
            if current_points and (cur[0] != start[0] or cur[1] != start[1]):
                seg = sample_line(cur[0], cur[1], start[0], start[1], samples_per_unit)
                current_points.extend(seg[1:])
            cur = start
            closed_flag_for_current = True
            flush()
            last_cubic_cp = None
            last_quad_cp = None
        else:
            # ignore unknown
            pass

    if current_points:
        flush()

    return subpaths


def parse_points_attribute(s):
    pts = []
    if not s:
        return pts
    # points can be "x1,y1 x2,y2 ..." or "x1 y1 x2 y2 ..."
    nums = [float(v) for v in re.split(r'[,\s]+', s.strip()) if v]
    for i in range(0, len(nums) - 1, 2):
        pts.append((nums[i], nums[i+1]))
    return pts


# ----------------------------
# Camera and spherical projection
# ----------------------------

def rotation_matrix_yaw_pitch_roll(yaw_deg, pitch_deg, roll_deg):
    ya = math.radians(yaw_deg)
    pa = math.radians(pitch_deg)
    ra = math.radians(roll_deg)
    cy, sy = math.cos(ya), math.sin(ya)
    cp, sp = math.cos(pa), math.sin(pa)
    cr, sr = math.cos(ra), math.sin(ra)

    # R = Rz(roll) * Rx(pitch) * Ry(yaw)
    Rz = [
        cr, -sr, 0.0,
        sr,  cr, 0.0,
        0.0, 0.0, 1.0
    ]
    Rx = [
        1.0, 0.0, 0.0,
        0.0,  cp, -sp,
        0.0,  sp,  cp
    ]
    Ry = [
        cy, 0.0, sy,
        0.0, 1.0, 0.0,
        -sy, 0.0, cy
    ]

    def mul3(a, b):
        o = [0.0]*9
        o[0] = a[0]*b[0] + a[1]*b[3] + a[2]*b[6]
        o[1] = a[0]*b[1] + a[1]*b[4] + a[2]*b[7]
        o[2] = a[0]*b[2] + a[1]*b[5] + a[2]*b[8]
        o[3] = a[3]*b[0] + a[4]*b[3] + a[5]*b[6]
        o[4] = a[3]*b[1] + a[4]*b[4] + a[5]*b[7]
        o[5] = a[3]*b[2] + a[4]*b[5] + a[5]*b[8]
        o[6] = a[6]*b[0] + a[7]*b[3] + a[8]*b[6]
        o[7] = a[6]*b[1] + a[7]*b[4] + a[8]*b[7]
        o[8] = a[6]*b[2] + a[7]*b[5] + a[8]*b[8]
        return o

    R = mul3(Rz, mul3(Rx, Ry))
    return R


def apply_R(R, v):
    x = R[0]*v[0] + R[1]*v[1] + R[2]*v[2]
    y = R[3]*v[0] + R[4]*v[1] + R[5]*v[2]
    z = R[6]*v[0] + R[7]*v[1] + R[8]*v[2]
    return (x, y, z)


def apply_R_T(R, v):
    # Transpose of R (inverse rotation)
    x = R[0]*v[0] + R[3]*v[1] + R[6]*v[2]
    y = R[1]*v[0] + R[4]*v[1] + R[7]*v[2]
    z = R[2]*v[0] + R[5]*v[1] + R[8]*v[2]
    return (x, y, z)


def equirect_to_sphere(x, y, width_units, height_units):
    # x in [0, W), y in [0, H), y=0 is top (lat=+pi/2)
    lon = (x / width_units) * 2.0 * math.pi - math.pi
    lat = math.pi/2.0 - (y / height_units) * math.pi
    clat = math.cos(lat)
    return (clat * math.cos(lon), math.sin(lat), clat * math.sin(lon))


def sphere_to_equirect(v, width_units, height_units):
    x, y, z = v
    lon = math.atan2(z, x)
    lat = math.asin(max(-1.0, min(1.0, y)))
    xx = (lon + math.pi) / (2.0 * math.pi) * width_units
    yy = (math.pi/2.0 - lat) / math.pi * height_units
    return (xx, yy)


def stereo_project_from_south(v):
    # Project from south pole onto plane tangent at north pole (y=1)
    x, y, z = v
    denom = 1.0 + y
    if denom < 1e-9:
        denom = 1e-9
    u = 2.0 * x / denom
    w = 2.0 * z / denom
    return (u, w)


def stereo_unproject_to_sphere(u, w):
    # Inverse of above
    R2 = u*u + w*w
    x = 4.0 * u / (R2 + 4.0)
    y = (4.0 - R2) / (R2 + 4.0)
    z = 4.0 * w / (R2 + 4.0)
    return (x, y, z)


def forward_transform_point(px, py, width_units, height_units, center_x, center_y, base_scale, R):
    # From panorama pixel (px,py) to stereographic plane pixel
    v = equirect_to_sphere(px, py, width_units, height_units)
    v_rot = apply_R(R, v)
    u, w = stereo_project_from_south(v_rot)
    X = center_x + base_scale * u
    Y = center_y + base_scale * w
    return (X, Y)


def inverse_transform_point(px, py, width_units, height_units, center_x, center_y, base_scale, R):
    # From stereographic plane pixel (px,py) to panorama pixel
    u = (px - center_x) / base_scale
    w = (py - center_y) / base_scale
    v_plane = stereo_unproject_to_sphere(u, w)
    v_world = apply_R_T(R, v_plane)
    ex, ey = sphere_to_equirect(v_world, width_units, height_units)
    return (ex, ey)


# ----------------------------
# SVG traversal and processing
# ----------------------------

def get_svg():
    print("Open SVG", end="\r")
    root = tkinter.Tk()
    root.withdraw()
    svg_path = filedialog.askopenfilename(filetypes=[("SVG", ".svg")])
    root.destroy()
    if not svg_path:
        print("No file selected.")
        sys.exit(1)
    svg_name = os.path.basename(svg_path)
    return svg_path, svg_name


def gather_drawables_with_context(element, inherited_transform=None, inherited_style=None, path=None):
    if inherited_transform is None:
        inherited_transform = mat_identity()
    if inherited_style is None:
        inherited_style = {}
    if path is None:
        path = []

    items = []

    tag = local_name(element.tag)
    # Merge transforms
    elem_transform = parse_transform_attr(element.get('transform'))
    combined_transform = mat_mul(inherited_transform, elem_transform)

    # Merge styles
    style_dict = dict(inherited_style)
    # Style attribute
    elem_style_dict = parse_style_attr(element.get('style'))
    style_dict.update(elem_style_dict)
    # Presentation attributes override
    for k in ['stroke', 'fill', 'stroke-width', 'stroke-linecap', 'stroke-linejoin', 'stroke-dasharray', 'stroke-dashoffset', 'opacity', 'fill-opacity', 'stroke-opacity']:
        if element.get(k) is not None:
            style_dict[k] = element.get(k)

    # Recurse groups
    if tag == 'g' or tag == 'svg' or tag == 'symbol':
        for child in element:
            items.extend(gather_drawables_with_context(child, combined_transform, style_dict, path + [element]))
        return items

    # Drawable elements
    if tag in ('line', 'path', 'polyline', 'polygon'):
        items.append((element, tag, combined_transform, style_dict))
    else:
        # pass through others (e.g., defs) but still dive
        for child in element:
            items.extend(gather_drawables_with_context(child, combined_transform, style_dict, path + [element]))
    return items


def build_style_attributes(style_dict):
    # We use a single 'style' attribute composed of style_dict
    style_str = make_style_attr(style_dict)
    out = {}
    if style_str:
        out['style'] = style_str
    # Also set presentation attributes if provided explicitly in style_dict,
    # to increase compatibility without overriding 'style' precedence
    for k in ['stroke', 'fill', 'stroke-width', 'stroke-linecap', 'stroke-linejoin', 'stroke-dasharray', 'stroke-dashoffset', 'opacity', 'fill-opacity', 'stroke-opacity']:
        if k in style_dict:
            out[k] = style_dict[k]
    return out


def convert_svg(input_path, pan_deg=0.0, tilt_deg=0.0, spin_deg=0.0, zoom=1.0, inverse=False, samples_per_unit=2.0, output_filename=None):
    t0 = time.time()
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Dimensions and viewBox
    in_width = parse_length(root.get('width'))
    in_height = parse_length(root.get('height'))
    viewBox = parse_viewbox(root.get('viewBox'))
    if viewBox:
        vb_x, vb_y, vb_w, vb_h = viewBox
        units_width = vb_w
        units_height = vb_h
    else:
        if in_width is None or in_height is None:
            # fallback: 1000x500 default panorama-like proportion
            in_width = 1000.0
            in_height = 500.0
        units_width = in_width
        units_height = in_height

    # Output root
    new_root = ET.Element('svg', xmlns='http://www.w3.org/2000/svg')
    if root.get('xmlns'):
        new_root.set('xmlns', root.get('xmlns'))
    # Preserve width/height and viewBox
    if in_width is not None:
        new_root.set('width', str(in_width))
    if in_height is not None:
        new_root.set('height', str(in_height))
    if viewBox:
        new_root.set('viewBox', root.get('viewBox'))

    # Determine projection center and scale in "units" space (viewBox if present)
    center_x = units_width / 2.0
    center_y = units_height / 2.0
    base_scale = min(units_width, units_height) * 0.4 * max(1e-6, zoom)

    # Rotation parameters:
    # If inverse mapping: pan = yaw (horizontal LOS movement), spin = roll
    # Else (forward): pan acts like spin (roll), spin also roll. yaw = 0
    yaw_deg = pan_deg if inverse else 0.0
    roll_deg = spin_deg + (pan_deg if not inverse else 0.0)
    pitch_deg = tilt_deg

    R = rotation_matrix_yaw_pitch_roll(yaw_deg, pitch_deg, roll_deg)

    # Gather drawables
    items = gather_drawables_with_context(root)
    total = len(items)

    # Iterate and convert
    for idx, (elem, tag, transform2d, style_dict) in enumerate(items, start=1):
        progress_update(idx, total, "Converting elements")

        # Geometry extraction and sampling to list of subpaths: [(points, closed), ...]
        subpaths = []

        if tag == 'line':
            x1 = parse_length(elem.get('x1')) or 0.0
            y1 = parse_length(elem.get('y1')) or 0.0
            x2 = parse_length(elem.get('x2')) or 0.0
            y2 = parse_length(elem.get('y2')) or 0.0
            p0 = apply_mat(transform2d, x1, y1)
            p1 = apply_mat(transform2d, x2, y2)
            pts = sample_line(p0[0], p0[1], p1[0], p1[1], samples_per_unit)
            subpaths.append((pts, False))

        elif tag == 'polyline':
            pts_raw = parse_points_attribute(elem.get('points'))
            pts = [apply_mat(transform2d, x, y) for (x, y) in pts_raw]
            # Further sample consecutive segments
            if len(pts) >= 2:
                sampled = [pts[0]]
                for a, b in zip(pts, pts[1:]):
                    seg = sample_line(a[0], a[1], b[0], b[1], samples_per_unit)
                    sampled.extend(seg[1:])
                subpaths.append((sampled, False))

        elif tag == 'polygon':
            pts_raw = parse_points_attribute(elem.get('points'))
            pts = [apply_mat(transform2d, x, y) for (x, y) in pts_raw]
            # Close polygon
            if len(pts) >= 2:
                sampled = [pts[0]]
                for a, b in zip(pts, pts[1:]):
                    seg = sample_line(a[0], a[1], b[0], b[1], samples_per_unit)
                    sampled.extend(seg[1:])
                # close
                seg = sample_line(pts[-1][0], pts[-1][1], pts[0][0], pts[0][1], samples_per_unit)
                sampled.extend(seg[1:])
                subpaths.append((sampled, True))

        elif tag == 'path':
            d = elem.get('d') or ''
            subs = path_to_subpaths(d, samples_per_unit)
            # Apply transform to each point
            for pts, closed in subs:
                if not pts:
                    continue
                tr_pts = [apply_mat(transform2d, x, y) for (x, y) in pts]
                subpaths.append((tr_pts, closed))

        # Transform points through stereographic mapping
        for pts, closed in subpaths:
            if len(pts) < 2:
                continue
            transformed = []
            if not inverse:
                for (x, y) in pts:
                    nx, ny = forward_transform_point(x, y, units_width, units_height, center_x, center_y, base_scale, R)
                    transformed.append((nx, ny))
            else:
                for (x, y) in pts:
                    nx, ny = inverse_transform_point(x, y, units_width, units_height, center_x, center_y, base_scale, R)
                    transformed.append((nx, ny))

            # Create output element: polyline or polygon depending on closure
            if closed and len(transformed) >= 3:
                out_elem = ET.SubElement(new_root, 'polygon')
                points_str = ' '.join(f"{p[0]:.3f},{p[1]:.3f}" for p in transformed)
                out_elem.set('points', points_str)
            else:
                out_elem = ET.SubElement(new_root, 'polyline')
                points_str = ' '.join(f"{p[0]:.3f},{p[1]:.3f}" for p in transformed)
                out_elem.set('points', points_str)

            # Style
            style_attrs = build_style_attributes(style_dict)
            for k, v in style_attrs.items():
                out_elem.set(k, v)

    # Copy over defs (gradients etc.) to preserve styling references if any
    # Also copy <defs> children directly, without geometry transforms
    for child in root:
        if local_name(child.tag) == 'defs':
            new_root.append(child)

    # Save
    out_tree = ET.ElementTree(new_root)
    if not output_filename:
        base = os.path.basename(input_path)
        name, ext = os.path.splitext(base)
        suffix = '_inv' if inverse else '_stereo'
        output_filename = f"new_{name}{suffix}.svg"
    out_tree.write(output_filename, encoding='utf-8', xml_declaration=True)

    print(f"\nDone. Wrote: {output_filename}  (time: {time.time() - t0:.3f}s)")
    return output_filename


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Convert SVG lines/paths using stereographic projection with camera controls',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog='''
Parameters:
  Pan    Horizontal camera panning.
         If --inverse is set, pan moves line of sight horizontally (yaw).
         Else, pan acts like Spin (roll).

  Tilt   Vertical camera panning (pitch). Moves line of sight up/down.

  Spin   Spin angle around camera axis (roll). Rotates planet around line of sight.

  Zoom   Zoom level. Zooms planet in/out.

  Inverse transform:
         If set, do the inverse mapping (stereographic plane -> panorama);
         useful for touching up zenith, nadir, or other parts of panorama.
''')
    parser.add_argument('input_file', nargs='?', help='Input SVG file path (if omitted, opens file dialog)')
    parser.add_argument('-p', '--pan', type=float, default=0.0, help='Pan (degrees). See description.')
    parser.add_argument('-t', '--tilt', type=float, default=0.0, help='Tilt (degrees)')
    parser.add_argument('-s', '--spin', type=float, default=-90.0, help='Spin (degrees), default -90')
    parser.add_argument('-z', '--zoom', type=float, default=0.5, help='Zoom (scale factor, default 0.5)')
    parser.add_argument('-i', '--inverse', action='store_true', help='Use inverse transform (stereo plane -> panorama)')
    parser.add_argument('--samples', type=float, default=2.0, help='Sampling density per unit length (default: 2.0)')
    parser.add_argument('-o', '--output', help='Output filename (default: new_[input]_[stereo|inv].svg)')

    args = parser.parse_args()

    if args.input_file:
        input_path = args.input_file
    else:
        input_path, _ = get_svg()

    if not os.path.isfile(input_path):
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    convert_svg(
        input_path=input_path,
        pan_deg=args.pan,
        tilt_deg=args.tilt,
        spin_deg=args.spin,
        zoom=args.zoom,
        inverse=args.inverse,
        samples_per_unit=args.samples,
        output_filename=args.output
    )


if __name__ == '__main__':
    main()
