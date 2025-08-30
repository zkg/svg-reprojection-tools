#!/usr/bin/env python3
"""
Reproject Blender Freestyle SVG cube-face line art to an equirectangular SVG panorama.

Dependencies:
  pip install svgelements numpy

Usage:
  python reproj_svg_to_equirect.py metadata.json output.svg --pano-width 8192 --pano-height 4096 --tol 0.5
"""

import argparse
import json
import math
import os
import sys
from typing import List, Tuple, Dict

import numpy as np
from svgelements import SVG, Path, Line, Polyline, Polygon, Move, Close

# -----------------------------
# Math / geometry utilities
# -----------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def mat_from_rowmajor(rows: List[List[float]]) -> np.ndarray:
    return np.array(rows, dtype=float)


def point_to_segment_distance(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    abx, aby = (bx - ax), (by - ay)
    apx, apy = (px - ax), (py - ay)
    ab2 = abx * abx + aby * aby
    if ab2 == 0.0:
        return math.hypot(apx, apy)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    cx, cy = (ax + t * abx, ay + t * aby)
    return math.hypot(px - cx, py - cy)


def adjust_to_near(u: float, ref: float, W: float) -> float:
    """
    Shift u by integer multiples of W so it's as close as possible to ref (unwrap).
    """
    du = u - ref
    if du > W * 0.5:
        return u - W
    if du < -W * 0.5:
        return u + W
    return u


def unwrap_sequence(us_mod: List[float], W: float) -> List[float]:
    out = []
    for i, u in enumerate(us_mod):
        if i == 0:
            out.append(u)
        else:
            out.append(adjust_to_near(u, out[-1], W))
    return out


def split_polyline_on_seams(poly_unwrapped: List[Tuple[float, float]], W: float) -> List[List[Tuple[float, float]]]:
    """
    Given a polyline with continuous (unwrapped) u values, split it where it crosses seams u = k*W.
    Return chunks shifted (not modulo) into the correct [0, W] interval so that a seam endpoint stays at 0 or W.
    """
    if len(poly_unwrapped) < 2:
        return []

    out: List[List[Tuple[float, float]]] = []
    cur: List[Tuple[float, float]] = [poly_unwrapped[0]]
    u0, v0 = poly_unwrapped[0]

    for i in range(1, len(poly_unwrapped)):
        u1, v1 = poly_unwrapped[i]
        if u1 == u0:
            cur.append((u1, v1))
            u0, v0 = u1, v1
            continue

        u_lo, u_hi = (u0, u1) if u1 >= u0 else (u1, u0)
        k_start = math.floor(u_lo / W) + 1
        k_end = math.floor(u_hi / W)

        last_u, last_v = u0, v0
        for k in range(k_start, k_end + 1):
            seam_u = k * W
            t = (seam_u - last_u) / (u1 - last_u)
            v_seam = last_v + t * (v1 - last_v)

            # add intersection at the seam
            cur.append((seam_u, v_seam))

            # flush current chunk; shift it into [0, W] with a single offset
            min_u = min(u for (u, _) in cur)
            offset = math.floor(min_u / W) * W
            chunk = [(uu - offset, vv) for (uu, vv) in cur]
            # clamp minor FP drift
            chunk = [(min(max(uu, 0.0), W), vv) for (uu, vv) in chunk]
            out.append(chunk)

            # start new chunk from the seam point
            cur = [(seam_u, v_seam)]
            last_u, last_v = seam_u, v_seam

        # append the real endpoint to current chunk
        cur.append((u1, v1))
        u0, v0 = u1, v1

    if len(cur) >= 2:
        min_u = min(u for (u, _) in cur)
        offset = math.floor(min_u / W) * W
        chunk = [(uu - offset, vv) for (uu, vv) in cur]
        chunk = [(min(max(uu, 0.0), W), vv) for (uu, vv) in chunk]
        out.append(chunk)

    return out

# -----------------------------
# Camera and projection
# -----------------------------

class CameraModel:
    def __init__(self, fx: float, fy: float, cx: float, cy: float, R_world_from_cam: np.ndarray):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.R = R_world_from_cam

    def project_pixel_to_world_dir(self, x_px: float, y_px: float) -> np.ndarray:
        # Blender camera convention: +X right, +Y up, looks along -Z
        x_cam = (x_px - self.cx) / self.fx
        y_cam = -(y_px - self.cy) / self.fy
        z_cam = -1.0
        d_cam = normalize(np.array([x_cam, y_cam, z_cam], dtype=float))
        d_world = self.R @ d_cam
        return normalize(d_world)


def world_dir_to_equirect(d: np.ndarray, Wp: float, Hp: float) -> Tuple[float, float]:
    # lon: [-pi, pi], lat: [-pi/2, pi/2]
    lon = math.atan2(d[0], d[2])  # 0 at +Z, + towards +X
    lat = math.asin(max(-1.0, min(1.0, d[1])))
    u = (lon / (2.0 * math.pi) + 0.5) * Wp
    v = (0.5 - lat / math.pi) * Hp
    return (u % Wp, v)


# -----------------------------
# SVG parsing / flattening
# -----------------------------

def get_viewbox_transform(svg_doc: SVG, target_width_px: float, target_height_px: float):
    """
    Returns a function that maps document coordinates to pixel coordinates using the viewBox.
    If there is no viewBox, assume coordinates are already in pixels matching the target resolution.
    """
    vb = getattr(svg_doc, "viewbox", None)
    if vb is None:
        minx, miny, vbw, vbh = 0.0, 0.0, float(target_width_px), float(target_height_px)
    else:
        # svgelements.Viewbox has x, y, width, height
        try:
            minx, miny, vbw, vbh = float(vb.x), float(vb.y), float(vb.width), float(vb.height)
        except Exception:
            # If it's already a tuple/list
            minx, miny, vbw, vbh = vb

    sx = float(target_width_px) / float(vbw) if vbw != 0 else 1.0
    sy = float(target_height_px) / float(vbh) if vbh != 0 else 1.0

    def to_pixels(pt: complex) -> Tuple[float, float]:
        x = (pt.real - minx) * sx
        y = (pt.imag - miny) * sy
        return (x, y)

    return to_pixels


def segment_point(seg, t: float) -> complex:
    # Every segment type in svgelements supports .point(t) except Move
    return seg.point(t)


def project_from_complex(pt: complex, to_pixels_fn, project_fn):
    x_px, y_px = to_pixels_fn(pt)
    return project_fn((x_px, y_px))


def flatten_segment_projected(seg, project_fn, tol_px: float, pano_W: float, max_depth: int = 12) -> List[Tuple[float, float]]:
    """
    Recursively flatten a segment by evaluating midpoints and checking deviation in OUTPUT (equirect) space.
    Returns a list of (u_unwrapped, v) points (including endpoints).
    """
    p0 = segment_point(seg, 0.0)
    p1 = segment_point(seg, 1.0)
    u0m, v0 = project_fn(p0)  # modulo-W
    u1m, v1 = project_fn(p1)
    u0 = u0m
    u1 = adjust_to_near(u1m, u0, pano_W)

    pts: List[Tuple[float, float]] = [(u0, v0)]

    def recurse(t0, u0, v0, t1, u1, v1, depth):
        tm = 0.5 * (t0 + t1)
        pm = segment_point(seg, tm)
        um_mod, vm = project_fn(pm)
        um = adjust_to_near(um_mod, 0.5 * (u0 + u1), pano_W)

        dist = point_to_segment_distance((um, vm), (u0, v0), (u1, v1))
        if dist <= tol_px or depth >= max_depth:
            pts.append((u1, v1))
            return
        recurse(t0, u0, v0, tm, um, vm, depth + 1)
        recurse(tm, um, vm, t1, u1, v1, depth + 1)

    recurse(0.0, u0, v0, 1.0, u1, v1, 0)
    return pts  # u is unwrapped here


def element_to_polylines_projected(
    element,
    to_pixels_fn,
    project_fn,
    pano_W: float,
    tol_px: float,
    max_depth: int,
) -> List[List[Tuple[float, float]]]:
    """
    Convert an svgelements element to a list of projected polylines (each a list of (u, v) in [0,W) x [0,H)).
    Handles seam splitting robustly.
    """
    polylines: List[List[Tuple[float, float]]] = []

    def add_polyline_unwrapped(seq: List[Tuple[float, float]]):
        # seq may still have modulo-W u values or local unwrapping; make it globally continuous first.
        if not seq or len(seq) < 2:
            return
        us = [u for (u, v) in seq]
        vs = [v for (u, v) in seq]
        us_unwrapped = unwrap_sequence(us, pano_W)  # enforce continuity across the whole subpath
        uv_unwrapped = list(zip(us_unwrapped, vs))
        # Now split exactly at seams and wrap each chunk back to [0, W)
        for chunk in split_polyline_on_seams(uv_unwrapped, pano_W):
            if len(chunk) >= 2:
                polylines.append(chunk)

    # Convert all supported types to a Path to unify handling and transforms
    if isinstance(element, (Path, Line, Polyline, Polygon)):
        #path: Path = element.as_path()
        path = element if isinstance(element, Path) else element.as_path()
        subpath_pts: List[Tuple[float, float]] = []
        for seg in path.segments(transformed=True):
            if isinstance(seg, Move):
                # flush current subpath
                if len(subpath_pts) >= 2:
                    add_polyline_unwrapped(subpath_pts)
                subpath_pts = []
                continue
            # For any drawable segment (including Close), flatten
            projected_unwrapped = flatten_segment_projected(
                seg, lambda p: project_from_complex(p, to_pixels_fn, project_fn), tol_px, pano_W, max_depth
            )
            if not subpath_pts:
                subpath_pts = projected_unwrapped
            else:
                subpath_pts.extend(projected_unwrapped[1:])  # avoid duplicating the shared vertex
        if len(subpath_pts) >= 2:
            add_polyline_unwrapped(subpath_pts)

    return polylines


# -----------------------------
# Style utilities
# -----------------------------

def extract_style(element, stroke_scale: float = 1.0) -> Dict[str, str]:
    vals = getattr(element, "values", {}) or {}
    style: Dict[str, str] = {}
    stroke = vals.get("stroke", None)
    if stroke is None:
        try:
            s = element.stroke
            if s is not None and str(s) != "none":
                stroke = str(s)
        except Exception:
            pass
    if not stroke or str(stroke) == "none":
        stroke = "#000000"

    sw = vals.get("stroke-width", None)
    if sw is None:
        try:
            if element.stroke_width is not None:
                sw = float(element.stroke_width)
        except Exception:
            pass
    if sw is None:
        sw = 1.0
    else:
        try:
            sw = float(sw)
        except Exception:
            try:
                sw = float(str(sw).replace("px", ""))
            except Exception:
                sw = 1.0
    sw *= stroke_scale

    linecap = vals.get("stroke-linecap", "round")
    linejoin = vals.get("stroke-linejoin", "round")
    miterlimit = vals.get("stroke-miterlimit", "4")

    style["fill"] = "none"
    style["stroke"] = str(stroke)
    style["stroke-width"] = f"{sw:.6g}"
    style["stroke-linecap"] = str(linecap)
    style["stroke-linejoin"] = str(linejoin)
    style["stroke-miterlimit"] = str(miterlimit)
    return style


def style_key(style: Dict[str, str]) -> str:
    return "|".join([
        style.get("stroke", ""),
        style.get("stroke-width", ""),
        style.get("stroke-linecap", ""),
        style.get("stroke-linejoin", ""),
        style.get("stroke-miterlimit", ""),
    ])


# -----------------------------
# Output SVG writer
# -----------------------------

def write_output_svg(output_path: str, W: int, H: int, layers: List[Tuple[List[List[Tuple[float, float]]], Dict[str, str]]]):
    from xml.etree.ElementTree import Element, SubElement, ElementTree

    svg = Element("svg", attrib={
        "xmlns": "http://www.w3.org/2000/svg",
        "version": "1.1",
        "width": str(W),
        "height": str(H),
        "viewBox": f"0 0 {W} {H}",
    })

    for polylines, style in layers:
        if not polylines:
            continue
        g = SubElement(svg, "g", attrib={
            "fill": style.get("fill", "none"),
            "stroke": style.get("stroke", "#000"),
            "stroke-width": style.get("stroke-width", "1"),
            "stroke-linecap": style.get("stroke-linecap", "round"),
            "stroke-linejoin": style.get("stroke-linejoin", "round"),
            "stroke-miterlimit": style.get("stroke-miterlimit", "4"),
            "vector-effect": "non-scaling-stroke",
        })
        for poly in polylines:
            if len(poly) < 2:
                continue
            d = []
            u0, v0 = poly[0]
            d.append(f"M {u0:.4f},{v0:.4f}")
            for (u, v) in poly[1:]:
                d.append(f"L {u:.4f},{v:.4f}")
            SubElement(g, "path", attrib={"d": " ".join(d)})

    tree = ElementTree(svg)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


# -----------------------------
# Main processing
# -----------------------------

def process(metadata_path: str, output_svg: str, pano_W: int, pano_H: int, tol_px: float, max_depth: int, stroke_scale: float):
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    faces = meta.get("faces", [])
    if not faces:
        print("No faces found in metadata.", file=sys.stderr)
        sys.exit(1)

    # Intrinsics (assumed constant across faces)
    intr = meta.get("intrinsics", faces[0].get("intrinsics", {}))
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    src_W = int(intr["width"])
    src_H = int(intr["height"])

    style_groups: Dict[str, Tuple[List[List[Tuple[float, float]]], Dict[str, str]]] = {}

    faces_processed = 0
    input_paths = 0
    output_polylines = 0

    for face in faces:
        svg_path = face.get("expected_svg") or face.get("output_base", "") + ".svg"
        if not svg_path or not os.path.isfile(svg_path):
            print(f"Warning: SVG not found for face {face.get('name')}: {svg_path}", file=sys.stderr)
            continue

        R_rows = face["R_world_from_cam_3x3_rowmajor"]
        R = mat_from_rowmajor(R_rows)
        cam = CameraModel(fx=fx, fy=fy, cx=cx, cy=cy, R_world_from_cam=R)

        def project_fn_from_pixels(pxy: Tuple[float, float]) -> Tuple[float, float]:
            x_px, y_px = pxy
            d_world = cam.project_pixel_to_world_dir(x_px, y_px)
            return world_dir_to_equirect(d_world, pano_W, pano_H)

        # Parse SVG (must use parse() to load file)
        svg_doc = SVG.parse(svg_path)
        to_pixels_fn = get_viewbox_transform(svg_doc, src_W, src_H)

        # Collect elements with strokes
        for elem in svg_doc.elements():
            if not isinstance(elem, (Path, Line, Polyline, Polygon)):
                continue

            style = extract_style(elem, stroke_scale=stroke_scale)
            polylines = element_to_polylines_projected(
                elem, to_pixels_fn, project_fn_from_pixels, pano_W, tol_px, max_depth
            )
            if not polylines:
                continue

            key = style_key(style)
            if key not in style_groups:
                style_groups[key] = ([], style)
            style_groups[key][0].extend(polylines)
            input_paths += 1
            output_polylines += len(polylines)

        faces_processed += 1

    layers = list(style_groups.values())
    write_output_svg(output_svg, pano_W, pano_H, layers)

    print(f"Done. Faces processed: {faces_processed}")
    print(f"Input paths: {input_paths}, output polylines: {output_polylines}")
    print(f"Output SVG: {output_svg}")


def main():
    ap = argparse.ArgumentParser(description="Reproject Freestyle SVG cube-face line art to an equirectangular SVG panorama.")
    ap.add_argument("metadata", help="Path to JSON metadata exported from Blender script")
    ap.add_argument("output_svg", help="Output SVG path")
    ap.add_argument("--pano-width", type=int, default=8192, help="Output panorama width in pixels")
    ap.add_argument("--pano-height", type=int, default=4096, help="Output panorama height in pixels")
    ap.add_argument("--tol", type=float, default=0.5, help="Adaptive flattening tolerance in OUTPUT pixels")
    ap.add_argument("--max-depth", type=int, default=12, help="Max recursion depth for flattening")
    ap.add_argument("--stroke-scale", type=float, default=1.0, help="Multiply all stroke-widths by this factor")
    args = ap.parse_args()

    process(args.metadata, args.output_svg, args.pano_width, args.pano_height, args.tol, args.max_depth, args.stroke_scale)


if __name__ == "__main__":
    main()
