# in your terminal, run the following:
# "pip install streamlit cadquery streamlit-stl numpy scipy scikit-image trimesh"
# next "cd (folder where interface file is saved)" 
# next "streamlit run Interface.py" to start the UI
# "cntrl + c" to stop Interface UI 

import streamlit as st
import cadquery as cq
import random
import math
import os
from streamlit_stl import stl_from_file
import zipfile
import io

import numpy as np
import scipy.ndimage as ndi
from skimage import measure
import trimesh

# 1. HOLLOW GEOMETRY GENERATOR
def generate_spaced_angles(n_points, min_gap_degrees=12.0):
    min_gap_rad = math.radians(min_gap_degrees)
    if n_points * min_gap_rad >= 2 * math.pi:
        n_points = int((2 * math.pi) / min_gap_rad) - 1
        
    angles = []
    while len(angles) < n_points:
        candidate = random.uniform(0, 2 * math.pi)
        if all(min(abs(candidate - a), 2 * math.pi - abs(candidate - a)) >= min_gap_rad for a in angles):
            angles.append(candidate)
            
    return sorted(angles)

def smooth_circular_array(arr, passes):
    n = len(arr)
    smoothed = list(arr)
    for _ in range(passes):
        temp = []
        for i in range(n):
            prev_val = smoothed[i - 1]
            curr_val = smoothed[i]
            next_val = smoothed[(i + 1) % n]
            temp.append((prev_val + curr_val + next_val) / 3)
        smoothed = temp
    return smoothed

def generate_loft_path(n_sections, total_height, n_points, angles, master_radii, max_drift, use_straight_lines, thickness_offset=0.0, vertical_smoothing=0):
    section_height = total_height / n_sections
    wp = cq.Workplane("XY")
    
    cur_x, cur_y = 0, 0
    current_radii = list(master_radii)
    
    for i in range(n_sections + 1):
        layer_points = []
        for j in range(n_points):
            angle = angles[j]
            r = max(0.5, current_radii[j] - thickness_offset) 
            
            px = (r * math.cos(angle)) + cur_x
            py = (r * math.sin(angle)) + cur_y
            layer_points.append((px, py))
        
        wp = wp.polyline(layer_points).close()
        
        if i < n_sections:
            shift_x = random.uniform(-max_drift * 0.4, max_drift * 0.4)
            shift_y = random.uniform(-max_drift * 0.4, max_drift * 0.4)
            center_move_dist = math.sqrt(shift_x**2 + shift_y**2)
            
            remaining_budget = max_drift - center_move_dist
            
            raw_dr = [random.uniform(-remaining_budget, remaining_budget) for _ in range(n_points)]
            smoothed_dr = smooth_circular_array(raw_dr, passes=vertical_smoothing)
            
            for j in range(n_points):
                current_radii[j] += smoothed_dr[j]
                current_radii[j] = max(1.0, current_radii[j])

            cur_x += shift_x
            cur_y += shift_y
            wp = wp.workplane(offset=section_height)
            
    return wp.loft(ruled=use_straight_lines, combine=True)

def create_thick_hollow_geometry(height_range, wall_thick_range, r_range, points_range, sections_range, use_straight_lines, vertical_smoothing):
    total_height = random.randint(height_range[0], height_range[1])
    wall_thickness = random.uniform(wall_thick_range[0], wall_thick_range[1])
    n_sections = random.randint(sections_range[0], sections_range[1])
    
    min_gap_deg = 8.0  
    max_allowable_points = int(360 / min_gap_deg) - 1
    
    chosen_points = random.randint(points_range[0], points_range[1])
    n_points = min(chosen_points, max_allowable_points)

    angles = generate_spaced_angles(n_points, min_gap_degrees=min_gap_deg)
    master_radii = [random.uniform(r_range[0], r_range[1]) for _ in range(len(angles))] 
    
    section_height = total_height / n_sections
    max_drift = section_height * 0.5 
    
    state = random.getstate()
    
    # Outer solid volume
    random.setstate(state)
    outer_solid = generate_loft_path(
        n_sections, total_height, len(angles), angles, master_radii, 
        max_drift, use_straight_lines, thickness_offset=0.0, vertical_smoothing=vertical_smoothing
    )
    
    # Inner solid volume
    random.setstate(state)
    inner_solid = generate_loft_path(
        n_sections, total_height, len(angles), angles, master_radii, 
        max_drift, use_straight_lines, thickness_offset=wall_thickness, vertical_smoothing=vertical_smoothing
    )
    
    thick_walled_geometry = outer_solid.cut(inner_solid)
    
    return thick_walled_geometry, inner_solid

def create_wall(shape, inner_core, t_range, l_range, h_range):
    try:
        bbox = shape.val().BoundingBox()
        spawn_z = random.uniform(bbox.zmin + 10, bbox.zmax - 5)
        
        base_cx = (bbox.xmin + bbox.xmax) / 2
        base_cy = (bbox.ymin + bbox.ymax) / 2

        thickness = random.uniform(t_range[0], t_range[1])
        length = random.uniform(l_range[0], l_range[1])
        outward_height = random.uniform(h_range[0], h_range[1]) 
        rotation = random.uniform(0, 360)
        rotation_rad = math.radians(rotation)
        
        n_segments = random.randint(1, 5)
        step = length / n_segments
        
        shape_params = []
        for _ in range(n_segments):
            shape_params.append((
                random.choice(["curvy", "jagged", "straight"]),
                random.uniform(-4, 4),
                random.uniform(-7, 7)
            ))

        def get_wire(z_pos, x_offset):
            path = cq.Workplane("XY").workplane(offset=z_pos).moveTo(x_offset, 0)
            curr_x = x_offset
            for mode, val1, val2 in shape_params:
                curr_x += step
                if mode == "straight":
                    path = path.lineTo(curr_x, 0)
                elif mode == "curvy":
                    path = path.spline([(curr_x - step/2, val1), (curr_x, 0)], includeCurrent=True)
                elif mode == "jagged":
                    path = path.lineTo(curr_x - step/2, val2).lineTo(curr_x, 0)
            return path.wire().offset2D(thickness / 2, kind="arc").objects[0]

        cos_a = abs(math.cos(rotation_rad))
        sin_a = abs(math.sin(rotation_rad))
        half_w = (bbox.xmax - bbox.xmin) / 2
        half_h = (bbox.ymax - bbox.ymin) / 2
        
        if half_w * sin_a > half_h * cos_a:
            approx_radius = half_h / sin_a if sin_a > 0 else half_h
        else:
            approx_radius = half_w / cos_a if cos_a > 0 else half_w

        touch_radius = approx_radius * 0.80
        
        spawn_x = base_cx + touch_radius * math.cos(rotation_rad)
        spawn_y = base_cy + touch_radius * math.sin(rotation_rad)

        wire_bottom = get_wire(0, -spawn_z) 
        wire_middle = get_wire(spawn_z, 0)

        support_section = cq.Workplane("XY")
        support_section.ctx.pendingWires.append(wire_bottom)
        support_section.ctx.pendingWires.append(wire_middle)
        support_solid = support_section.loft(combine=True)

        upper_wall = cq.Workplane("XY").add(wire_middle).toPending().extrude(outward_height)
        combined_wall_feature = support_solid.union(upper_wall).clean()

        combined_wall_feature = (
            combined_wall_feature
            .rotate((0, 0, 0), (0, 0, 1), rotation)
            .translate((spawn_x, spawn_y, 0))
        )

        joined_shape = shape.union(combined_wall_feature).clean()
        final_shape = joined_shape.cut(inner_core)

        return final_shape

    except Exception as e:
        print(f"Mid-body wall failed: {e}")
        return shape

def apply_clean_angled_cut(shape):
    try:
        bbox = shape.val().BoundingBox()
        base_w = bbox.xmax - bbox.xmin
        base_h = bbox.zmax - bbox.zmin
        center_x = (bbox.xmin + bbox.xmax) / 2
        center_y = (bbox.ymin + bbox.ymax) / 2
        
        cutter_size = base_w * 3
        cutter = cq.Workplane("XY").box(cutter_size, cutter_size, cutter_size, centered=(True, True, False))
        
        tilt_x = random.uniform(15, 35) * random.choice([-1, 1])
        tilt_y = random.uniform(15, 35) * random.choice([-1, 1])
        
        cutter = cutter.rotate((0,0,0), (1,0,0), tilt_x)
        cutter = cutter.rotate((0,0,0), (0,1,0), tilt_y)
        
        cut_depth = base_h * random.uniform(0.1, 0.2)
        target_z = bbox.zmax - cut_depth
        
        cutter = cutter.translate((center_x, center_y, target_z))
        return shape.cut(cutter)
    except Exception as e:
        print(f"Simple cut failed: {e}")
        return shape

def apply_choppy_top(shape):
    try:
        bbox = shape.val().BoundingBox()
        width = (bbox.xmax - bbox.xmin) * 4  
        top_z = bbox.zmax
        
        n_points = 7
        pts = []
        for i in range(n_points):
            x = -width/2 + (i * width/(n_points-1))
            z = random.uniform(-4, 4) 
            pts.append((x, z))
            
        cutter = (
            cq.Workplane("XZ")
            .spline(pts)
            .lineTo(width/2, 20)  
            .lineTo(-width/2, 20)
            .close()
            .extrude(width, both=True) 
        )
        
        cutter = cutter.rotateAboutCenter((0, 0, 1), random.uniform(0, 360))
        cutter = cutter.rotateAboutCenter((1, 0, 0), random.uniform(-10, 10))
        
        cut_depth = random.uniform(3, 6)
        cutter = cutter.translate((0, 0, top_z - cut_depth))
        
        return shape.cut(cutter)
    except Exception as e:
        print(f"Choppy cut failed: {e}")
        return shape

def create_shape(ui_params):
    shape, inner_core = create_thick_hollow_geometry(
        height_range=ui_params["height"],
        wall_thick_range=ui_params["wall_thick"],
        r_range=ui_params["radius"],
        points_range=ui_params["points"],
        sections_range=ui_params["sections"],
        use_straight_lines=ui_params["straight_lines"],
        vertical_smoothing=ui_params["smoothing"]
    )
    roof_feature_functions = [apply_clean_angled_cut, apply_choppy_top]
    
    nroof_features = random.randint(ui_params["roof_count"][0], ui_params["roof_count"][1])
    for _ in range(nroof_features):
        feature_fn = random.choice(roof_feature_functions)
        shape = feature_fn(shape)

    n_walls = random.randint(ui_params["wall_count"][0], ui_params["wall_count"][1])
    for _ in range(n_walls):
        shape = create_wall(
            shape, 
            inner_core,
            t_range=ui_params["wall_t"],
            l_range=ui_params["wall_l"],
            h_range=ui_params["wall_h"]
        )

    return shape


# 2. ADVANCED FIN ARCHITECTURE
def create_fin_architecture(ui_params):
    total_target_height = random.uniform(ui_params["height"][0], ui_params["height"][1])
    wall_thickness = random.uniform(ui_params["wall_thick"][0], ui_params["wall_thick"][1])
    n_points = random.randint(ui_params["n_points"][0], ui_params["n_points"][1])
    n_symmetry = random.randint(ui_params["n_symmetry"][0], ui_params["n_symmetry"][1]) 
    use_smooth_spline = ui_params["smooth"]
    if ui_params["fin_type"] == "Random":
        fin_style = random.choice(["rectangle", "triangle", "ellipse", "polygon"])
    else:
        fin_style = ui_params["fin_type"].lower()
    

    z_steps = [0.0]
    chunks = [random.uniform(6.0, 15.0) for _ in range(n_points - 1)]
    total_chunks = sum(chunks)
    for c in chunks:
        z_steps.append(z_steps[-1] + (c / total_chunks) * total_target_height)

    current_r = random.uniform(10, 45)
    wp_outer = cq.Workplane("XZ").moveTo(current_r, 0.0)

    outer_track = [(current_r, 0.0)]

    for i in range(1, n_points):
        prev_r, prev_z = outer_track[-1]
        next_z = z_steps[i]
        dz = next_z - prev_z
        
        # 45-degree constraint
        max_dr = dz * 0.90  
        dr = random.uniform(-max_dr, max_dr)
        
        next_r = max(5.0, prev_r + dr)
        
        if use_smooth_spline and abs(dr) > 2.0:
            mid_r = prev_r + (dr / 2.0)
            mid_z = prev_z + (dz / 2.0)
            wp_outer = wp_outer.spline([(mid_r, mid_z), (next_r, next_z)], includeCurrent=True)
        else:
            wp_outer = wp_outer.lineTo(next_r, next_z)
            
        outer_track.append((next_r, next_z))

    wp_hollow = wp_outer
    inner_track = []
    for r, z in reversed(outer_track):
        inner_r = max(2.0, r - wall_thickness)
        inner_track.append((inner_r, z))

    for ir, iz in inner_track:
        wp_hollow = wp_hollow.lineTo(ir, iz)
        
    hub_hollow = wp_hollow.close().revolve(360)
    hub_outer_solid = cq.Workplane("XZ").polyline(outer_track + [(0, total_target_height), (0, 0)]).close().revolve(360)
    hub_inner_solid = cq.Workplane("XZ").polyline([(0, 0)] + inner_track[::-1] + [(0, total_target_height)]).close().revolve(360)

    max_base_r = max(r for r, z in outer_track)
    
    absolute_max_radius = 75.0 
    available_room = absolute_max_radius - max_base_r
    
    b_min, b_max = ui_params["blade_l"]
    safe_b_max = max(b_min, min(b_max, available_room)) 
    
    blade_length = random.uniform(b_min, safe_b_max)
    blade_thick = random.uniform(ui_params["blade_t"][0], ui_params["blade_t"][1]) 
    total_fin_reach = max_base_r + blade_length

    crowding_factor = n_symmetry * blade_thick
    
    if ui_params["path_style"] == "Random":
        if crowding_factor > 18:
            path_style = random.choice(["linear", "bow"])
        else:
            path_style = random.choice(["linear", "bow", "s_curve"])
    else:
        path_style = ui_params["path_style"].lower().replace("-", "_")

    if path_style == "linear":
        overhang_limit = total_target_height * 0.85
    elif path_style == "bow":
        overhang_limit = total_target_height * 0.40 
    else: 
        overhang_limit = total_target_height * 0.14 

    min_hub_circumference = 2 * math.pi * current_r
    fin_gap = min_hub_circumference / n_symmetry
    collision_limit = fin_gap * 0.40 

    max_safe_lean = min(overhang_limit, collision_limit)
    lean_dir = random.choice([-1, 1])
    lean_y = random.uniform(max_safe_lean * 0.2, max_safe_lean) * lean_dir

    if path_style == "linear":
        path_pts = [(0, 0), (lean_y, total_target_height)]
    elif path_style == "bow":
        path_pts = [(0, 0), (lean_y, total_target_height * 0.5), (0, total_target_height)]
    else: 
        path_pts = [(0, 0), (lean_y, total_target_height * 0.33), (-lean_y, total_target_height * 0.66), (0, total_target_height)]

    fin_path = cq.Workplane("YZ").spline(path_pts, includeCurrent=False)

    use_alternating = random.choice([True, False])
    secondary_reach = max_base_r + (blade_length * random.uniform(0.3, 0.7))

    def make_fin_profile(reach):
        wp = cq.Workplane("XY")
        anchor_depth = max_base_r 
        total_len = reach + anchor_depth
        
        if fin_style == "rectangle":
            wp = wp.moveTo(-anchor_depth, -blade_thick/2).rect(total_len, blade_thick, centered=(False, True))
        elif fin_style == "triangle":
            wp = wp.polyline([(-anchor_depth, 0), (reach, -blade_thick/2), (-anchor_depth, blade_thick/2)]).close()
        elif fin_style == "ellipse":
            wp = wp.ellipse(total_len/2, blade_thick/2).translate(((reach - anchor_depth)/2, 0))
        elif fin_style == "polygon":
            p1 = (-anchor_depth, -blade_thick / 2)
            p2 = (reach, -blade_thick / 4)
            p3 = (reach * random.uniform(0.7, 0.95), blade_thick / 2)
            p4 = (-anchor_depth, blade_thick * random.uniform(0.2, 0.7))
            wp = wp.polyline([p1, p2, p3, p4]).close()
        return wp

    r_shave_top = outer_track[-1][0] + random.uniform(0.5, 6.0)
    shave_profile = [(0, 0), (total_fin_reach, 0), (r_shave_top, total_target_height), (0, total_target_height)]
    shave_cone = cq.Workplane("XZ").polyline(shave_profile).close().revolve(360)

    raw_fin_primary = make_fin_profile(total_fin_reach).sweep(fin_path, transition='round').intersect(shave_cone)
    raw_fin_secondary = make_fin_profile(secondary_reach).sweep(fin_path, transition='round').intersect(shave_cone)

    featured_assembly = hub_outer_solid
    
    for s in range(n_symmetry):
        if random.random() < 0.15:
            continue
            
        angle = s * (360.0 / n_symmetry)
        current_fin = raw_fin_secondary if (use_alternating and s % 2 != 0) else raw_fin_primary
        rotated_fin = current_fin.rotate((0, 0, 0), (0, 0, 1), angle)
        featured_assembly = featured_assembly.union(rotated_fin)

    final_shape = featured_assembly.cut(hub_inner_solid)
    combined_part = final_shape.union(hub_hollow).clean()

    return combined_part



# 3. HEAT EXCHANGER GENERATOR
def create_advanced_chaotic_heat_exchanger(ui_params):
    total_length = random.uniform(ui_params["length"][0], ui_params["length"][1])      
    base_fin_height = random.uniform(ui_params["height"][0], ui_params["height"][1])    
    fin_thickness = random.uniform(ui_params["fin_t"][0], ui_params["fin_t"][1])   
    shroud_thick = random.uniform(ui_params["shroud_t"][0], ui_params["shroud_t"][1])    
    
    if ui_params["macro_layout"] == "Random":
        macro_layout = random.choice(["venturi", "s_curve", "tapered"])
    else:
        macro_layout = ui_params["macro_layout"].lower().replace("-", "_")
        
    nominal_width = random.uniform(ui_params["width"][0], ui_params["width"][1])
    target_pitch = random.uniform(ui_params["pitch"][0], ui_params["pitch"][1])
    n_fins = max(3, int(nominal_width / target_pitch))
    
    n_control_points = 6
    y_steps = [((total_length / (n_control_points - 1)) * i) for i in range(n_control_points)]
    
    inner_left_pts = []
    inner_right_pts = []

    for i, y in enumerate(y_steps):
        progress = y / total_length
        
        if macro_layout == "venturi":
            width_factor = 1.0 + 0.8 * math.sin(progress * math.pi)
            current_width = nominal_width * width_factor
            x_offset = 0.0
        elif macro_layout == "s_curve":
            current_width = nominal_width * random.uniform(0.9, 1.1)
            x_offset = 22.0 * math.sin(progress * 1.5 * math.pi)
        else: # "tapered"
            width_factor = 1.4 - 0.9 * progress
            current_width = nominal_width * width_factor
            x_offset = random.uniform(-6.0, 6.0) * progress
            
        inner_left_pts.append((x_offset - current_width / 2.0, y))
        inner_right_pts.append((x_offset + current_width / 2.0, y))

    outer_left_pts = [(x - shroud_thick, y) for x, y in inner_left_pts]
    outer_right_pts = [(x + shroud_thick, y) for x, y in inner_right_pts]

    outer_shroud_solid = (
        cq.Workplane("XY")
        .moveTo(outer_left_pts[0][0], outer_left_pts[0][1])
        .spline(outer_left_pts[1:], includeCurrent=True)
        .lineTo(outer_right_pts[-1][0], outer_right_pts[-1][1])
        .spline(outer_right_pts[::-1][1:], includeCurrent=True)
        .close()
        .extrude(base_fin_height + shroud_thick + 10.0)
    )
    
    inner_tunnel_cutter = (
        cq.Workplane("XY")
        .moveTo(inner_left_pts[0][0], inner_left_pts[0][1])
        .spline(inner_left_pts[1:], includeCurrent=True)
        .lineTo(inner_right_pts[-1][0], inner_right_pts[-1][1])
        .spline(inner_right_pts[::-1][1:], includeCurrent=True)
        .close()
        .extrude(base_fin_height + 40.0) 
        .translate((0, 0, shroud_thick)) 
    )
    
    shroud = outer_shroud_solid.cut(inner_tunnel_cutter)
    assembly = shroud

    for f in range(n_fins):
        t = f / (n_fins - 1) if n_fins > 1 else 0.5
        
        amplitude = random.uniform(1.8, 3.8)
        phase_offset = random.uniform(0, 2 * math.pi) 
        frequency = random.choice([1.0, 1.5, 2.0])
        
        fin_left_track = []
        fin_right_track = []
        
        for i, y in enumerate(y_steps):
            xl = inner_left_pts[i][0] - 2.0
            xr = inner_right_pts[i][0] + 2.0
            
            x_nominal = xl + t * (xr - xl) 
            
            if ui_params["fin_pattern"] == "Parallel to Shroud":
                x_center = x_nominal
            else:
                wave_y = (y / total_length) * (2 * math.pi) * frequency
                x_center = x_nominal + (amplitude * math.sin(wave_y + phase_offset))
            
            fin_left_track.append((x_center - fin_thickness / 2.0, y))
            fin_right_track.append((x_center + fin_thickness / 2.0, y))
            
        fin_wp = (
            cq.Workplane("XY")
            .moveTo(fin_left_track[0][0], fin_left_track[0][1])
            .spline(fin_left_track[1:], includeCurrent=True)
            .lineTo(fin_right_track[-1][0], fin_right_track[-1][1])
            .spline(fin_right_track[::-1][1:], includeCurrent=True)
            .close()
        )
        
        fin_solid = fin_wp.extrude(base_fin_height + shroud_thick + 10.0)
        assembly = assembly.union(fin_solid)

    # pins 
    if ui_params["enable_cuts"]: 
        num_cuts = random.randint(ui_params["num_cuts"][0], ui_params["num_cuts"][1]) 
        cut_width = random.uniform(1.0, 0.5 * total_length / num_cuts)
        
        for i in range(1, num_cuts):
            cut_y = (total_length / num_cuts) * i
            cutter = (
                cq.Workplane("XY")
                .workplane(offset=2.0) 
                .center(0, cut_y)
                .box(nominal_width * 3, cut_width, base_fin_height * 3, centered=(True, True, False))
            )
            assembly = assembly.cut(cutter)

    n_top_segments = random.randint(ui_params["top_segments"][0], ui_params["top_segments"][1])
    if ui_params["top_style"] == "Random":
        top_cut_style = random.choice(["smooth_wave", "faceted_stepped"])
    else:
        top_cut_style = ui_params["top_style"].lower().replace(" ", "_")
 

    top_y_steps = [((total_length / (n_top_segments - 1)) * i) for i in range(n_top_segments)]
    
    top_profile_pts = []
    for y_val in top_y_steps:
        h_val = base_fin_height * random.uniform(0.40, 1.15)
        top_profile_pts.append((y_val, h_val))
        
    shaper_wp = cq.Workplane("YZ").moveTo(top_profile_pts[0][0], top_profile_pts[0][1])
    
    if top_cut_style == "smooth_wave":
        shaper_wp = shaper_wp.spline(top_profile_pts[1:], includeCurrent=True)
    else:
        for pt in top_profile_pts[1:]:
            shaper_wp = shaper_wp.lineTo(pt[0], pt[1])
            
    top_shaper_tool_wire = (
        shaper_wp
        .lineTo(total_length, base_fin_height + 60.0)
        .lineTo(0, base_fin_height + 60.0)
        .close()
    )
    
    max_x = max(abs(pt[0]) for pt in outer_left_pts + outer_right_pts) + 60.0
    top_shaper_tool_solid = top_shaper_tool_wire.extrude(max_x * 2.0, both=True)
    
    final_heat_exchanger = assembly.cut(top_shaper_tool_solid).clean()
    return final_heat_exchanger


# 4. BRACKET GENERATOR 
def make_standard_hole(size, shape_type):
    hw = size / 2
    if shape_type == "diamond":
        th = hw * 1.15 
        return [(0, th), (hw, 0), (0, -th), (-hw, 0)]
    elif shape_type == "elongated_diamond": 
        th = hw * random.uniform(1.8, 2.5)
        return [(0, th), (hw, 0), (0, -th), (-hw, 0)]
    elif shape_type == "hex": 
        roof_h = hw * 1.2  
        wall_h = hw * 0.4  
        w = hw * 0.7       
        return [(0, roof_h), (w, wall_h), (w, -wall_h), (0, -roof_h), (-w, -wall_h), (-w, wall_h)]
    elif shape_type == "teardrop":
        tip_y = hw * math.sqrt(2)
        pts = [(0, tip_y)]
        segments = 24
        start_angle = math.pi / 4
        end_angle = -5 * math.pi / 4
        for i in range(segments + 1):
            theta = start_angle + (end_angle - start_angle) * (i / segments)
            pts.append((hw * math.cos(theta), hw * math.sin(theta)))
        return pts
    return make_standard_hole(size, "diamond")

def generate_safe_holes(x_min, x_max, max_size, min_gap=2.5):
    holes = []
    curr_x = x_min + min_gap
    while True:
        available = x_max - curr_x - min_gap
        if available < 1.5: break
        
        if random.random() < 0.30: size = random.uniform(1.5, min(3.5, available))
        else: size = random.uniform(2.5, min(max_size, available))
            
        cx = curr_x + size / 2
        holes.append((cx, size))
        curr_x += size + min_gap
    return holes

def apply_wavy_profile(part, length, center_x, max_y, web_y, force_wave):
    if not force_wave: return part
    
    waves = random.randint(1, 4) 
    max_amp = (max_y - web_y) * 0.48 
    if max_amp <= 2.0: return part 
    amplitude = random.uniform(3.0, max_amp)
    
    pts = []
    segments = 60
    start_x = center_x - length / 2
    for i in range(segments + 1):
        t = i / segments
        pts.append((start_x + t * length, max_y/2 - math.sin(t * math.pi * waves) * amplitude))
    for i in range(segments + 1):
        t = 1 - (i / segments)
        pts.append((start_x + t * length, -max_y/2 + math.sin(t * math.pi * waves) * amplitude))

    wavy_box = cq.Workplane("XY").polyline(pts).close().extrude(2000, both=True)
    return part.intersect(wavy_box)

def make_wavy_sloped_edge(p_top, p_bot):
    dx = p_bot[0] - p_top[0]
    dz = p_bot[1] - p_top[1] 
    line_length = math.hypot(dx, dz)
    if line_length < 15.0: return [p_top, p_bot]
    max_waves = max(1, int(line_length / 25.0))
    waves = random.randint(1, min(4, max_waves))
    amp = random.uniform(0.8, 1.5) 
    nx = -dz / line_length
    nz = dx / line_length
    segments = max(40, waves * 15)
    pts = []
    for i in range(segments + 1):
        t = i / segments
        base_x = p_top[0] + t * dx
        base_z = p_top[1] + t * dz
        dampening = math.sin(t * math.pi)
        wave_val = math.sin(t * math.pi * waves * 2) * amp * dampening
        x = base_x + (nx * wave_val)
        z = base_z + (nz * wave_val)
        pts.append((x, z))
    return pts

def apply_professional_finishing(part):
    try: part = part.faces(">Z").edges().chamfer(random.uniform(0.5, 0.8))
    except: pass
    try: part = part.edges("not |X and not |Y and not |Z").chamfer(random.uniform(0.5, 0.8))
    except: pass
    try: part = part.edges("%CIRCLE").chamfer(random.uniform(0.4, 0.7))
    except: pass
    try: part = part.edges("%LINE and not <Z").chamfer(random.uniform(0.5, 0.8))
    except: pass
    try: part = part.edges("|Z").fillet(random.uniform(1.0, 2.0))
    except: pass
    return part

# Linear Brackets
def build_linear_bracket(ui_params):
    height = random.uniform(ui_params["lin_h"][0], ui_params["lin_h"][1]) 
    boss_len = random.uniform(ui_params["lin_boss_l"][0], ui_params["lin_boss_l"][1])
    boss_height = random.uniform(10, 20)
    
    if ui_params["lin_type"] == "Random": is_l_bracket = random.choice([True, False])
    else: is_l_bracket = ui_params["lin_type"] == "L-Bracket"
        
    if ui_params["pocket"] == "Random": pocket_style = random.choice(["Webbed", "Hollow-Frame"])
    else: pocket_style = ui_params["pocket"]
    
    base_y = random.uniform(ui_params["lin_base_y"][0], ui_params["lin_base_y"][1])
    web_y = random.uniform(ui_params["web_t"][0], ui_params["web_t"][1])
    pocket_wall = random.uniform(ui_params["pocket_wall"][0], ui_params["pocket_wall"][1])           
    base_thick = random.uniform(ui_params["base_t"][0], ui_params["base_t"][1])

    if is_l_bracket:
        a1_deg = 90 
        a2_deg = random.uniform(ui_params["lin_taper"][0], ui_params["lin_taper"][1]) 
        top_x_center = random.uniform(boss_len/2, boss_len * 1.2)
    else:
        a1_deg = random.uniform(ui_params["lin_taper"][0], ui_params["lin_taper"][1])
        a2_deg = random.uniform(ui_params["lin_taper"][0], ui_params["lin_taper"][1])
        top_x_center = random.uniform(-15, 15)

    dx1 = height / math.tan(math.radians(a1_deg))
    dx2 = height / math.tan(math.radians(a2_deg))
    C, D = (top_x_center - boss_len/2, height), (top_x_center + boss_len/2, height)
    A, B = (C[0] - dx1, base_thick), (D[0] + dx2, base_thick)
    
    left_tab_len = random.uniform(15, 25) if not is_l_bracket else base_thick + 5
    right_tab_len = random.uniform(15, 25)
    
    base_min_x, base_max_x = A[0] - left_tab_len, B[0] + right_tab_len
    base_len = base_max_x - base_min_x
    base_center_x = (base_max_x + base_min_x) / 2
    
    part = cq.Workplane("XY").workplane(offset=base_thick/2).center(base_center_x, 0).box(base_len, base_y, base_thick)
    flange_y_max = base_y * random.uniform(0.6, 0.8) 

    gusset_w = random.uniform(5, 10)
    gusset_pts = [(top_x_center, height), (top_x_center + 15, base_thick), (top_x_center - 15, base_thick)]
    part = part.union(cq.Workplane("XZ").polyline(gusset_pts).close().extrude(gusset_w/2, both=True))

    wavy_on = ui_params["lin_diag_wavy"]
    if is_l_bracket:
        pts_left = [C, A] 
    else:
        pts_left = make_wavy_sloped_edge(C, A) if wavy_on else [C, A]
        
    pts_left.reverse() 
    pts_right = make_wavy_sloped_edge(D, B) if wavy_on else [D, B]
    
    body = cq.Workplane("XZ").polyline(pts_left + pts_right).close().extrude(flange_y_max / 2, both=True)
    boss = cq.Workplane("XZ").center(top_x_center, height).rect(boss_len, boss_height).extrude(flange_y_max / 2, both=True)
    part = part.union(body).union(boss)

    part = apply_wavy_profile(part, base_len, base_center_x, base_y, web_y, ui_params["lin_base_wavy"])

    boss_y = random.uniform(4, 8) 
    taper_z_start = base_thick + random.uniform(1, 4)
    taper_pts = [(boss_y/2, height + 20), (base_y/2, taper_z_start), (base_y/2, -20), (-base_y/2, -20), (-base_y/2, taper_z_start), (-boss_y/2, height + 20)]
    part = part.intersect(cq.Workplane("YZ").polyline(taper_pts).close().extrude(2000, both=True))

    pocket_apex_x = top_x_center
    pocket_apex_z = height - pocket_wall
    max_dx = (pocket_apex_z - (base_thick + pocket_wall)) / 1.05 
    p_left_x = max(A[0] + pocket_wall, pocket_apex_x - max_dx)
    p_right_x = min(B[0] - pocket_wall, pocket_apex_x + max_dx)

    if (p_right_x - p_left_x) > 5.0: 
        pocket_pts = [(p_left_x, base_thick + pocket_wall), (p_right_x, base_thick + pocket_wall), (pocket_apex_x, pocket_apex_z)]
        if pocket_style == "Webbed":
            part = part.cut(cq.Workplane("XZ").workplane(offset=web_y/2).polyline(pocket_pts).close().extrude(flange_y_max * 1.5))
            part = part.cut(cq.Workplane("XZ").workplane(offset=-web_y/2).polyline(pocket_pts).close().extrude(-flange_y_max * 1.5))
            
            density = ui_params["lin_hole_density"]
            layer_map = {"Sparse": random.randint(2, 3), "Normal": random.randint(3, 5), "Dense": random.randint(5, 8)}
            gap_map = {"Sparse": 4.5, "Normal": 2.5, "Dense": 1.0}
            
            num_layers = layer_map[density]
            layer_spacing = (pocket_apex_z - base_thick - pocket_wall*2) / num_layers
            max_hole_h = layer_spacing * 0.9  
            
            for i in range(num_layers):
                cz = (base_thick + pocket_wall * 1.5) + i * layer_spacing
                z_ratio = (cz - base_thick) / (pocket_apex_z - base_thick)
                layer_left = p_left_x * (1 - z_ratio) + pocket_apex_x * z_ratio
                layer_right = p_right_x * (1 - z_ratio) + pocket_apex_x * z_ratio
                
                safe_holes = generate_safe_holes(layer_left, layer_right, max_size=max_hole_h, min_gap=gap_map[density])
                
                for cx, h_size in safe_holes:
                    if ui_params["lin_hole_shape"] == "Random":
                        shape_t = random.choice(["diamond", "teardrop", "hex", "elongated_diamond"])
                    else:
                        shape_t = ui_params["lin_hole_shape"].lower().replace(" ", "_")
                        
                    safe_size = h_size
                    if shape_t == "elongated_diamond": safe_size = h_size * 0.45 
                    elif shape_t in ["diamond", "teardrop"]: safe_size = h_size * 0.75
                    
                    part = part.cut(cq.Workplane("XZ").center(cx, cz + random.uniform(-0.5, 0.5)).polyline(make_standard_hole(safe_size, shape_t)).close().extrude(flange_y_max * 1.5, both=True))
        else: 
            part = part.cut(cq.Workplane("XZ").polyline(pocket_pts).close().extrude(flange_y_max * 3, both=True))

    base_holes = []
    if not is_l_bracket and left_tab_len > 12: base_holes.append((A[0] - left_tab_len / 2, 0))
    if right_tab_len > 12: base_holes.append((B[0] + right_tab_len / 2, 0))
    
    hole_r = random.uniform(2.5, 4.0)
    for h in base_holes:
        pad = cq.Workplane("XY").center(h[0], h[1]).circle(hole_r + random.uniform(3, 5)).extrude(base_thick + random.uniform(2, 4))
        part = part.union(pad)
    if base_holes:
        part = part.cut(cq.Workplane("XY").workplane(offset=-1).pushPoints(base_holes).circle(hole_r).extrude(base_thick + 8))

    part = part.cut(cq.Workplane("XZ").center(top_x_center, height).polyline(make_standard_hole(boss_height * 0.4, "hex")).close().extrude(flange_y_max + 10, both=True))
    return apply_professional_finishing(part)

# Radial Brackets
def build_asymmetric_radial_bracket(ui_params):
    if ui_params["rad_hub"] == "Random": hub_shape = random.choice(["cylinder", "hexagon", "square", "hollow_collar"])
    else: hub_shape = ui_params["rad_hub"].lower().replace(" ", "_")
        
    if ui_params["pocket"] == "Random": pocket_style = random.choice(["Webbed", "Hollow-Frame"])
    else: pocket_style = ui_params["pocket"]

    boss_r = random.uniform(ui_params["rad_boss_r"][0], ui_params["rad_boss_r"][1])
    hub_height = random.uniform(ui_params["rad_hub_h"][0], ui_params["rad_hub_h"][1])
    base_thick = random.uniform(ui_params["base_t"][0], ui_params["base_t"][1])
    
    if ui_params["rad_layout"] == "Random":
        layout_type = random.choice(["V-Corner", "Y-Junction", "Star-Cross", "Offset-T"])
    else:
        layout_type = ui_params["rad_layout"].split(" ")[0] 

    if layout_type == "V-Corner": angles = [0, random.uniform(60, 130)]
    elif layout_type == "Y-Junction": angles = [0, random.uniform(100, 150), random.uniform(210, 260)]
    elif layout_type == "Offset-T": angles = [0, 180, random.uniform(70, 110)]
    else: angles = [0, random.uniform(75, 105), random.uniform(165, 195), random.uniform(255, 285)]

    hub_wp = cq.Workplane("XY")
    if hub_shape in ["cylinder", "hollow_collar"]: part = hub_wp.circle(boss_r).extrude(hub_height)
    elif hub_shape == "hexagon": part = hub_wp.polygon(6, boss_r * 2).extrude(hub_height)
    else: part = hub_wp.rect(boss_r * 1.6, boss_r * 1.6).extrude(hub_height)
    
    cone_r_bot = boss_r * 1.5
    cone_r_top = boss_r * random.uniform(0.4, 0.8)
    cone = cq.Workplane("XY").circle(cone_r_bot).workplane(offset=hub_height + 5).circle(cone_r_top).loft()
    part = part.intersect(cone)
    
    for angle in angles:
        leg_height = random.uniform(hub_height * 0.4, hub_height) 
        leg_len = random.uniform(boss_r + 20, (leg_height - base_thick) / 1.3 + boss_r)
        tab_len = random.uniform(15, 50) 
        flange_y = random.uniform(10, boss_r * 1.5) 
        web_y = random.uniform(ui_params["web_t"][0], ui_params["web_t"][1])
        pocket_wall = random.uniform(ui_params["pocket_wall"][0], ui_params["pocket_wall"][1])
        pad_width = flange_y * random.uniform(1.1, 1.6)

        start_x = -boss_r * 0.2 
        total_pad_len = (leg_len - start_x) + tab_len
        pad_center_x = start_x + (total_pad_len / 2)
        leg = cq.Workplane("XY").workplane(offset=base_thick/2).center(pad_center_x, 0).box(total_pad_len, pad_width, base_thick)

        p_top, p_bot = (boss_r * 0.8, leg_height), (leg_len, base_thick)
        
        if ui_params["rad_diag_wavy"]:
            wavy_edge = make_wavy_sloped_edge(p_top, p_bot)
        else:
            wavy_edge = [p_top, p_bot]
            
        wavy_edge.reverse() 
        pts = [(start_x, base_thick)] + wavy_edge + [(start_x, leg_height)]
        leg = leg.union(cq.Workplane("XZ").polyline(pts).close().extrude(flange_y / 2, both=True))

        leg = apply_wavy_profile(leg, total_pad_len, pad_center_x, pad_width, web_y, ui_params["rad_base_wavy"])
        
        pocket_apex_z = leg_height - pocket_wall
        max_dx = (pocket_apex_z - (base_thick + pocket_wall)) / 1.05
        safe_inner_leg_len = min(leg_len - pocket_wall, (boss_r * 0.7) + max_dx)
        
        if safe_inner_leg_len > (boss_r * 0.7 + 5):
            pocket_pts = [(boss_r * 0.7, base_thick + pocket_wall), (safe_inner_leg_len, base_thick + pocket_wall), (boss_r * 0.7, pocket_apex_z)]
            if pocket_style == "Webbed":
                pocket_right = cq.Workplane("XZ").workplane(offset=web_y/2).polyline(pocket_pts).close().extrude(flange_y * 1.5)
                pocket_left = cq.Workplane("XZ").workplane(offset=-web_y/2).polyline(pocket_pts).close().extrude(-flange_y * 1.5)
                leg = leg.cut(pocket_right).cut(pocket_left)
            
                density = ui_params["rad_hole_density"]
                layer_map = {"Sparse": random.randint(2, 3), "Normal": random.randint(3, 5), "Dense": random.randint(5, 8)}
                gap_map = {"Sparse": 4.5, "Normal": 2.5, "Dense": 1.0}
                
                num_layers = layer_map[density]
                layer_spacing = (leg_height - base_thick - pocket_wall*2) / num_layers
                max_hole_h = layer_spacing * 0.85
                
                for i in range(num_layers):
                    cz = (base_thick + pocket_wall * 1.5) + i * layer_spacing
                    z_ratio = (cz - base_thick) / (leg_height - base_thick)
                    layer_left = boss_r * 0.7
                    layer_right = safe_inner_leg_len * (1 - z_ratio) + (boss_r * 0.7) * z_ratio
                    
                    safe_holes = generate_safe_holes(layer_left, layer_right, max_size=max_hole_h, min_gap=gap_map[density])
                    for cx, h_size in safe_holes:
                        if ui_params["rad_hole_shape"] == "Random":
                            shape_t = random.choice(["diamond", "hex", "elongated_diamond", "teardrop"])
                        else:
                            shape_t = ui_params["rad_hole_shape"].lower().replace(" ", "_")
                            
                        safe_size = h_size
                        if shape_t == "elongated_diamond": safe_size = h_size * 0.45 
                        elif shape_t in ["diamond", "teardrop"]: safe_size = h_size * 0.75
                        
                        leg = leg.cut(cq.Workplane("XZ").center(cx, cz).polyline(make_standard_hole(safe_size, shape_t)).close().extrude(flange_y * 1.5, both=True))
            else:
                leg = leg.cut(cq.Workplane("XZ").polyline(pocket_pts).close().extrude(flange_y * 3, both=True))

        hole_r = random.uniform(2.5, 4.5)
        pad_x = leg_len + tab_len / 2
        pad_h = base_thick + random.uniform(1.5, 3.5)
        leg = leg.union(cq.Workplane("XY").center(pad_x, 0).circle(hole_r + random.uniform(3, 6)).extrude(pad_h))
        leg = leg.cut(cq.Workplane("XY").center(pad_x, 0).circle(hole_r).extrude(pad_h * 2, both=True))
        
        part = part.union(leg.rotate((0, 0, 0), (0, 0, 1), angle))

    if hub_shape == "hollow_collar": part = part.cut(cq.Workplane("XY").circle(boss_r * 0.65).extrude(hub_height * 3, both=True))
    else: part = part.cut(cq.Workplane("XY").circle(boss_r * random.uniform(0.2, 0.4)).extrude(hub_height * 3, both=True))

    return apply_professional_finishing(part)

# Angled extrusion Brackets
def build_angled_extension_bracket(ui_params):
    base_thick = random.uniform(ui_params["base_t"][0], ui_params["base_t"][1])
    arm_base_len = random.uniform(ui_params["ang_arm_l"][0], ui_params["ang_arm_l"][1])
    arm_width = random.uniform(ui_params["ang_arm_w"][0], ui_params["ang_arm_w"][1])
    h_arm = random.uniform(ui_params["ang_arm_h"][0], ui_params["ang_arm_h"][1]) 
    
    lean_angle = math.radians(random.uniform(ui_params["ang_lean"][0], ui_params["ang_lean"][1])) 
    pocket_wall = random.uniform(ui_params["pocket_wall"][0], ui_params["pocket_wall"][1])
    
    top_len = random.uniform(10, 50)
    web_y = random.uniform(ui_params["web_t"][0], ui_params["web_t"][1])
    
    if ui_params["pocket"] == "Random": pocket_style = random.choice(["Webbed", "Hollow-Frame"])
    else: pocket_style = ui_params["pocket"]
    
    generate_serrations = ui_params["ang_serrations"] 

    p1, p2 = (-arm_base_len/2, 0), (arm_base_len/2, 0)
    p3 = (p2[0] + h_arm/math.tan(lean_angle), h_arm)
    p4 = (p3[0] - top_len, h_arm)
    arm = cq.Workplane("XZ").polyline([p1, p2, p3, p4]).close().extrude(arm_width/2, both=True)

    top_w = arm_width * random.uniform(0.3, 0.6)
    wedge_pts = [(arm_width/2, 0), (top_w/2, h_arm+10), (-top_w/2, h_arm+10), (-arm_width/2, 0)]
    wedge = cq.Workplane("YZ").polyline(wedge_pts).close().extrude(arm_base_len * 4, both=True)
    arm = arm.intersect(wedge)

    if generate_serrations:
        num_teeth = random.randint(3, 12)
        tooth_pitch = top_len / max(1, num_teeth)
        tooth_depth = random.uniform(1.5, 4.0)
        for i in range(num_teeth):
            tx, tz = p4[0] + (i + 0.5) * tooth_pitch, h_arm
            groove_pts = [(tx - tooth_pitch/3, tz + 1), (tx + tooth_pitch/3, tz + 1), (tx, tz - tooth_depth)]
            arm = arm.cut(cq.Workplane("XZ").polyline(groove_pts).close().extrude(arm_width + 10, both=True))

    pocket_z_bottom = pocket_wall
    pocket_apex_z = h_arm - pocket_wall - (3.0 if generate_serrations else 0)
    pck_apex_x = (p3[0] + p4[0]) / 2
    max_dx = (pocket_apex_z - pocket_z_bottom) / 1.05
    wall_left_x = p1[0] + pocket_z_bottom/math.tan(lean_angle) + pocket_wall
    wall_right_x = p2[0] + pocket_z_bottom/math.tan(lean_angle) - pocket_wall
    pck_bot_left_x = max(wall_left_x, pck_apex_x - max_dx)
    pck_bot_right_x = min(wall_right_x, pck_apex_x + max_dx)

    if (pck_bot_right_x - pck_bot_left_x) > 10:
        pocket_pts = [(pck_bot_left_x, pocket_z_bottom), (pck_bot_right_x, pocket_z_bottom), (pck_apex_x, pocket_apex_z)]
        if pocket_style == "Webbed":
            p_right = cq.Workplane("XZ").workplane(offset=web_y/2).polyline(pocket_pts).close().extrude(arm_width)
            p_left = cq.Workplane("XZ").workplane(offset=-web_y/2).polyline(pocket_pts).close().extrude(-arm_width)
            arm = arm.cut(p_right).cut(p_left)

            density = ui_params["ang_hole_density"]
            layer_map = {"Sparse": random.randint(2, 3), "Normal": random.randint(3, 5), "Dense": random.randint(5, 8)}
            gap_map = {"Sparse": 4.5, "Normal": 2.5, "Dense": 1.0}
            
            num_layers = layer_map[density]
            layer_spacing = (pocket_apex_z - pocket_z_bottom) / num_layers
            max_hole_h = layer_spacing * 0.85
            
            for i in range(num_layers):
                cz = pocket_z_bottom + layer_spacing * (i + 0.5)
                z_ratio = (cz - pocket_z_bottom) / (pocket_apex_z - pocket_z_bottom)
                layer_left = pck_bot_left_x * (1 - z_ratio) + pck_apex_x * z_ratio
                layer_right = pck_bot_right_x * (1 - z_ratio) + pck_apex_x * z_ratio
                
                safe_holes = generate_safe_holes(layer_left, layer_right, max_size=max_hole_h, min_gap=gap_map[density])
                for cx, h_size in safe_holes:
                    if ui_params["ang_hole_shape"] == "Random":
                        shape_t = random.choice(["hex", "elongated_diamond", "teardrop", "diamond"])
                    else:
                        shape_t = ui_params["ang_hole_shape"].lower().replace(" ", "_")
                        
                    safe_size = h_size
                    if shape_t == "elongated_diamond": safe_size = h_size * 0.45 
                    elif shape_t in ["diamond", "teardrop"]: safe_size = h_size * 0.75
                    
                    arm = arm.cut(cq.Workplane("XZ").center(cx, cz).polyline(make_standard_hole(safe_size, shape_t)).close().extrude(arm_width*1.5, both=True))
        else:
            arm = arm.cut(cq.Workplane("XZ").polyline(pocket_pts).close().extrude(arm_width*3, both=True))

    arm = apply_wavy_profile(arm, (p3[0] - p1[0]) + 20, (p1[0]+p3[0])/2, h_arm, web_y, ui_params["ang_wavy"])

    root_radius = math.hypot(arm_base_len/2, arm_width/2)
    
    base_pad_x = random.uniform(ui_params["ang_base_pad"][0], ui_params["ang_base_pad"][1])
    base_pad_y = random.uniform(ui_params["ang_base_pad"][0], ui_params["ang_base_pad"][1])
    base_x, base_y = root_radius * 2 + base_pad_x, root_radius * 2 + base_pad_y
    
    yaw_angle = random.uniform(-180, 180) 
    
    if ui_params["ang_base"] == "Random": base_style = random.choice(["rect", "circle", "chamfered_poly"])
    else: base_style = ui_params["ang_base"].lower().replace(" ", "_")
        
    hole_r = random.uniform(3, 5)
    pad_h = base_thick + random.uniform(2, 4)
    holes = []
    
    if base_style in ["rect", "chamfered_poly"]:
        if base_style == "rect": part = cq.Workplane("XY").rect(base_x, base_y).extrude(base_thick)
        else: part = cq.Workplane("XY").rect(base_x, base_y).edges("|Z").chamfer(random.uniform(5, 15)).extrude(base_thick)
            
        mx, my = base_x/2 - 12, base_y/2 - 12
        holes = [(-mx, my), (-mx, -my), (mx, my), (mx, -my)]
        if base_style == "chamfered_poly": holes = random.sample(holes, random.randint(2, 4))
        
        scallop_r_x, scallop_r_y = base_x * random.uniform(0.2, 0.4), base_y * random.uniform(0.2, 0.4)
        scallops = [(0, base_y/2 + scallop_r_y*0.2), (0, -base_y/2 - scallop_r_y*0.2), (base_x/2 + scallop_r_x*0.2, 0), (-base_x/2 - scallop_r_x*0.2, 0)]
        for sx, sy in scallops:
            part = part.cut(cq.Workplane("XY").center(sx, sy).circle(max(scallop_r_x, scallop_r_y)).extrude(base_thick*3, both=True))

        safe_dx, safe_dy = max(0, mx - root_radius), max(0, my - root_radius)
        offset_x, offset_y = random.uniform(-safe_dx, safe_dx), random.uniform(-safe_dy, safe_dy)

    elif base_style == "circle":
        r = max(base_x, base_y) / 2
        part = cq.Workplane("XY").circle(r).extrude(base_thick)
        num_holes = random.randint(3, 6)
        hr = r - 12
        for i in range(num_holes):
            theta = (i / num_holes) * 2 * math.pi
            holes.append((hr * math.cos(theta), hr * math.sin(theta)))
        safe_r = max(0, hr - root_radius)
        rand_angle = random.uniform(0, 2 * math.pi)
        rand_dist = random.uniform(0, safe_r)
        offset_x, offset_y = rand_dist * math.cos(rand_angle), rand_dist * math.sin(rand_angle)

    part = part.union(cq.Workplane("XY").center(offset_x, offset_y).circle(root_radius*1.1).extrude(pad_h))
    if holes:
        part = part.union(cq.Workplane("XY").pushPoints(holes).circle(hole_r + random.uniform(4, 7)).extrude(pad_h))
        part = part.cut(cq.Workplane("XY").pushPoints(holes).circle(hole_r).extrude(pad_h * 3, both=True))

    arm = arm.rotate((0,0,0), (0,0,1), yaw_angle).translate((offset_x, offset_y, base_thick - 1.5))
    part = part.union(arm, clean=True)
    return apply_professional_finishing(part)

# Decider
def generate_master_bracket(ui_params):
    arch = ui_params["bracket_type"]
    if arch == "Random":
        arch = random.choice(["Linear / Inline", "Asymmetric Radial", "Angled Extension"])
        
    if arch == "Linear / Inline": return build_linear_bracket(ui_params)
    elif arch == "Asymmetric Radial": return build_asymmetric_radial_bracket(ui_params)
    else: return build_angled_extension_bracket(ui_params)

# 6. TOPOLOGY OPTIMIZED GENERATOR
def create_topology_optimized_geometry(ui_params):
    SEED = None

    # Domain size in mm
    SIZE_X_MM = 300.0
    SIZE_Y_MM = 300.0
    SIZE_Z_MM = 300.0

    # --- UI PARAMETERS INJECTED HERE ---
    RESOLUTION = ui_params["resolution"]
    TARGET_VOLFRAC = ui_params["target_vol"]
    N_SUPPORTS = ui_params["n_supports"]
    N_LOADS = ui_params["n_loads"]
    N_EXTRA_BRANCHES = ui_params["n_branches"]
    MIN_FEATURE_SIZE_MM = ui_params["min_feature"]
    BASE_STRUT_RADIUS_MM = ui_params["strut_r"]
    OVERHANG_ANGLE_DEG = ui_params["overhang"]
    USE_SELF_SUPPORT_FILTER = ui_params["self_support"]
    # -----------------------------------
    
    RADIUS_RANDOMNESS_MM = 1.0
    NODE_BLOB_RADIUS_MM = BASE_STRUT_RADIUS_MM * 1.5 # Scaled relative to strut size

    FIELD_SMOOTHING_SIGMA = 0.8
    MESH_SMOOTHING_ITERATIONS = 15

    BUILD_DIRECTION = np.array([0.0, 0.0, 1.0])
    ALLOW_BASE_OVERHANG_HEIGHT_MM = 5.0

    MAX_GENERATION_ATTEMPTS = 20
    MAX_CAD_FACES = 25000
    VERBOSE = False # Turned off so it doesn't spam your Streamlit console
        
    #-Helper function
    def set_random_seed(seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)


    def make_grid(nx, ny, nz, sx, sy, sz):
        x = np.linspace(-sx / 2, sx / 2, nx)
        y = np.linspace(-sy / 2, sy / 2, ny)
        z = np.linspace(0, sz, nz)

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        dx = sx / (nx - 1)
        dy = sy / (ny - 1)
        dz = sz / (nz - 1)

        return X, Y, Z, dx, dy, dz


    def volume_fraction(binary):
        return float(np.mean(binary))


    def ball_structure(radius_vox):
        r = int(max(1, radius_vox))
        ax = np.arange(-r, r + 1)
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        return (X**2 + Y**2 + Z**2) <= r**2


    def disk_structure(radius_vox):
        r = int(max(1, radius_vox))
        ax = np.arange(-r, r + 1)
        X, Y = np.meshgrid(ax, ax, indexing="ij")
        return (X**2 + Y**2) <= r**2


    def keep_largest_component(binary):
        labeled, n = ndi.label(binary)

        if n == 0:
            return binary

        counts = np.bincount(labeled.ravel())
        counts[0] = 0

        largest = counts.argmax()
        return labeled == largest


    # ============================================================
    # RANDOM SUPPORTS AND LOADS
    # ============================================================

    def random_point_on_bottom(sx, sy):
        margin = 0.08
        return np.array([
            random.uniform(-sx / 2 * (1 - margin), sx / 2 * (1 - margin)),
            random.uniform(-sy / 2 * (1 - margin), sy / 2 * (1 - margin)),
            0.0
        ])


    def random_point_on_upper_region(sx, sy, sz):
        side = random.choice(["top", "side_x", "side_y"])

        if side == "top":
            return np.array([
                random.uniform(-sx * 0.46, sx * 0.46),
                random.uniform(-sy * 0.46, sy * 0.46),
                random.uniform(sz * 0.78, sz * 0.96)
            ])

        if side == "side_x":
            return np.array([
                random.choice([-1, 1]) * random.uniform(sx * 0.40, sx * 0.49),
                random.uniform(-sy * 0.42, sy * 0.42),
                random.uniform(sz * 0.45, sz * 0.90)
            ])

        return np.array([
            random.uniform(-sx * 0.42, sx * 0.42),
            random.choice([-1, 1]) * random.uniform(sy * 0.40, sy * 0.49),
            random.uniform(sz * 0.45, sz * 0.90)
        ])


    def random_load_vector():
        v = np.random.normal(size=3)
        v[2] -= abs(v[2]) * 0.5
        v = v / (np.linalg.norm(v) + 1e-12)
        magnitude = random.uniform(100.0, 1000.0)
        return magnitude * v


    # ============================================================
    # IMPLICIT GEOMETRY FUNCTIONS
    # ============================================================

    def distance_to_segment_field(X, Y, Z, a, b):
        ab = b - a
        ab2 = np.dot(ab, ab) + 1e-12

        apx = X - a[0]
        apy = Y - a[1]
        apz = Z - a[2]

        t = (apx * ab[0] + apy * ab[1] + apz * ab[2]) / ab2
        t = np.clip(t, 0.0, 1.0)

        cx = a[0] + t * ab[0]
        cy = a[1] + t * ab[1]
        cz = a[2] + t * ab[2]

        return np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)


    def add_tube_to_field(field, X, Y, Z, a, b, radius):
        d = distance_to_segment_field(X, Y, Z, a, b)
        contribution = radius - d
        np.maximum(field, contribution, out=field)


    def add_blob_to_field(field, X, Y, Z, center, radius):
        d = np.sqrt(
            (X - center[0])**2 +
            (Y - center[1])**2 +
            (Z - center[2])**2
        )
        contribution = radius - d
        np.maximum(field, contribution, out=field)


    def enforce_upward_self_supporting_path(points, max_angle_deg):
        """
        Makes the path grow upward and limits horizontal movement per vertical step.
        This helps create 45-degree printable branches.
        """
        angle = math.radians(max_angle_deg)
        max_horizontal_per_vertical = math.tan(angle)

        fixed = [points[0].copy()]

        for p in points[1:]:
            prev = fixed[-1].copy()
            p = p.copy()

            dz = max(p[2] - prev[2], 1e-6)

            horizontal = p[:2] - prev[:2]
            h_len = np.linalg.norm(horizontal)

            max_h = max_horizontal_per_vertical * dz

            if h_len > max_h:
                horizontal = horizontal / h_len * max_h
                p[:2] = prev[:2] + horizontal

            p[2] = max(p[2], prev[2] + 1e-3)
            fixed.append(p)

        return fixed


    def make_random_polyline(start, end, sx, sy, sz, n_mid=3):
        points = [start.copy()]

        for i in range(n_mid):
            t = (i + 1) / (n_mid + 1)

            p = (1 - t) * start + t * end

            p[0] += random.uniform(-0.12 * sx, 0.12 * sx)
            p[1] += random.uniform(-0.12 * sy, 0.12 * sy)

            p[0] = np.clip(p[0], -0.46 * sx, 0.46 * sx)
            p[1] = np.clip(p[1], -0.46 * sy, 0.46 * sy)
            p[2] = np.clip(p[2], 0.02 * sz, 0.98 * sz)

            points.append(p)

        points.append(end.copy())

        points = sorted(points, key=lambda p: p[2])
        points = enforce_upward_self_supporting_path(points, OVERHANG_ANGLE_DEG)

        return points


    # ============================================================
    # FIELD GENERATION
    # ============================================================

    def create_organic_field(X, Y, Z, sx, sy, sz):
        field = np.full(X.shape, -1e6, dtype=np.float32)

        supports = [random_point_on_bottom(sx, sy) for _ in range(N_SUPPORTS)]
        loads = [random_point_on_upper_region(sx, sy, sz) for _ in range(N_LOADS)]
        load_vectors = [random_load_vector() for _ in range(N_LOADS)]

        # Choose one root support so everything is connected.
        root = supports[0]

        all_nodes = []

        # Foot blobs
        for s in supports:
            c = s + np.array([0.0, 0.0, NODE_BLOB_RADIUS_MM * 0.45])
            add_blob_to_field(
                field, X, Y, Z,
                c,
                NODE_BLOB_RADIUS_MM * random.uniform(0.9, 1.3)
            )
            all_nodes.append(c)

        # Connect all supports to root close to the base.
        for s in supports[1:]:
            a = root + np.array([0.0, 0.0, NODE_BLOB_RADIUS_MM * 0.35])
            b = s + np.array([0.0, 0.0, NODE_BLOB_RADIUS_MM * 0.35])
            r = BASE_STRUT_RADIUS_MM * random.uniform(0.9, 1.3)
            add_tube_to_field(field, X, Y, Z, a, b, r)

        # Load blobs
        for l in loads:
            add_blob_to_field(
                field, X, Y, Z,
                l,
                NODE_BLOB_RADIUS_MM * random.uniform(0.7, 1.1)
            )
            all_nodes.append(l)

        # Main support-to-load branches
        for l in loads:
            s = random.choice(supports)

            path = make_random_polyline(
                s + np.array([0.0, 0.0, NODE_BLOB_RADIUS_MM * 0.3]),
                l,
                sx, sy, sz,
                n_mid=random.randint(2, 4)
            )

            for a, b in zip(path[:-1], path[1:]):
                r = BASE_STRUT_RADIUS_MM + random.uniform(
                    -RADIUS_RANDOMNESS_MM,
                    RADIUS_RANDOMNESS_MM
                )
                r = max(r, MIN_FEATURE_SIZE_MM / 2)
                add_tube_to_field(field, X, Y, Z, a, b, r)

            for p in path:
                all_nodes.append(p)
                if random.random() < 0.6:
                    add_blob_to_field(
                        field, X, Y, Z,
                        p,
                        NODE_BLOB_RADIUS_MM * random.uniform(0.35, 0.75)
                    )

        # Extra organic cross branches
        for _ in range(N_EXTRA_BRANCHES):
            a = random.choice(all_nodes)
            b = random.choice(all_nodes)

            if b[2] < a[2]:
                a, b = b, a

            if abs(b[2] - a[2]) < 0.12 * sz:
                continue

            path = make_random_polyline(
                a, b,
                sx, sy, sz,
                n_mid=random.randint(1, 3)
            )

            for p0, p1 in zip(path[:-1], path[1:]):
                r = BASE_STRUT_RADIUS_MM * random.uniform(0.55, 1.05)
                r = max(r, MIN_FEATURE_SIZE_MM / 2)
                add_tube_to_field(field, X, Y, Z, p0, p1, r)

        # Add smooth organic noise
        noise = np.random.normal(0.0, 1.0, size=field.shape)
        noise = ndi.gaussian_filter(noise, sigma=random.uniform(3.0, 5.5))
        noise = noise / (np.std(noise) + 1e-12)

        field += noise.astype(np.float32) * random.uniform(0.4, 1.2)

        return field, supports, loads, load_vectors


    def binary_from_field_with_target_volume(field, target_volfrac):
        """
        Uses an iso-level to approach the target volume fraction.

        Important:
        We do not allow too much positive thresholding, because that can cut
        the struts apart and create disconnected fragments.
        """
        q = np.quantile(field, 1.0 - target_volfrac)

        max_shrink_level = MIN_FEATURE_SIZE_MM * 0.20
        level = min(q, max_shrink_level)

        binary = field > level

        # If too little material appears, fall back to level 0.
        if volume_fraction(binary) < 0.035:
            binary = field > 0.0

        # If still too little, grow slightly.
        grow_iter = 0
        while volume_fraction(binary) < max(0.05, 0.45 * target_volfrac) and grow_iter < 3:
            binary = ndi.binary_dilation(binary, structure=ball_structure(1))
            grow_iter += 1

        return binary


    # ============================================================
    # LPBF SUPPORT FILTER
    # ============================================================

    def self_support_filter(binary, dx, dz, overhang_angle_deg):
        """
        Voxel-level approximation:
        material in each layer must be reachable from supported material below
        within a 45-degree cone.
        """
        angle = math.radians(overhang_angle_deg)
        allowed_horizontal_mm = math.tan(angle) * dz

        r_vox = max(1, int(math.ceil(allowed_horizontal_mm / dx)))
        disk = disk_structure(r_vox)

        supported = np.zeros_like(binary, dtype=bool)

        base_layers = max(1, int(math.ceil(ALLOW_BASE_OVERHANG_HEIGHT_MM / dz)))
        base_layers = min(base_layers, binary.shape[2])

        supported[:, :, :base_layers] = binary[:, :, :base_layers]

        for k in range(base_layers, binary.shape[2]):
            reachable = ndi.binary_dilation(supported[:, :, k - 1], structure=disk)
            supported[:, :, k] = binary[:, :, k] & reachable

        return supported


    # ============================================================
    # MESHING
    # ============================================================

    def binary_to_smooth_sdf(binary, sigma_vox):
        inside = ndi.distance_transform_edt(binary)
        outside = ndi.distance_transform_edt(~binary)

        sdf = inside - outside

        if sigma_vox > 0:
            sdf = ndi.gaussian_filter(sdf, sigma=sigma_vox)

        return sdf


    def mesh_from_binary(binary, dx, dy, dz):
        if binary.sum() == 0:
            raise ValueError("Cannot mesh empty binary geometry.")

        sdf = binary_to_smooth_sdf(binary, FIELD_SMOOTHING_SIGMA)

        # Padding prevents open surfaces at domain boundaries.
        sdf = np.pad(sdf, pad_width=1, mode="constant", constant_values=-10)

        verts, faces, normals, values = measure.marching_cubes(
            sdf,
            level=0.0,
            spacing=(dx, dy, dz)
        )

        # Remove padding offset
        verts -= np.array([dx, dy, dz])

        # Center X and Y around zero
        verts[:, 0] -= SIZE_X_MM / 2
        verts[:, 1] -= SIZE_Y_MM / 2

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

        # Keep largest mesh component
        comps = mesh.split(only_watertight=False)
        if len(comps) > 1:
            mesh = max(comps, key=lambda m: m.area)

        try:
            mesh.remove_unreferenced_vertices()
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
        except Exception:
            pass

        mesh.fix_normals()

        try:
            trimesh.smoothing.filter_taubin(
                mesh,
                lamb=0.5,
                nu=-0.53,
                iterations=MESH_SMOOTHING_ITERATIONS
            )
        except Exception:
            warnings.warn("Mesh smoothing failed, continuing without extra smoothing.")

        mesh.fix_normals()

        return mesh


    # ============================================================
    # CHECKS
    # ============================================================

    def approximate_min_feature_check(binary, dx, min_feature_mm):
        r_vox = max(1, int(math.ceil((min_feature_mm / 2) / dx)))
        opened = ndi.binary_opening(binary, structure=ball_structure(r_vox))

        original = binary.sum()

        if original == 0:
            return 1.0

        removed_fraction = 1.0 - opened.sum() / original
        return float(removed_fraction)


    def check_overhangs(mesh, overhang_angle_deg, base_ignore_height_mm):
        normals = mesh.face_normals
        centers = mesh.triangles_center
        areas = mesh.area_faces

        limit = -math.cos(math.radians(overhang_angle_deg))

        downward_bad = normals[:, 2] < limit
        not_base = centers[:, 2] > base_ignore_height_mm

        violation_mask = downward_bad & not_base

        total_area = np.sum(areas)
        violation_area = np.sum(areas[violation_mask])

        if total_area <= 0:
            violation_percent = 100.0
        else:
            violation_percent = 100.0 * violation_area / total_area

        return {
            "overhang_violation_faces": int(np.sum(violation_mask)),
            "overhang_violation_area_percent": float(violation_percent),
            "worst_normal_z": float(np.min(normals[:, 2]))
        }


    def check_mesh(mesh, binary, dx):
        comps = mesh.split(only_watertight=False)

        return {
            "mesh_faces": int(len(mesh.faces)),
            "mesh_vertices": int(len(mesh.vertices)),
            "is_watertight": bool(mesh.is_watertight),
            "is_volume": bool(mesh.is_volume),
            "connected_components": int(len(comps)),
            "voxel_volume_fraction": volume_fraction(binary),
            "small_feature_fraction_estimate": approximate_min_feature_check(
                binary,
                dx,
                MIN_FEATURE_SIZE_MM
            )
        }


    # ============================================================
    # FALLBACK GEOMETRY
    # ============================================================

    def create_fallback_field(X, Y, Z, sx, sy, sz):
        """
        Guaranteed simple connected printable-ish branching structure.
        Used only if all random attempts fail.
        """
        field = np.full(X.shape, -1e6, dtype=np.float32)

        root = np.array([0.0, 0.0, 0.0])
        trunk_top = np.array([0.0, 0.0, 0.75 * sz])

        add_blob_to_field(field, X, Y, Z, root + np.array([0, 0, 6]), 10.0)
        add_tube_to_field(field, X, Y, Z, root, trunk_top, BASE_STRUT_RADIUS_MM)

        top_points = [
            np.array([0.25 * sx, 0.20 * sy, 0.95 * sz]),
            np.array([-0.25 * sx, 0.20 * sy, 0.95 * sz]),
            np.array([0.20 * sx, -0.25 * sy, 0.92 * sz]),
            np.array([-0.20 * sx, -0.25 * sy, 0.92 * sz]),
        ]

        for p in top_points:
            add_tube_to_field(field, X, Y, Z, trunk_top, p, BASE_STRUT_RADIUS_MM * 0.9)
            add_blob_to_field(field, X, Y, Z, p, NODE_BLOB_RADIUS_MM)

        return field


    # ============================================================
    # GENERATION PIPELINE
    # ============================================================

    def generate_part():
        nx = ny = nz = RESOLUTION

        X, Y, Z, dx, dy, dz = make_grid(
            nx, ny, nz,
            SIZE_X_MM, SIZE_Y_MM, SIZE_Z_MM
        )

        best = None

        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            if VERBOSE:
                print(f"\nGeneration attempt {attempt}/{MAX_GENERATION_ATTEMPTS}")

            try:
                field, supports, loads, load_vectors = create_organic_field(
                    X, Y, Z,
                    SIZE_X_MM, SIZE_Y_MM, SIZE_Z_MM
                )

                binary_raw = binary_from_field_with_target_volume(
                    field,
                    TARGET_VOLFRAC
                )

                # Fill tiny holes and keep connected part.
                binary_raw = ndi.binary_closing(binary_raw, structure=ball_structure(1))
                binary_raw = ndi.binary_fill_holes(binary_raw)
                binary_raw = keep_largest_component(binary_raw)

                # Optional support filter.
                binary = binary_raw

                if USE_SELF_SUPPORT_FILTER:
                    filtered = self_support_filter(
                        binary_raw,
                        dx=dx,
                        dz=dz,
                        overhang_angle_deg=OVERHANG_ANGLE_DEG
                    )

                    filtered = keep_largest_component(filtered)

                    # Important robustness fix:
                    # If the support filter destroys almost everything,
                    # do not discard the whole attempt immediately.
                    if filtered.sum() > 0.25 * binary_raw.sum() and volume_fraction(filtered) > 0.025:
                        binary = filtered
                    else:
                        if VERBOSE:
                            print("Support filter was too destructive; using unfiltered connected geometry.")

                binary = keep_largest_component(binary)

                if binary.sum() == 0:
                    if VERBOSE:
                        print("Empty geometry after filtering. Retrying...")
                    continue

                mesh = mesh_from_binary(binary, dx, dy, dz)

                checks = check_mesh(mesh, binary, dx)
                overhangs = check_overhangs(
                    mesh,
                    OVERHANG_ANGLE_DEG,
                    ALLOW_BASE_OVERHANG_HEIGHT_MM
                )

                if VERBOSE:
                    print("Mesh checks:")
                    for k, v in checks.items():
                        print(f"  {k}: {v}")

                    print("Overhang checks:")
                    for k, v in overhangs.items():
                        print(f"  {k}: {v}")

                # Lower score is better.
                score = (
                    checks["connected_components"],
                    0 if checks["is_watertight"] else 1,
                    overhangs["overhang_violation_area_percent"],
                    abs(checks["voxel_volume_fraction"] - TARGET_VOLFRAC)
                )

                candidate = {
                    "mesh": mesh,
                    "binary": binary,
                    "checks": checks,
                    "overhangs": overhangs,
                    "score": score
                }

                if best is None or score < best["score"]:
                    best = candidate

                # Accept if good enough.
                if (
                    checks["connected_components"] == 1
                    and checks["is_watertight"]
                    and overhangs["overhang_violation_area_percent"] < 12.0
                ):
                    if VERBOSE:
                        print("Accepted geometry.")
                    return mesh, binary, checks, overhangs

            except Exception as e:
                if VERBOSE:
                    print(f"Attempt failed with error: {e}")
                continue

        # Critical fix:
        # If all random attempts failed, create a deterministic fallback instead
        # of doing best["mesh"] when best is None.
        if best is not None:
            warnings.warn(
                "Could not satisfy all checks perfectly. Returning the best generated result."
            )
            return best["mesh"], best["binary"], best["checks"], best["overhangs"]

        warnings.warn(
            "All random attempts failed. Creating guaranteed fallback geometry."
        )

        field = create_fallback_field(X, Y, Z, SIZE_X_MM, SIZE_Y_MM, SIZE_Z_MM)
        binary = binary_from_field_with_target_volume(field, TARGET_VOLFRAC)
        binary = ndi.binary_closing(binary, structure=ball_structure(1))
        binary = ndi.binary_fill_holes(binary)
        binary = keep_largest_component(binary)

        mesh = mesh_from_binary(binary, dx, dy, dz)
        checks = check_mesh(mesh, binary, dx)
        overhangs = check_overhangs(
            mesh,
            OVERHANG_ANGLE_DEG,
            ALLOW_BASE_OVERHANG_HEIGHT_MM
        )

        return mesh, binary, checks, overhangs


    # ============================================================
    # MESH SIMPLIFICATION
    # ============================================================

    def simplify_mesh_if_needed(mesh, max_faces):
        if len(mesh.faces) <= max_faces:
            return mesh

        if VERBOSE:
            print(f"Mesh has {len(mesh.faces)} faces. Trying to simplify to {max_faces} faces...")

        simplified = None

        try:
            simplified = mesh.simplify_quadric_decimation(face_count=max_faces)
        except Exception:
            pass

        if simplified is None:
            try:
                simplified = mesh.simplify_quadratic_decimation(max_faces)
            except Exception:
                pass

        if simplified is None:
            warnings.warn(
                "Mesh simplification failed. Try lowering RESOLUTION if CadQuery is slow."
            )
            return mesh

        simplified.fix_normals()
        return simplified


    # ============================================================
    # CADQUERY CONVERSION
    # ============================================================

    def mesh_to_cadquery_solid(mesh):
        """
        Converts triangular mesh to a CadQuery object.

        First tries to sew triangular faces into a solid.
        If that fails, it returns a CadQuery compound of triangular faces
        so that you still get a visible object instead of a crash.
        """
        cq_faces = []

        verts = mesh.vertices
        faces = mesh.faces

        for tri in faces:
            p0_np = verts[tri[0]]
            p1_np = verts[tri[1]]
            p2_np = verts[tri[2]]

            # Skip degenerate triangles
            area_vec = np.cross(p1_np - p0_np, p2_np - p0_np)
            if np.linalg.norm(area_vec) < 1e-9:
                continue

            try:
                p0 = cq.Vector(float(p0_np[0]), float(p0_np[1]), float(p0_np[2]))
                p1 = cq.Vector(float(p1_np[0]), float(p1_np[1]), float(p1_np[2]))
                p2 = cq.Vector(float(p2_np[0]), float(p2_np[1]), float(p2_np[2]))

                wire = cq.Wire.makePolygon([p0, p1, p2], close=True)
                face = cq.Face.makeFromWires(wire)
                cq_faces.append(face)

            except Exception:
                continue

        if len(cq_faces) == 0:
            raise RuntimeError("CadQuery conversion failed: no valid triangular faces.")

        try:
            shell = cq.Shell.makeShell(cq_faces)
            solid = cq.Solid.makeSolid(shell)
            return cq.Workplane("XY").add(solid)

        except Exception:
            warnings.warn(
                "Could not sew mesh into a true CadQuery solid. "
                "Returning a CadQuery face compound for visualization."
            )
            compound = cq.Compound.makeCompound(cq_faces)
            return cq.Workplane("XY").add(compound)

    set_random_seed(SEED)

    mesh, binary, checks, overhangs = generate_part()
    mesh = simplify_mesh_if_needed(mesh, MAX_CAD_FACES)

    # Convert the Trimesh to CadQuery so the UI can export it normally!
    part = mesh_to_cadquery_solid(mesh)
    
    return part


# 5. STREAMLIT USER INTERFACE & RANDOMIZER
st.set_page_config(page_title="LPBF Generator", layout="wide")
st.title("⚙️ LPBF Geometry Dataset Master Generator")

# 1. Randomizer, multiple or singular
st.sidebar.header("1. Generation Mode")
gen_mode = st.sidebar.radio("Select Mode:", ["Generate Single Architecture", "Random Weighted Mix"])

target_arch = None
w_hol, w_fin, w_hx, w_brk, w_top = 1.0, 1.0, 1.0, 1.0, 1.0

if gen_mode == "Generate Single Architecture":
    target_arch = st.sidebar.selectbox("Choose Part Type:", ["Thick Hollow Geometry", "Fin Architecture", "Heat Exchanger", "Brackets", "Topology Optimized" ])
else:
    st.sidebar.subheader("⚖️ Random Weights")
    st.sidebar.caption("Higher numbers mean that architecture is picked more often. Set to 0 to disable an architecture entirely.")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        w_hol = st.number_input("Hollow", value=1.0, min_value=0.0, step=0.5)
        w_fin = st.number_input("Fins", value=1.0, min_value=0.0, step=0.5)
        w_top = st.number_input("Topo", value=0.0, min_value=0.0, step=0.5)
    with col2:
        w_hx = st.number_input("Heat Exch.", value=1.0, min_value=0.0, step=0.5)
        w_brk = st.number_input("Brackets", value=5.0, min_value=0.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("2. Architecture Parameters")
st.sidebar.caption("Settings below are saved automatically. Open a tab to adjust its specific bounds before generating.")

# Holding Parameters
ui_params = {"Hollow": {}, "Fin": {}, "Heat": {}, "Brackets": {}, "Topo": {}}

# Hollow Geometry Parameters
with st.sidebar.expander("🔲 Thick Hollow Geometry", expanded=(target_arch == "Thick Hollow Geometry")):
    ui_params["Hollow"]["height"] = st.slider("Total Height Range (mm)", 20, 200, (50, 100), 5, key="h1")
    ui_params["Hollow"]["wall_thick"] = st.slider("Wall Thickness Range (mm)", 1.0, 15.0, (5.0, 10.0), 0.5, key="h2")
    ui_params["Hollow"]["radius"] = st.slider("Base Radius Range (mm)", 10, 100, (20, 45), key="h3")
    ui_params["Hollow"]["points"] = st.slider("Number of Points Range", 5, 25, (8, 15), key="h4")
    ui_params["Hollow"]["sections"] = st.slider("Vertical Sections Range", 1, 6, (1, 4), key="h5")
    ui_params["Hollow"]["straight_lines"] = not st.toggle("Smooth Splines (Toggle off for straight)", value=True, key="h6")
    ui_params["Hollow"]["smoothing"] = 1 if st.toggle("Vertical Smoothing", value=True, key="h7") else 0
    ui_params["Hollow"]["roof_count"] = st.slider("Number of Roof Features", 0, 8, (2, 5), key="h8")
    ui_params["Hollow"]["wall_count"] = st.slider("Number of Thin Walls", 0, 5, (0, 1), key="h9")
    ui_params["Hollow"]["wall_t"] = st.slider("Thin Wall Thickness Range (mm)", 0.1, 5.0, (0.5, 1.5), 0.1, key="h10")
    ui_params["Hollow"]["wall_l"] = st.slider("Thin Wall Length Range (mm)", 5.0, 50.0, (10.0, 30.0), 1.0, key="h11")
    ui_params["Hollow"]["wall_h"] = st.slider("Thin Wall Outward Height Range (mm)", 1.0, 20.0, (5.0, 10.0), 1.0, key="h12")

# Fin Architecture Parameters
with st.sidebar.expander("🦈 Fin Architecture", expanded=(target_arch == "Fin Architecture")):
    ui_params["Fin"]["height"] = st.slider("Total Height Range (mm)", 20.0, 200.0, (80.0, 150.0), 5.0, key="f1")
    ui_params["Fin"]["wall_thick"] = st.slider("Core Wall Thickness Range (mm)", 1.0, 10.0, (2.0, 5.0), 0.5, key="f2")
    ui_params["Fin"]["n_points"] = st.slider("Vertical Core Layers Range", 2, 15, (3, 7), key="f3")
    ui_params["Fin"]["smooth"] = st.toggle("Smooth Core Spline", value=True, key="f4")
    ui_params["Fin"]["n_symmetry"] = st.slider("Number of Fins Range", 3, 36, (6, 12), key="f5")
    ui_params["Fin"]["blade_l"] = st.slider("Blade Length Range (mm)", 5.0, 60.0, (15.0, 30.0), 1.0, key="f6")
    ui_params["Fin"]["blade_t"] = st.slider("Blade Thickness Range (mm)", 0.5, 10.0, (1.0, 3.0), 0.1, key="f7")
    ui_params["Fin"]["fin_type"] = st.selectbox("Fin Profile Style", ["Random", "Rectangle", "Triangle", "Ellipse", "Polygon"], key="f8")
    ui_params["Fin"]["path_style"] = st.selectbox("Fin Sweep Path", ["Random", "Linear", "Bow", "S-Curve"], key="f9")

# Heat Exchanger Parameters
with st.sidebar.expander("🌊 Heat Exchanger", expanded=(target_arch == "Heat Exchanger")):
    ui_params["Heat"]["length"] = st.slider("Total Length Range (mm)", 20.0, 200.0, (40.0, 150.0), 5.0, key="hx1")
    ui_params["Heat"]["width"] = st.slider("Nominal Width Range (mm)", 15.0, 100.0, (20.0, 50.0), 2.0, key="hx2")
    ui_params["Heat"]["height"] = st.slider("Base Fin Height Range (mm)", 10.0, 100.0, (10.0, 50.0), 2.0, key="hx3")
    ui_params["Heat"]["shroud_t"] = st.slider("Outer Shroud Thickness (mm)", 1.0, 10.0, (1.0, 5.0), 0.5, key="hx4")
    ui_params["Heat"]["macro_layout"] = st.selectbox("Macro Layout", ["Random", "Venturi", "S-Curve", "Tapered"], key="hx5")
    ui_params["Heat"]["fin_t"] = st.slider("Fin Thickness Range (mm)", 0.2, 5.0, (0.4, 1.8), 0.1, key="hx6")
    ui_params["Heat"]["pitch"] = st.slider("Target Pitch Range (Spacing)", 1.0, 15.0, (2.0, 8.0), 0.5, key="hx7")
    ui_params["Heat"]["fin_pattern"] = st.selectbox("Internal Fin Pattern", ["Parallel to Shroud", "Chaotic Waves"], key="hx8")
    ui_params["Heat"]["enable_cuts"] = st.toggle("Enable Horizontal Cuts", value=True, key="hx9")
    ui_params["Heat"]["num_cuts"] = st.slider("Number of Horizontal Cuts", 1, 12, (2, 6), key="hx10")
    ui_params["Heat"]["top_style"] = st.selectbox("Top Cut Style", ["Random", "Smooth Wave", "Faceted Stepped"], key="hx11")
    ui_params["Heat"]["top_segments"] = st.slider("Top Cut Control Points", 2, 15, (5, 8), key="hx12")

# Brackets Parameters
with st.sidebar.expander("🦾 Brackets", expanded=(target_arch == "Brackets")):
    ui_params["Brackets"]["bracket_type"] = st.selectbox("Bracket Architecture", ["Random", "Linear / Inline", "Asymmetric Radial", "Angled Extension"], key="b1")
    ui_params["Brackets"]["base_t"] = st.slider("Base Membrane Thickness Range (mm)", 1.0, 15.0, (3.0, 5.0), 0.5, key="b2")
    ui_params["Brackets"]["pocket"] = st.selectbox("Internal Pocket Style", ["Random", "Webbed", "Hollow-Frame"], key="b3")
    ui_params["Brackets"]["web_t"] = st.slider("Web Thickness (If Webbed)", 1.0, 12.0, (3.0, 7.0), 0.5, key="b4")
    ui_params["Brackets"]["pocket_wall"] = st.slider("Outer Wall Thickness", 2.0, 15.0, (4.0, 8.0), 0.5, key="b5")

    st.markdown("---")
    ui_params["Brackets"]["lin_type"] = st.selectbox("(Linear) Shape Override", ["Random", "L-Bracket", "Inline"], key="b6")
    ui_params["Brackets"]["lin_h"] = st.slider("(Linear) Bracket Height Range", 40.0, 250.0, (40.0, 200.0), 5.0, key="b7")
    ui_params["Brackets"]["lin_base_y"] = st.slider("(Linear) Base Depth Range", 20.0, 150.0, (30.0, 90.0), 5.0, key="b8")
    ui_params["Brackets"]["lin_boss_l"] = st.slider("(Linear) Top Boss Length Range", 10.0, 80.0, (10.0, 40.0), 2.0, key="b9")
    ui_params["Brackets"]["lin_taper"] = st.slider("(Linear) Wall Taper Angle Range", 45.0, 85.0, (55.0, 80.0), 1.0, key="b10")
    ui_params["Brackets"]["lin_base_wavy"] = st.toggle("(Linear) Wavy Base Cut Out", value=True, key="b11")
    ui_params["Brackets"]["lin_diag_wavy"] = st.toggle("(Linear) Wavy Diagonal Flange", value=True, key="b12")
    ui_params["Brackets"]["lin_hole_shape"] = st.selectbox("(Linear) Hole Shape", ["Random", "Diamond", "Hex", "Teardrop", "Elongated Diamond"], key="b13")
    ui_params["Brackets"]["lin_hole_density"] = st.select_slider("(Linear) Hole Packing", options=["Sparse", "Normal", "Dense"], value="Normal", key="b14")

    st.markdown("---")
    ui_params["Brackets"]["rad_layout"] = st.selectbox("(Radial) Arm Layout", ["Random", "V-Corner (2 Arms)", "Y-Junction (3 Arms)", "Offset-T (3 Arms)", "Star-Cross (4 Arms)"], key="b15")
    ui_params["Brackets"]["rad_hub"] = st.selectbox("(Radial) Hub Override", ["Random", "Cylinder", "Hexagon", "Square", "Hollow Collar"], key="b16")
    ui_params["Brackets"]["rad_hub_h"] = st.slider("(Radial) Hub Height Range", 30.0, 200.0, (40.0, 140.0), 5.0, key="b17")
    ui_params["Brackets"]["rad_boss_r"] = st.slider("(Radial) Boss Radius Range", 5.0, 60.0, (10.0, 30.0), 2.0, key="b18")
    ui_params["Brackets"]["rad_base_wavy"] = st.toggle("(Radial) Wavy Base Cut Out", value=True, key="b19")
    ui_params["Brackets"]["rad_diag_wavy"] = st.toggle("(Radial) Wavy Diagonal Flange", value=True, key="b20")
    ui_params["Brackets"]["rad_hole_shape"] = st.selectbox("(Radial) Hole Shape", ["Random", "Diamond", "Hex", "Teardrop", "Elongated Diamond"], key="b21")
    ui_params["Brackets"]["rad_hole_density"] = st.select_slider("(Radial) Hole Packing", options=["Sparse", "Normal", "Dense"], value="Normal", key="b22")

    st.markdown("---")
    ui_params["Brackets"]["ang_base"] = st.selectbox("(Angled) Base Plate Override", ["Random", "Rect", "Circle", "Chamfered Poly"], key="b23")
    ui_params["Brackets"]["ang_base_pad"] = st.slider("(Angled) Base Padding Range", 10.0, 100.0, (30.0, 60.0), 5.0, key="b24")
    ui_params["Brackets"]["ang_arm_h"] = st.slider("(Angled) Arm Height Range", 30.0, 250.0, (40.0, 180.0), 5.0, key="b25")
    ui_params["Brackets"]["ang_arm_l"] = st.slider("(Angled) Base Length Range", 10.0, 120.0, (20.0, 80.0), 5.0, key="b26")
    ui_params["Brackets"]["ang_arm_w"] = st.slider("(Angled) Arm Width Range", 5.0, 80.0, (10.0, 50.0), 2.0, key="b27")
    ui_params["Brackets"]["ang_lean"] = st.slider("(Angled) Lean Angle Range", 45.0, 85.0, (47.0, 80.0), 1.0, key="b28")
    ui_params["Brackets"]["ang_wavy"] = st.toggle("(Angled) Wavy Flange Cuts", value=True, key="b29")
    ui_params["Brackets"]["ang_serrations"] = st.toggle("(Angled) Rivety Top", value=True, key="b30")
    ui_params["Brackets"]["ang_hole_shape"] = st.selectbox("(Angled) Hole Shape", ["Random", "Diamond", "Hex", "Teardrop", "Elongated Diamond"], key="b31")
    ui_params["Brackets"]["ang_hole_density"] = st.select_slider("(Angled) Hole Packing", options=["Sparse", "Normal", "Dense"], value="Normal", key="b32")

# Topology Optimizer Parameters
with st.sidebar.expander("🦴 Topology Optimized", expanded=(target_arch == "Topology Optimized")):
    ui_params["Topo"]["target_vol"] = st.slider("Target Volume Fraction", 0.01, 0.20, 0.06, 0.01, key="t1")
    ui_params["Topo"]["resolution"] = st.select_slider("Grid Resolution (Higher = Slower)", options=[32, 64, 96, 128], value=64, key="t2")
    
    st.markdown("---")
    ui_params["Topo"]["n_supports"] = st.slider("Number of Supports", 1, 10, 3, key="t3")
    ui_params["Topo"]["n_loads"] = st.slider("Number of Loads", 1, 10, 5, key="t4")
    ui_params["Topo"]["n_branches"] = st.slider("Extra Branches", 0, 10, 4, key="t5")
    
    st.markdown("---")
    ui_params["Topo"]["min_feature"] = st.slider("Min Feature Size (mm)", 1.0, 10.0, 3.0, 0.5, key="t6")
    ui_params["Topo"]["strut_r"] = st.slider("Base Strut Radius (mm)", 1.0, 10.0, 4.0, 0.5, key="t7")
    
    st.markdown("---")
    ui_params["Topo"]["overhang"] = st.slider("LPBF Overhang Limit (Deg)", 30.0, 60.0, 45.0, 1.0, key="t8")
    ui_params["Topo"]["self_support"] = st.toggle("Enforce LPBF Self-Supporting Rules", value=True, key="t9")

# Buttons and Interface 
st.write("### Ready to Generate")


col1, col2, col3 = st.columns([5, 2, 1])
with col3:
    batch_size = st.number_input("Batch Size", min_value=2, max_value=500, value=10, step=1, label_visibility="collapsed")

with col1:
    generate_single = st.button("🚀 Generate Single Part", type="primary", use_container_width=True)
    
with col2:
    generate_batch = st.button(f"📦 Batch ({batch_size})", type="secondary", use_container_width=True)


# Single generation
if generate_single:
    if gen_mode == "Random Weighted Mix":
        choices = ["Thick Hollow Geometry", "Fin Architecture", "Heat Exchanger", "Brackets", "Topology Optimized"]
        weights = [w_hol, w_fin, w_hx, w_brk, w_top]
        if sum(weights) == 0:
            st.error("Weights cannot all be zero! Please add weight to at least one architecture.")
            st.stop()
        active_arch = random.choices(choices, weights=weights, k=1)[0]
        st.info(f"🎲 Randomizer Selected: **{active_arch}**")
    else:
        active_arch = target_arch
        
    with st.spinner(f"Calculating complex geometry for {active_arch}..."):
        try:
            if active_arch == "Thick Hollow Geometry": final_part = create_shape(ui_params["Hollow"])
            elif active_arch == "Fin Architecture": final_part = create_fin_architecture(ui_params["Fin"])
            elif active_arch == "Heat Exchanger": final_part = create_advanced_chaotic_heat_exchanger(ui_params["Heat"])
            elif active_arch == "Brackets": final_part = generate_master_bracket(ui_params["Brackets"])
            elif active_arch == "Topology Optimized": final_part = create_topology_optimized_geometry(ui_params["Topo"])
            
            export_path = "generated_part.stl"
            cq.exporters.export(final_part, export_path)
            st.success("✅ Part generated successfully!")
            
            # --- COLOR DICTIONARY ---
            color_map = {
                "Thick Hollow Geometry": "#6ffffa", 
                "Fin Architecture": "#53ff6d",     
                "Heat Exchanger": "#fff956",        
                "Brackets": "#FF9900",               # 
                "Topology Optimized": "#dc97ff"
            }
            part_color = color_map.get(active_arch, "#FF9900") # Default to Orange if missing
            
            st.write("### 3D Preview")
            stl_from_file(file_path=export_path, color=part_color, material='material', shininess=0)
            
            with open(export_path, "rb") as file:
                st.download_button(
                    label="💾 Download STL File",
                    data=file,
                    file_name=f"LPBF_{active_arch.replace(' ', '_')}.stl",
                    mime="application/octet-stream"
                )
        except Exception as e:
            st.error(f"Generation failed: {e}")


# Batch generation into folder
if generate_batch:
    with st.spinner(f"🏭 Generating {batch_size} parts... (This might take a while!)"):
        try:
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                # Loop uses the variable instead of a hardcoded 10
                for i in range(1, batch_size + 1):
                    
                    if gen_mode == "Random Weighted Mix":
                        active_arch = random.choices(
                            ["Thick Hollow Geometry", "Fin Architecture", "Heat Exchanger", "Brackets"], 
                            weights=[w_hol, w_fin, w_hx, w_brk], k=1
                        )[0]
                    else:
                        active_arch = target_arch
                        
                    if active_arch == "Thick Hollow Geometry": final_part = create_shape(ui_params["Hollow"])
                    elif active_arch == "Fin Architecture": final_part = create_fin_architecture(ui_params["Fin"])
                    elif active_arch == "Heat Exchanger": final_part = create_advanced_chaotic_heat_exchanger(ui_params["Heat"])
                    elif active_arch == "Brackets": final_part = generate_master_bracket(ui_params["Brackets"])
                    
                    temp_filename = f"LPBF_{active_arch.replace(' ', '_')}_{i}.stl"
                    cq.exporters.export(final_part, temp_filename)
                    zip_file.write(temp_filename)
                    os.remove(temp_filename)
            
            st.session_state['batch_zip'] = zip_buffer.getvalue()
            # Save the size of the batch to session state so the download button knows what to say
            st.session_state['last_batch_size'] = batch_size 
            st.success(f"✅ Batch of {batch_size} parts generated and zipped successfully!")
            
        except Exception as e:
            st.error(f"Batch generation failed on part {i}: {e}")

# Zip folder download
if 'batch_zip' in st.session_state:
    last_size = st.session_state.get('last_batch_size', batch_size)
    st.write(f"### 📦 Batch of {last_size} Ready")
    st.download_button(
        label=f"💾 Download {last_size}-Part ZIP File",
        data=st.session_state['batch_zip'],
        file_name=f"LPBF_Dataset_Batch_{last_size}.zip",
        mime="application/zip",
        type="primary"
    )