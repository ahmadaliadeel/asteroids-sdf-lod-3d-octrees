"""
3D Volume Texture SDF Ray Marching with Octree LOD
Renders a sphere signed distance field using ray marching through octree-organized 3D textures.
Each octree node has border cells for seamless neighbor sampling.
"""

import numpy as np
import wgpu
import glfw
from rendercanvas.glfw import RenderCanvas, loop
from dataclasses import dataclass
from typing import Optional

# Volume texture resolution per node (excluding borders)
VOLUME_SIZE = 16
# Border size for neighbor sampling (1 cell on each side)
BORDER_SIZE = 1
# Total texture size including borders
TEXTURE_SIZE = VOLUME_SIZE + 2 * BORDER_SIZE
# Maximum octree depth (5 gives high detail near surface)
MAX_DEPTH = 5
# Maximum number of octree nodes (for GPU buffer allocation)
MAX_NODES = 2048


@dataclass
class OctreeNode:
    """Represents a node in the octree volume hierarchy."""
    # World-space bounds
    center: np.ndarray  # vec3
    half_size: float
    # Tree structure
    depth: int
    index: int  # Index in the flat node array for GPU
    parent_index: int
    children_start: int  # Index of first child (-1 if leaf)
    # Volume data
    volume_data: Optional[np.ndarray] = None  # 3D SDF data with borders
    is_leaf: bool = True
    
    @property
    def min_corner(self) -> np.ndarray:
        return self.center - self.half_size
    
    @property
    def max_corner(self) -> np.ndarray:
        return self.center + self.half_size


def sphere_sdf(p: np.ndarray, center: np.ndarray, radius: float) -> float:
    """Compute sphere SDF at point p."""
    return np.linalg.norm(p - center) - radius


def sample_sdf_world(p: np.ndarray) -> float:
    """
    Sample the world SDF at a point. This is the "ground truth" SDF.
    Sphere with sine displacement at origin.
    """
    sphere_radius = 0.5
    dist = np.linalg.norm(p) - sphere_radius
    
    # Add sine displacement for visual interest
    freq = 8.0
    amplitude = 0.05
    displacement = amplitude * np.sin(p[0] * freq) * np.sin(p[1] * freq) * np.sin(p[2] * freq)
    
    return dist + displacement


def sample_sdf_world_vectorized(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Vectorized version of sample_sdf_world for fast volume generation.
    x, y, z are 3D arrays of coordinates.
    """
    sphere_radius = 0.4
    dist = np.sqrt(x**2 + y**2 + z**2) - sphere_radius
    
    # Add sine displacement for visual interest
    freq = 20.0
    amplitude = 0.075
    displacement = amplitude * np.sin(x * freq) * np.sin(y * freq) * np.sin(z * freq)
    
    return dist + displacement


def should_subdivide(node: OctreeNode, threshold: float = 0.1) -> bool:
    """
    Determine if a node should be subdivided based on SDF variation.
    Subdivide if the surface passes through this node.
    """
    if node.depth >= MAX_DEPTH:
        return False
    
    # Sample corners and center
    corners = []
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            for dz in [-1, 1]:
                corner = node.center + np.array([dx, dy, dz]) * node.half_size
                corners.append(sample_sdf_world(corner))
    
    center_sdf = sample_sdf_world(node.center)
    
    # Check if surface crosses this volume (sign change)
    min_sdf = min(min(corners), center_sdf)
    max_sdf = max(max(corners), center_sdf)
    
    # Only subdivide if surface actually crosses this volume
    surface_crosses = min_sdf < 0 and max_sdf > 0
    
    return surface_crosses


def is_node_near_surface(node: OctreeNode, threshold_factor: float = 2.0) -> bool:
    """
    Check if the surface passes through or near this node.
    Returns False for nodes entirely inside or outside the surface.
    """
    # Sample corners and center
    samples = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                pos = node.center + np.array([dx, dy, dz], dtype=np.float32) * node.half_size
                samples.append(sample_sdf_world(pos))
    
    min_sdf = min(samples)
    max_sdf = max(samples)
    
    # Surface crosses if sign change
    if min_sdf < 0 and max_sdf > 0:
        return True
    
    # Surface is near if closest distance is less than node diagonal
    threshold = node.half_size * threshold_factor * np.sqrt(3)
    return abs(min_sdf) < threshold or abs(max_sdf) < threshold


def generate_node_volume(node: OctreeNode) -> np.ndarray:
    """
    Generate 3D SDF volume data for a node, including border cells.
    Border cells extend beyond the node's bounds to enable seamless sampling.
    Uses vectorized numpy operations for speed.
    """
    # Create coordinate arrays for all voxels at once
    indices = np.arange(TEXTURE_SIZE, dtype=np.float32)
    
    # Map voxel indices to normalized coordinates
    # Border cells are at indices 0 and (TEXTURE_SIZE-1)
    # Interior cells are at indices 1 to (TEXTURE_SIZE-2)
    uvw = (indices - BORDER_SIZE + 0.5) / VOLUME_SIZE
    
    # Create 3D meshgrid
    u, v, w = np.meshgrid(uvw, uvw, uvw, indexing='ij')
    
    # Convert to world positions
    x = node.center[0] + (u - 0.5) * 2 * node.half_size
    y = node.center[1] + (v - 0.5) * 2 * node.half_size
    z = node.center[2] + (w - 0.5) * 2 * node.half_size
    
    # Sample SDF vectorized
    volume = sample_sdf_world_vectorized(x, y, z)
    
    return volume.astype(np.float32)


def build_octree(center: np.ndarray, half_size: float) -> list[OctreeNode]:
    """
    Build an octree with adaptive subdivision based on SDF.
    Uses breadth-first construction to ensure children are contiguous.
    Returns a flat list of all nodes.
    """
    nodes = []
    leaf_count = [0]
    
    # Create root node
    root = OctreeNode(
        center=center,
        half_size=half_size,
        depth=0,
        index=0,
        parent_index=-1,
        children_start=-1,
    )
    nodes.append(root)
    
    # Process queue for breadth-first traversal
    queue = [0]  # Start with root index
    volume_count = [0]
    skipped_count = [0]
    
    while queue:
        node_idx = queue.pop(0)
        node = nodes[node_idx]
        
        # Only generate volume data for nodes near the surface
        if is_node_near_surface(node):
            node.volume_data = generate_node_volume(node)
            volume_count[0] += 1
            if volume_count[0] % 20 == 0:
                print(f"  Generated {volume_count[0]} volumes (skipped {skipped_count[0]} empty)...")
        else:
            node.volume_data = None
            skipped_count[0] += 1
        
        if not should_subdivide(node):
            # Leaf node - no children
            node.is_leaf = True
            continue
        
        # Subdivide into 8 children - add them all contiguously
        node.is_leaf = False
        node.children_start = len(nodes)
        child_half_size = node.half_size / 2
        
        for i in range(8):
            # Child octant based on index bits: x=bit2, y=bit1, z=bit0
            dx = ((i >> 2) & 1) * 2 - 1  # -1 or +1
            dy = ((i >> 1) & 1) * 2 - 1
            dz = (i & 1) * 2 - 1
            
            child_center = node.center + np.array([dx, dy, dz], dtype=np.float32) * child_half_size
            child = OctreeNode(
                center=child_center,
                half_size=child_half_size,
                depth=node.depth + 1,
                index=len(nodes),
                parent_index=node_idx,
                children_start=-1,
            )
            nodes.append(child)
            queue.append(child.index)
    
    return nodes


# WGSL Shader for octree-based ray marching through 3D SDF volumes
SHADER_CODE = """
struct Uniforms {
    resolution: vec2f,
    time: f32,
    num_nodes: u32,
    max_lod_depth: u32,
    volumes_per_side: u32,
    lod_distance_scale: f32,  // Controls how aggressively LOD drops with distance
    use_distance_lod: u32,    // 0 = fixed LOD, 1 = distance-based LOD
    camera_pos: vec3f,
    _pad: f32,
}

struct OctreeNode {
    center: vec3f,
    half_size: f32,
    children_start: i32,  // -1 if leaf
    volume_index: i32,    // Index into volume texture array (-1 if not leaf)
    depth: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> nodes: array<OctreeNode>;
@group(0) @binding(2) var volume_textures: texture_3d<f32>;
@group(0) @binding(3) var volume_sampler: sampler;

const VOLUME_SIZE: f32 = """ + str(VOLUME_SIZE) + """.0;
const BORDER_SIZE: f32 = """ + str(BORDER_SIZE) + """.0;
const TEXTURE_SIZE: f32 = """ + str(TEXTURE_SIZE) + """.0;
const MAX_DEPTH: u32 = """ + str(MAX_DEPTH) + """u;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

// Full-screen triangle vertices
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0)
    );
    
    var output: VertexOutput;
    let pos = positions[vertex_index];
    output.position = vec4f(pos, 0.0, 1.0);
    output.uv = pos * 0.5 + 0.5;
    return output;
}

// Check if point is inside node bounds
fn point_in_node(p: vec3f, node_idx: u32) -> bool {
    let node = nodes[node_idx];
    let d = abs(p - node.center);
    return all(d <= vec3f(node.half_size));
}

// Calculate LOD depth based on distance from camera
fn get_distance_lod(p: vec3f) -> u32 {
    if (uniforms.use_distance_lod == 0u) {
        return uniforms.max_lod_depth;
    }
    
    let dist = length(p - uniforms.camera_pos);
    
    // Scale: closer = higher LOD, farther = lower LOD
    // LOD = max_depth - floor(dist * scale)
    // At dist=0, full detail. As dist increases, LOD decreases.
    let lod_reduction = u32(floor(dist * uniforms.lod_distance_scale));
    
    if (lod_reduction >= uniforms.max_lod_depth) {
        return 0u;  // Minimum detail
    }
    
    return uniforms.max_lod_depth - lod_reduction;
}

// Find the leaf node containing point p, respecting distance-based LOD
fn find_leaf_node(p: vec3f) -> i32 {
    var current: i32 = 0;
    
    // Get LOD depth based on distance
    let target_depth = get_distance_lod(p);
    
    for (var depth = 0u; depth < MAX_DEPTH + 1u; depth++) {
        let node = nodes[current];
        
        // Check if this is a leaf OR we've reached LOD limit for this distance
        if (node.children_start < 0 || node.depth >= target_depth) {
            return current;
        }
        
        // Determine which child octant contains the point
        let offset = step(node.center, p);  // 0 or 1 for each axis
        let child_idx = i32(offset.x) * 4 + i32(offset.y) * 2 + i32(offset.z);
        current = node.children_start + child_idx;
        
        if (current >= i32(uniforms.num_nodes)) {
            return -1;
        }
    }
    
    return current;
}

// Sample SDF from a specific node's volume
fn sample_node_sdf(p: vec3f, node_idx: i32) -> f32 {
    if (node_idx < 0 || node_idx >= i32(uniforms.num_nodes)) {
        return 1.0;
    }
    
    let node = nodes[node_idx];
    
    // Transform world position to local UV coordinates [0, 1]
    let local_p = (p - node.center) / (node.half_size * 2.0) + 0.5;
    
    // Account for border cells in texture coordinates
    // Interior region is from BORDER_SIZE/TEXTURE_SIZE to (TEXTURE_SIZE-BORDER_SIZE)/TEXTURE_SIZE
    let border_offset = BORDER_SIZE / TEXTURE_SIZE;
    let interior_scale = VOLUME_SIZE / TEXTURE_SIZE;
    let tex_coord = local_p * interior_scale + border_offset;
    
    // Calculate which slice in the 3D texture array based on node's volume index
    let volume_idx = node.volume_index;
    if (volume_idx < 0) {
        return 1.0;  // Not a leaf node
    }
    
    // Sample from the combined 3D texture (volumes arranged in 3D grid)
    let vps = uniforms.volumes_per_side;
    let vol_x = u32(volume_idx) % vps;
    let vol_y = (u32(volume_idx) / vps) % vps;
    let vol_z = u32(volume_idx) / (vps * vps);
    
    let atlas_coord = vec3f(
        (f32(vol_x) + tex_coord.x) / f32(vps),
        (f32(vol_y) + tex_coord.y) / f32(vps),
        (f32(vol_z) + tex_coord.z) / f32(vps)
    );
    
    return textureSampleLevel(volume_textures, volume_sampler, atlas_coord, 0.0).r;
}

// Analytical SDF for debugging
fn analytical_sdf(p: vec3f) -> f32 {
    let sphere_radius = 0.5;
    let dist = length(p) - sphere_radius;
    let freq = 8.0;
    let amplitude = 0.05;
    let displacement = amplitude * sin(p.x * freq) * sin(p.y * freq) * sin(p.z * freq);
    return dist + displacement;
}

// Sample the SDF with octree traversal
fn sample_sdf(p: vec3f) -> f32 {
    // Check if point is inside the octree bounds (root node is at origin with half_size 1.0)
    if (any(abs(p) > vec3f(1.0))) {
        return analytical_sdf(p);  // Fallback outside octree
    }
    
    let node_idx = find_leaf_node(p);
    if (node_idx < 0) {
        return analytical_sdf(p);  // Fallback if node not found
    }
    
    // Check if this node has volume data (volume_index >= 0)
    let node = nodes[node_idx];
    if (node.volume_index < 0) {
        return analytical_sdf(p);  // Empty node, use analytical
    }
    
    return sample_node_sdf(p, node_idx);
}

// Calculate normal from SDF gradient
fn calc_normal(p: vec3f) -> vec3f {
    let eps = 0.005;
    let n = vec3f(
        sample_sdf(p + vec3f(eps, 0.0, 0.0)) - sample_sdf(p - vec3f(eps, 0.0, 0.0)),
        sample_sdf(p + vec3f(0.0, eps, 0.0)) - sample_sdf(p - vec3f(0.0, eps, 0.0)),
        sample_sdf(p + vec3f(0.0, 0.0, eps)) - sample_sdf(p - vec3f(0.0, 0.0, eps))
    );
    return normalize(n);
}

// Ray march through the octree volume
fn ray_march(ro: vec3f, rd: vec3f) -> vec4f {
    let MAX_STEPS = 200;
    let MAX_DIST = 10.0;
    let SURF_DIST = 0.0005;
    
    var t = 0.0;
    
    for (var i = 0; i < MAX_STEPS; i++) {
        let p = ro + rd * t;
        let d = sample_sdf(p);
        
        if (d < SURF_DIST) {
            // Hit surface - calculate shading
            let normal = calc_normal(p);
            
            // Light direction (rotating with time)
            let light_dir = normalize(vec3f(
                sin(uniforms.time * 0.5),
                0.8,
                cos(uniforms.time * 0.5)
            ));
            
            // Diffuse lighting
            let diff = max(dot(normal, light_dir), 0.0);
            
            // Ambient occlusion approximation
            let ao = 1.0 - f32(i) / f32(MAX_STEPS) * 0.5;
            
            // Fresnel-like rim lighting
            let view_dir = -rd;
            let rim = pow(1.0 - max(dot(normal, view_dir), 0.0), 3.0);
            
            // Color based on octree depth (for visualization)
            let node_idx = find_leaf_node(p);
            var depth_color = vec3f(0.2, 0.5, 0.9);
            if (node_idx >= 0) {
                let depth = f32(nodes[node_idx].depth) / f32(MAX_DEPTH);
                depth_color = mix(vec3f(0.2, 0.5, 0.9), vec3f(0.9, 0.3, 0.2), depth);
            }
            
            // Add normal-based color variation
            let base_color = depth_color + normal * 0.15;
            
            // Final color
            let color = base_color * (diff * 0.7 + 0.3) * ao + vec3f(0.8, 0.6, 0.4) * rim * 0.3;
            
            return vec4f(color, 1.0);
        }
        
        // Adaptive step size based on distance
        t += max(d * 0.9, 0.001);
        
        if (t > MAX_DIST) {
            break;
        }
    }
    
    // Background gradient
    let bg_top = vec3f(0.05, 0.05, 0.15);
    let bg_bottom = vec3f(0.02, 0.02, 0.05);
    return vec4f(mix(bg_bottom, bg_top, rd.y * 0.5 + 0.5), 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    // Normalized coordinates with aspect ratio correction
    let aspect = uniforms.resolution.x / uniforms.resolution.y;
    var uv = in.uv * 2.0 - 1.0;
    uv.x *= aspect;
    
    // Camera setup - orbiting around the scene
    let angle = uniforms.time * 0.3;
    let cam_dist = 1.1;
    let cam_pos = vec3f(
        sin(angle) * cam_dist,
        sin(uniforms.time * 0.2) * 0.3 + 0.3,
        cos(angle) * cam_dist
    );
    
    // Look at center
    let look_at = vec3f(0.0, 0.0, 0.0);
    let forward = normalize(look_at - cam_pos);
    let right = normalize(cross(vec3f(0.0, 1.0, 0.0), forward));
    let up = cross(forward, right);
    
    // Ray direction
    let fov = 1.5;
    let rd = normalize(forward * fov + right * uv.x + up * uv.y);
    
    // Ray march
    let color = ray_march(cam_pos, rd);
    
    // Gamma correction
    return vec4f(pow(color.rgb, vec3f(1.0 / 2.2)), color.a);
}
"""


def create_volume_atlas(nodes: list[OctreeNode]) -> tuple[np.ndarray, dict[int, int], int]:
    """
    Create a 3D texture atlas containing ALL node volumes (for LOD support).
    Returns the atlas data and a mapping from node index to volume index.
    """
    # Get all nodes with volume data
    volume_nodes = [n for n in nodes if n.volume_data is not None]
    num_volumes = len(volume_nodes)
    
    print(f"Creating atlas for {num_volumes} node volumes")
    
    # Arrange volumes in a 3D grid - calculate needed size
    volumes_per_side = max(8, int(np.ceil(num_volumes ** (1/3))))  # At least 8, or more if needed
    atlas_size = volumes_per_side * TEXTURE_SIZE
    print(f"  Atlas grid: {volumes_per_side}x{volumes_per_side}x{volumes_per_side} = {volumes_per_side**3} slots")
    
    atlas = np.zeros((atlas_size, atlas_size, atlas_size), dtype=np.float32)
    node_to_volume = {}
    
    for vol_idx, node in enumerate(volume_nodes):
        # Calculate position in atlas
        vol_x = vol_idx % volumes_per_side
        vol_y = (vol_idx // volumes_per_side) % volumes_per_side
        vol_z = vol_idx // (volumes_per_side * volumes_per_side)
        
        # Copy volume data to atlas
        x_start = vol_x * TEXTURE_SIZE
        y_start = vol_y * TEXTURE_SIZE
        z_start = vol_z * TEXTURE_SIZE
        
        atlas[
            x_start:x_start + TEXTURE_SIZE,
            y_start:y_start + TEXTURE_SIZE,
            z_start:z_start + TEXTURE_SIZE
        ] = node.volume_data
        
        node_to_volume[node.index] = vol_idx
    
    return atlas, node_to_volume, volumes_per_side


def create_node_buffer_data(nodes: list[OctreeNode], node_to_volume: dict[int, int]) -> np.ndarray:
    """
    Create GPU buffer data for octree nodes.
    Each node: center(vec3f), half_size(f32), children_start(i32), volume_index(i32), depth(u32), pad(u32)
    """
    # Reinterpret as proper types for GPU
    result = np.zeros(len(nodes) * 8, dtype=np.float32)
    for node in nodes:
        i = node.index
        base = i * 8
        result[base + 0] = node.center[0]
        result[base + 1] = node.center[1]
        result[base + 2] = node.center[2]
        result[base + 3] = node.half_size
        # Store integers properly
        int_view = result[base + 4:base + 8].view(np.int32)
        int_view[0] = node.children_start if not node.is_leaf else -1
        # ALL nodes now have volume data for LOD support
        int_view[1] = node_to_volume.get(node.index, -1)
        int_view[2] = node.depth
        int_view[3] = 0
    
    return result


def main():
    # Request adapter and device with float32-filterable feature
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync(
        required_features=["float32-filterable"],
    )
    
    # Create the render canvas
    canvas = RenderCanvas(
        title="SDF Ray Marching - Octree LOD",
        size=(800, 600),
        max_fps=60,
    )
    
    # Get the render context
    context = canvas.get_context("wgpu")
    render_texture_format = context.get_preferred_format(adapter)
    context.configure(device=device, format=render_texture_format)
    
    # Build octree
    print("Building octree...")
    nodes = build_octree(
        center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        half_size=1.0
    )
    print(f"Octree has {len(nodes)} nodes")
    print(f"  Leaf nodes: {sum(1 for n in nodes if n.is_leaf)}")
    print(f"  Max depth: {max(n.depth for n in nodes)}")
    
    # Create volume atlas
    print("Creating volume atlas...")
    atlas_data, node_to_volume, volumes_per_side = create_volume_atlas(nodes)
    print(f"Atlas size: {atlas_data.shape}")
    
    # Create 3D atlas texture
    atlas_size = atlas_data.shape[0]
    volume_texture = device.create_texture(
        size=(atlas_size, atlas_size, atlas_size),
        dimension="3d",
        format=wgpu.TextureFormat.r32float,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    
    # Upload atlas data (transpose for correct GPU layout: numpy [x,y,z] -> GPU [z,y,x] so x varies fastest)
    atlas_gpu = np.ascontiguousarray(np.transpose(atlas_data, (2, 1, 0)))
    device.queue.write_texture(
        {"texture": volume_texture},
        atlas_gpu.tobytes(),
        {"bytes_per_row": atlas_size * 4, "rows_per_image": atlas_size},
        (atlas_size, atlas_size, atlas_size),
    )
    
    # Create sampler with trilinear filtering
    volume_sampler = device.create_sampler(
        mag_filter="linear",
        min_filter="linear",
        mipmap_filter="linear",
        address_mode_u="clamp-to-edge",
        address_mode_v="clamp-to-edge",
        address_mode_w="clamp-to-edge",
    )
    
    # Create node buffer
    node_buffer_data = create_node_buffer_data(nodes, node_to_volume)
    node_buffer = device.create_buffer(
        size=node_buffer_data.nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    device.queue.write_buffer(node_buffer, 0, node_buffer_data.tobytes())
    
    # Create uniform buffer (16 floats = 64 bytes for alignment)
    # resolution(2) + time(1) + num_nodes(1) + max_lod_depth(1) + volumes_per_side(1) + 
    # lod_distance_scale(1) + use_distance_lod(1) + camera_pos(3) + pad(1) = 12, padded to 16
    uniform_data = np.zeros(16, dtype=np.float32)
    uniform_buffer = device.create_buffer(
        size=uniform_data.nbytes,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )
    
    # LOD control state
    current_lod = [MAX_DEPTH]  # Use list for closure mutation
    use_distance_lod = [1]  # 1 = enabled, 0 = fixed LOD
    lod_distance_scale = [3.0]  # How aggressively LOD drops with distance
    
    # Create shader module
    shader_module = device.create_shader_module(code=SHADER_CODE)
    
    # Create bind group layout
    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": "uniform"},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": "read-only-storage"},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": "float",
                    "view_dimension": "3d",
                },
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": "filtering"},
            },
        ]
    )
    
    # Create pipeline layout
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    
    # Create render pipeline
    render_pipeline = device.create_render_pipeline(
        layout=pipeline_layout,
        vertex={
            "module": shader_module,
            "entry_point": "vs_main",
        },
        fragment={
            "module": shader_module,
            "entry_point": "fs_main",
            "targets": [{"format": render_texture_format}],
        },
        primitive={
            "topology": "triangle-list",
        },
    )
    
    # Create texture view
    volume_texture_view = volume_texture.create_view()
    
    # Create bind group
    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {"binding": 0, "resource": {"buffer": uniform_buffer}},
            {"binding": 1, "resource": {"buffer": node_buffer}},
            {"binding": 2, "resource": volume_texture_view},
            {"binding": 3, "resource": volume_sampler},
        ],
    )
    
    # Time tracking
    import time
    start_time = time.perf_counter()
    
    # Get the GLFW window handle for keyboard input
    glfw_window = canvas._window
    
    def key_callback(window, key, scancode, action, mods):
        if action == glfw.PRESS or action == glfw.REPEAT:
            # Number keys 1-6 set max LOD level (0-5 depth limit)
            if key == glfw.KEY_1:
                current_lod[0] = 0
                print(f"Max LOD: 0 (root only)")
            elif key == glfw.KEY_2:
                current_lod[0] = 1
                print(f"Max LOD: 1")
            elif key == glfw.KEY_3:
                current_lod[0] = 2
                print(f"Max LOD: 2")
            elif key == glfw.KEY_4:
                current_lod[0] = 3
                print(f"Max LOD: 3")
            elif key == glfw.KEY_5:
                current_lod[0] = 4
                print(f"Max LOD: 4")
            elif key == glfw.KEY_6:
                current_lod[0] = 5
                print(f"Max LOD: 5 (max detail)")
            elif key == glfw.KEY_0:
                current_lod[0] = MAX_DEPTH
                print(f"Max LOD: {MAX_DEPTH} (full detail)")
            # D key toggles distance-based LOD
            elif key == glfw.KEY_D:
                use_distance_lod[0] = 1 - use_distance_lod[0]
                mode = "Distance-based" if use_distance_lod[0] else "Fixed"
                print(f"LOD Mode: {mode}")
            # +/- keys adjust distance scale
            elif key == glfw.KEY_EQUAL or key == glfw.KEY_KP_ADD:
                lod_distance_scale[0] = min(10.0, lod_distance_scale[0] + 0.5)
                print(f"LOD Distance Scale: {lod_distance_scale[0]:.1f}")
            elif key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT:
                lod_distance_scale[0] = max(0.5, lod_distance_scale[0] - 0.5)
                print(f"LOD Distance Scale: {lod_distance_scale[0]:.1f}")
    
    glfw.set_key_callback(glfw_window, key_callback)
    
    def draw_frame():
        # Update uniforms
        current_time = time.perf_counter() - start_time
        width, height = canvas.get_physical_size()
        
        # Safety: ensure we have valid dimensions
        if width == 0 or height == 0:
            canvas.request_draw(draw_frame)
            return
        
        # Calculate camera position (same as shader)
        angle = current_time * 0.3
        cam_dist = 1.1
        cam_x = np.sin(angle) * cam_dist
        cam_y = np.sin(current_time * 0.2) * 0.3 + 0.3
        cam_z = np.cos(angle) * cam_dist
        
        uniform_data[0] = float(width)
        uniform_data[1] = float(height)
        uniform_data[2] = current_time
        # Store integers as uint32 for slots 3-7
        int_view = uniform_data[3:8].view(np.uint32)
        int_view[0] = len(nodes)          # num_nodes
        int_view[1] = current_lod[0]      # max_lod_depth
        int_view[2] = volumes_per_side    # volumes_per_side
        # Slot 6-7: lod_distance_scale (float) and use_distance_lod (uint)
        uniform_data[6] = lod_distance_scale[0]
        int_view2 = uniform_data[7:8].view(np.uint32)
        int_view2[0] = use_distance_lod[0]
        # Camera position
        uniform_data[8] = cam_x
        uniform_data[9] = cam_y
        uniform_data[10] = cam_z
        uniform_data[11] = 0.0  # padding
        
        device.queue.write_buffer(uniform_buffer, 0, uniform_data.tobytes())
        
        # Get current texture to render to
        current_texture = context.get_current_texture()
        
        # Create command encoder
        command_encoder = device.create_command_encoder()
        
        # Begin render pass
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": current_texture.create_view(),
                    "load_op": "clear",
                    "store_op": "store",
                    "clear_value": (0.0, 0.0, 0.0, 1.0),
                }
            ]
        )
        
        # Draw
        render_pass.set_pipeline(render_pipeline)
        render_pass.set_bind_group(0, bind_group)
        render_pass.draw(3)  # Full-screen triangle
        render_pass.end()
        
        # Submit
        device.queue.submit([command_encoder.finish()])
        
        # Request next frame for animation
        canvas.request_draw(draw_frame)
    
    # Register draw callback
    canvas.request_draw(draw_frame)
    
    print("Starting render loop...")
    print("Controls:")
    print(f"  1-6: Set max LOD level (1=coarse, 6=finest, max={MAX_DEPTH})")
    print("  0: Full detail")
    print("  D: Toggle distance-based LOD")
    print("  +/-: Adjust distance LOD scale")
    print("  Close window to exit")
    print("Color indicates octree depth (blue=shallow, red=deep)")
    print(f"Distance LOD: {'ON' if use_distance_lod[0] else 'OFF'}, Scale: {lod_distance_scale[0]:.1f}")
    
    # Run the event loop
    loop.run()


if __name__ == "__main__":
    main()
