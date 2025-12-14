"""
3D Volume Texture SDF Ray Marching with Octree LOD
Renders a sphere signed distance field using ray marching through octree-organized 3D textures.
Each octree node has border cells for seamless neighbor sampling.
"""

import numpy as np
import wgpu
from rendercanvas.auto import RenderCanvas, loop
from dataclasses import dataclass
from typing import Optional

# Volume texture resolution per node (excluding borders)
VOLUME_SIZE = 16
# Border size for neighbor sampling (1 cell on each side)
BORDER_SIZE = 1
# Total texture size including borders
TEXTURE_SIZE = VOLUME_SIZE + 2 * BORDER_SIZE
# Maximum octree depth (3 gives up to ~300 leaf nodes near sphere surface)
MAX_DEPTH = 3
# Maximum number of octree nodes (for GPU buffer allocation)
MAX_NODES = 512


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
    sphere_radius = 0.5
    dist = np.sqrt(x**2 + y**2 + z**2) - sphere_radius
    
    # Add sine displacement for visual interest
    freq = 8.0
    amplitude = 0.05
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
    Returns a flat list of all nodes.
    """
    nodes = []
    leaf_count = [0]  # Use list for closure mutation
    
    def subdivide(node: OctreeNode) -> None:
        nodes.append(node)
        
        if not should_subdivide(node):
            # Leaf node - generate volume data
            node.is_leaf = True
            node.volume_data = generate_node_volume(node)
            leaf_count[0] += 1
            if leaf_count[0] % 10 == 0:
                print(f"  Generated {leaf_count[0]} leaf volumes...")
            return
        
        # Subdivide into 8 children
        node.is_leaf = False
        node.children_start = len(nodes)
        child_half_size = node.half_size / 2
        
        child_idx = 0
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                for dz in [-1, 1]:
                    child_center = node.center + np.array([dx, dy, dz]) * child_half_size
                    child = OctreeNode(
                        center=child_center,
                        half_size=child_half_size,
                        depth=node.depth + 1,
                        index=len(nodes) + child_idx,
                        parent_index=node.index,
                        children_start=-1,
                    )
                    child_idx += 1
        
        # Process children (they get added to nodes list)
        children_to_process = []
        for i in range(8):
            dx = (i // 4) * 2 - 1
            dy = ((i // 2) % 2) * 2 - 1
            dz = (i % 2) * 2 - 1
            child_center = node.center + np.array([dx, dy, dz]) * child_half_size
            child = OctreeNode(
                center=child_center,
                half_size=child_half_size,
                depth=node.depth + 1,
                index=node.children_start + i,
                parent_index=node.index,
                children_start=-1,
            )
            children_to_process.append(child)
        
        # Update children indices after we know how many there are
        node.children_start = len(nodes)
        for child in children_to_process:
            child.index = len(nodes)
            subdivide(child)
    
    # Create root node
    root = OctreeNode(
        center=center,
        half_size=half_size,
        depth=0,
        index=0,
        parent_index=-1,
        children_start=-1,
    )
    
    subdivide(root)
    
    return nodes


# WGSL Shader for octree-based ray marching through 3D SDF volumes
SHADER_CODE = """
struct Uniforms {
    resolution: vec2f,
    time: f32,
    num_nodes: u32,
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

// Find the leaf node containing point p
fn find_leaf_node(p: vec3f) -> i32 {
    var current: i32 = 0;
    
    for (var depth = 0u; depth < MAX_DEPTH + 1u; depth++) {
        let node = nodes[current];
        
        // Check if this is a leaf
        if (node.children_start < 0) {
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
    
    // Sample from the combined 3D texture (volumes stacked in Z)
    let volumes_per_side = 8u;  // 8x8 grid of volumes in XY, stacked in Z
    let vol_x = u32(volume_idx) % volumes_per_side;
    let vol_y = (u32(volume_idx) / volumes_per_side) % volumes_per_side;
    let vol_z = u32(volume_idx) / (volumes_per_side * volumes_per_side);
    
    let atlas_coord = vec3f(
        (f32(vol_x) + tex_coord.x) / f32(volumes_per_side),
        (f32(vol_y) + tex_coord.y) / f32(volumes_per_side),
        (f32(vol_z) + tex_coord.z) / f32(volumes_per_side)
    );
    
    return textureSampleLevel(volume_textures, volume_sampler, atlas_coord, 0.0).r;
}

// Sample the SDF with octree traversal
fn sample_sdf(p: vec3f) -> f32 {
    let node_idx = find_leaf_node(p);
    if (node_idx < 0) {
        return 1.0;  // Outside octree
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
    let cam_dist = 2.0;
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


def create_volume_atlas(nodes: list[OctreeNode]) -> tuple[np.ndarray, dict[int, int]]:
    """
    Create a 3D texture atlas containing all leaf node volumes.
    Returns the atlas data and a mapping from node index to volume index.
    """
    # Get all leaf nodes
    leaf_nodes = [n for n in nodes if n.is_leaf and n.volume_data is not None]
    num_leaves = len(leaf_nodes)
    
    print(f"Creating atlas for {num_leaves} leaf nodes")
    
    # Arrange volumes in a 3D grid
    volumes_per_side = 8  # 8x8x8 = 512 max volumes
    atlas_size = volumes_per_side * TEXTURE_SIZE
    
    atlas = np.zeros((atlas_size, atlas_size, atlas_size), dtype=np.float32)
    node_to_volume = {}
    
    for vol_idx, node in enumerate(leaf_nodes):
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
    
    return atlas, node_to_volume


def create_node_buffer_data(nodes: list[OctreeNode], node_to_volume: dict[int, int]) -> np.ndarray:
    """
    Create GPU buffer data for octree nodes.
    Each node: center(vec3f), half_size(f32), children_start(i32), volume_index(i32), depth(u32), pad(u32)
    """
    # 8 floats per node (32 bytes, properly aligned)
    data = np.zeros((len(nodes), 8), dtype=np.float32)
    
    for node in nodes:
        i = node.index
        data[i, 0] = node.center[0]  # center.x
        data[i, 1] = node.center[1]  # center.y
        data[i, 2] = node.center[2]  # center.z
        data[i, 3] = node.half_size   # half_size
        # Pack integers as float bits
        data[i, 4] = np.float32(node.children_start if not node.is_leaf else -1).view(np.float32)
        volume_idx = node_to_volume.get(node.index, -1) if node.is_leaf else -1
        data[i, 5] = np.float32(volume_idx).view(np.float32)
        data[i, 6] = np.float32(node.depth).view(np.float32)
        data[i, 7] = 0.0  # padding
    
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
        int_view[1] = node_to_volume.get(node.index, -1) if node.is_leaf else -1
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
    atlas_data, node_to_volume = create_volume_atlas(nodes)
    print(f"Atlas size: {atlas_data.shape}")
    
    # Create 3D atlas texture
    atlas_size = atlas_data.shape[0]
    volume_texture = device.create_texture(
        size=(atlas_size, atlas_size, atlas_size),
        dimension="3d",
        format=wgpu.TextureFormat.r32float,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    
    # Upload atlas data
    device.queue.write_texture(
        {"texture": volume_texture},
        atlas_data.tobytes(),
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
    
    # Create uniform buffer
    uniform_data = np.zeros(4, dtype=np.float32)  # resolution(2) + time(1) + num_nodes(1)
    uniform_buffer = device.create_buffer(
        size=uniform_data.nbytes,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )
    
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
    
    def draw_frame():
        # Update uniforms
        current_time = time.perf_counter() - start_time
        width, height = canvas.get_physical_size()
        
        uniform_data[0] = float(width)
        uniform_data[1] = float(height)
        uniform_data[2] = current_time
        # Store num_nodes as uint32
        uniform_data[3] = np.float32(len(nodes)).view(np.float32)
        int_view = uniform_data[3:4].view(np.uint32)
        int_view[0] = len(nodes)
        
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
    
    # Register draw callback
    canvas.request_draw(draw_frame)
    
    print("Starting render loop...")
    print("Controls: Close window to exit")
    print("Color indicates octree depth (blue=shallow, red=deep)")
    
    # Run the event loop
    loop.run()


if __name__ == "__main__":
    main()
