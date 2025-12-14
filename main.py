"""
3D Volume Texture SDF Ray Marching with wgpu-py
Renders a sphere signed distance field using ray marching through a 3D texture.
"""

import numpy as np
import wgpu
from rendercanvas.auto import RenderCanvas, loop

# Volume texture resolution
VOLUME_SIZE = 64

# WGSL Shader for ray marching through 3D SDF volume
SHADER_CODE = """
struct Uniforms {
    resolution: vec2f,
    time: f32,
    _pad: f32,
    camera_pos: vec3f,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var volume_texture: texture_3d<f32>;
@group(0) @binding(2) var volume_sampler: sampler;

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

// Sample the SDF from 3D texture
fn sample_sdf(p: vec3f) -> f32 {
    // Transform world position to texture coordinates [0, 1]
    let tex_coord = p * 0.5 + 0.5;
    
    // Check if we're inside the volume
    if (any(tex_coord < vec3f(0.0)) || any(tex_coord > vec3f(1.0))) {
        return 1.0; // Outside volume, return large distance
    }
    
    return textureSampleLevel(volume_texture, volume_sampler, tex_coord, 0.0).r;
}

// Calculate normal from SDF gradient
fn calc_normal(p: vec3f) -> vec3f {
    let eps = 0.01;
    let n = vec3f(
        sample_sdf(p + vec3f(eps, 0.0, 0.0)) - sample_sdf(p - vec3f(eps, 0.0, 0.0)),
        sample_sdf(p + vec3f(0.0, eps, 0.0)) - sample_sdf(p - vec3f(0.0, eps, 0.0)),
        sample_sdf(p + vec3f(0.0, 0.0, eps)) - sample_sdf(p - vec3f(0.0, 0.0, eps))
    );
    return normalize(n);
}

// Ray march through the volume
fn ray_march(ro: vec3f, rd: vec3f) -> vec4f {
    let MAX_STEPS = 128;
    let MAX_DIST = 10.0;
    let SURF_DIST = 0.001;
    
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
            
            // Color gradient based on normal
            let base_color = vec3f(0.2, 0.5, 0.9) + normal * 0.3;
            
            // Final color
            let color = base_color * (diff * 0.7 + 0.3) * ao + vec3f(0.8, 0.6, 0.4) * rim * 0.3;
            
            return vec4f(color, 1.0);
        }
        
        t += d;
        
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
    let cam_dist = 2.5;
    let cam_pos = vec3f(
        sin(angle) * cam_dist,
        sin(uniforms.time * 0.2) * 0.5 + 0.5,
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


def create_sphere_sdf_volume(size: int) -> np.ndarray:
    """
    Create a 3D numpy array containing a sphere SDF.
    The volume is centered with coordinates ranging from -1 to 1.
    """
    # Create coordinate grids
    coords = np.linspace(-1, 1, size, dtype=np.float32)
    x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
    
    # Sphere SDF: distance to sphere surface at origin with radius 0.5
    sphere_radius = 0.5
    sdf = np.sqrt(x**2 + y**2 + z**2) - sphere_radius
    
    # Add some interesting perturbation for visual interest
    # Sine wave displacement on the surface
    freq = 8.0
    amplitude = 0.05
    displacement = amplitude * np.sin(x * freq) * np.sin(y * freq) * np.sin(z * freq)
    sdf = sdf + displacement
    
    # Normalize to reasonable range for texture storage
    # SDF values typically range from negative (inside) to positive (outside)
    return sdf.astype(np.float32)


def main():
    # Request adapter and device
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    
    # Create the render canvas
    canvas = RenderCanvas(
        title="SDF Ray Marching - Sphere Volume",
        size=(800, 600),
        max_fps=60,
    )
    
    # Get the render context
    context = canvas.get_context("wgpu")
    render_texture_format = context.get_preferred_format(adapter)
    context.configure(device=device, format=render_texture_format)
    
    # Create 3D SDF volume texture
    print("Generating sphere SDF volume...")
    sdf_data = create_sphere_sdf_volume(VOLUME_SIZE)
    print(f"SDF range: [{sdf_data.min():.3f}, {sdf_data.max():.3f}]")
    
    # Create 3D texture
    volume_texture = device.create_texture(
        size=(VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE),
        dimension="3d",
        format=wgpu.TextureFormat.r32float,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    
    # Upload SDF data to texture
    device.queue.write_texture(
        {"texture": volume_texture},
        sdf_data.tobytes(),
        {"bytes_per_row": VOLUME_SIZE * 4, "rows_per_image": VOLUME_SIZE},
        (VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE),
    )
    
    # Create sampler for 3D texture (nearest filtering for unfilterable-float textures)
    volume_sampler = device.create_sampler(
        mag_filter="nearest",
        min_filter="nearest",
        mipmap_filter="nearest",
        address_mode_u="clamp-to-edge",
        address_mode_v="clamp-to-edge",
        address_mode_w="clamp-to-edge",
    )
    
    # Create uniform buffer
    uniform_data = np.zeros(8, dtype=np.float32)  # resolution(2) + time(1) + pad(1) + camera_pos(3) + pad(1)
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
                "texture": {
                    "sample_type": "unfilterable-float",
                    "view_dimension": "3d",
                },
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": "non-filtering"},
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
            {"binding": 1, "resource": volume_texture_view},
            {"binding": 2, "resource": volume_sampler},
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
        uniform_data[3] = 0.0  # padding
        uniform_data[4] = 0.0  # camera_pos.x (calculated in shader)
        uniform_data[5] = 0.0  # camera_pos.y
        uniform_data[6] = 0.0  # camera_pos.z
        uniform_data[7] = 0.0  # padding
        
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
    
    # Run the event loop
    loop.run()


if __name__ == "__main__":
    main()
