# PyMarch 🪨

A real-time **SDF ray marcher** rendering a procedural asteroid with **octree-based LOD** (Level of Detail), built entirely in Python using WebGPU.

![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)
![WebGPU](https://img.shields.io/badge/graphics-WebGPU-orange.svg)

## What is this?

This is a **highly unoptimized but educational** demonstration of several advanced real-time graphics techniques:

- **Signed Distance Field (SDF) Ray Marching** — Instead of traditional polygon meshes, the asteroid is defined mathematically as a distance field and rendered by marching rays through it
- **Octree Spatial Subdivision** — The SDF is precomputed into a hierarchical octree of 3D volume textures, enabling variable detail across space
- **Distance-Based LOD** — Regions closer to the camera render at higher detail; distant areas use coarser octree nodes
- **Procedural Generation** — The asteroid surface uses layered FBM (Fractal Brownian Motion) noise for organic, rocky detail

## Screenshots + Video

[![](https://img.shields.io/badge/▶️%20Video-Demo-informational)](media/recording.mp4)
![Screenshot](media/Screenshot%202025-12-23%20203034.png)

> **Watch a demo**: [media/recording.mp4](media/recording.mp4)


The demo renders a rocky asteroid floating in a procedural space environment with:
- Soft shadows with penumbra
- Ambient occlusion
- Subsurface scattering approximation
- Procedural starfield and nebulae
- Fresnel rim lighting

## Installation

Requires **Python 3.13+** and a GPU with Vulkan, Metal, or DX12 support.

```bash
# Clone the repository
git clone <your-repo-url>
cd pymarch

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Usage

```bash
# Run the demo
uv run python main.py

# Start in fullscreen
uv run python main.py --fullscreen

# Clear cached octree data (forces regeneration)
uv run python main.py --clear-cache
```

## Controls

| Key | Action |
|-----|--------|
| `W` `A` `S` `D` | Move camera |
| `Space` / `Shift` | Move up / down |
| `Mouse` | Look around (click to capture) |
| `ESC` | Release mouse |
| `F11` | Toggle fullscreen |
| `1` - `6` | Set max LOD level (1=coarse, 6=finest) |
| `0` | Full detail mode |
| `L` | Toggle distance-based LOD |
| `+` / `-` | Adjust LOD distance scale |

## How It Works

### 1. Octree Construction

The SDF is defined analytically as a noisy sphere:

```python
def sample_sdf_world(p):
    sphere_radius = 0.4
    dist = length(p) - sphere_radius
    displacement = amplitude * fbm(p * noise_scale)
    return dist + displacement
```

An octree is built by subdividing space wherever the surface passes through. Each node stores a 32³ (+ 2-cell border) 3D volume texture containing sampled SDF values.

### 2. Volume Atlas

All node volumes are packed into a single 3D texture atlas for efficient GPU access. The atlas uses a 3D grid layout to maximize texture utilization.

### 3. Ray Marching

The WGSL fragment shader:
1. Casts a ray from the camera through each pixel
2. Traverses the octree to find the appropriate LOD node
3. Samples the SDF from that node's volume texture
4. Steps along the ray until hitting the surface
5. Computes lighting with shadows, AO, and environment reflections

### 4. Distance-Based LOD

As points get farther from the camera, the shader stops at shallower octree depths:

```wgsl
fn get_distance_lod(p: vec3f) -> u32 {
    let dist = length(p - camera_pos);
    let lod_reduction = floor(dist * lod_distance_scale);
    return max_lod_depth - lod_reduction;
}
```

## Technical Details

| Parameter | Value |
|-----------|-------|
| Volume resolution | 32³ per node (+ 2-cell border) |
| Max octree depth | 5 levels |
| Max nodes | 2048 |
| Texture format | R32Float |
| Ray march steps | 400 max |
| Shadow steps | 64 max |

## Dependencies

- **[wgpu-py](https://github.com/pygfx/wgpu-py)** — WebGPU bindings for Python
- **[rendercanvas](https://github.com/pygfx/rendercanvas)** — Cross-platform canvas for wgpu
- **[glfw](https://github.com/FlorianRhworern/pyGLFW)** — Window and input handling
- **[NumPy](https://numpy.org/)** — Array operations for volume generation

## Performance Notes

> ⚠️ **This is intentionally unoptimized** for educational clarity.

Potential optimizations not implemented:
- Sphere tracing with proper step size adaptation
- Temporal reprojection / denoising  
- Compute shader for octree traversal
- Hierarchical empty space skipping
- GPU-side octree construction
- Mipmap chains for volume textures
- Sparse volume representation

## Caching

Octree generation is slow (lots of FBM noise evaluation), so results are cached to disk in the `cache/` directory. The cache key is based on all SDF parameters — if you modify the noise or sphere settings, regeneration will happen automatically.

## License

MIT

---

*Built with curiosity and too many ray march iterations* ✨

