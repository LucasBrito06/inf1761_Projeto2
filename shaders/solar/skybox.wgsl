// =============================================================================
// skybox.wgsl - fundo de espaco por cubemap (Projeto 2)
// O SkyBoxTransform recentra o cubo no olho da camera. A posicao e forcada
// para o plano distante (z = w), e a pipeline desliga a escrita de
// profundidade: o fundo fica sempre atras de tudo.
// =============================================================================
struct Matrix {
  vertex: mat4x4<f32>,
}
@group(1) @binding(0) var<storage, read> matrix: array<Matrix>;

struct Global {
  projection: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> global: Global;

@group(2) @binding(0) var sky_texture: texture_cube<f32>;
@group(2) @binding(1) var sky_sampler: sampler;

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) direction: vec3<f32>,
}

@vertex
fn vs_main (@builtin(instance_index) instance_index: u32,
            @location(0) coord: vec3<f32>) -> VertexOutput {
  var out: VertexOutput;
  let p = global.projection * (matrix[instance_index].vertex * vec4<f32>(coord, 1.0));
  out.clip_position = vec4<f32>(p.x, p.y, p.w, p.w);
  out.direction = coord;
  return out;
}

@fragment
fn fs_main (in: VertexOutput) -> @location(0) vec4<f32> {
  return textureSample(sky_texture, sky_sampler, in.direction);
}
