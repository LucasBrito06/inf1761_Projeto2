// =============================================================================
// sun.wgsl - o Sol (Projeto 2)
// O Sol e a fonte de luz: nao recebe iluminacao, so mostra a propria textura
// (emissivo), mas tambem respeita o plano de corte quando ativo.
// =============================================================================
struct Matrix {
  vertex: mat4x4<f32>,
}
@group(1) @binding(0) var<storage, read> matrix: array<Matrix>;

struct Global {
  projection: mat4x4<f32>,
  clip_plane: vec4<f32>,
  clip_plane_color: vec4<f32>,
}
@group(0) @binding(0) var<uniform> global: Global;

@group(2) @binding(0) var decal_texture: texture_2d<f32>;
@group(2) @binding(1) var decal_sampler: sampler;

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) texcoord: vec2<f32>,
  @location(1) vpos: vec3<f32>,
}

@vertex
fn vs_main (@builtin(instance_index) instance_index: u32,
            @location(0) coord: vec3<f32>,
            @location(1) normal: vec3<f32>,
            @location(2) tangent: vec3<f32>,
            @location(3) texcoord: vec2<f32>) -> VertexOutput {
  var out: VertexOutput;
  let vpos = (matrix[instance_index].vertex * vec4<f32>(coord, 1.0)).xyz;
  out.texcoord = texcoord;
  out.vpos = vpos;
  out.clip_position = global.projection * vec4<f32>(vpos, 1.0);
  return out;
}

@fragment
fn fs_main (in: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
  let color = textureSample(decal_texture, decal_sampler, in.texcoord).rgb * 1.25;
  if (dot(global.clip_plane.xyz, in.vpos) + global.clip_plane.w < 0.0) {
    discard;
  }
  if (!front_facing) {
    return vec4<f32>(global.clip_plane_color.rgb, 1.0);
  }
  return vec4<f32>(color, 1.0);
}
