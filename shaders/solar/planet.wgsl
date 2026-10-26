// =============================================================================
// planet.wgsl - corpo celeste iluminado pelo Sol (Projeto 2)
//
// Iluminacao por fragmento (Blinn-Phong) com mapeamento de rugosidade:
//  - cor vem da textura "decal" (sRGB), multiplicada pela cor do material;
//  - a normal vem do mapa de normais (rgba8unorm, NAO sRGB), decodificada
//    por n = 2c - 1, e vive no ESPACO TANGENTE;
//  - por isso o vertex shader monta a base (t, b, n) de cada vertice e leva
//    luz e visao para o espaco tangente (leva-se luz e visao para la, e nao
//    a normal para ca - sao dois vetores por vertice, e nao um por pixel);
//  - teste da normal macro: um ponto com a micro-normal voltada para a luz
//    mas a macro de costas (light.z <= 0) nao recebe luz.
// Todos os astros usam este mesmo shader: o astro sem relevo recebe o mapa
// neutro (0.5, 0.5, 1.0), que decodifica para a normal em repouso (0,0,1).
// O espaco de iluminacao e o da camera (a origem e o observador).
// =============================================================================
struct Matrix {
  vertex: mat4x4<f32>,
  normal: mat4x4<f32>,
}
@group(1) @binding(0) var<storage, read> matrix: array<Matrix>;

struct Global {
  projection: mat4x4<f32>,
  light_ambient: vec3<f32>,
  light_diffuse: vec3<f32>,
  light_specular: vec3<f32>,
  light_position: vec4<f32>,
  clip_plane: vec4<f32>,        // plano de corte no espaco da camera (ax+by+cz+d)
  clip_plane_color: vec4<f32>,  // cor da secao do corte (face de dentro)
  bump_amount: f32,             // 0 = relevo desligado, 1 = relevo total
}
@group(0) @binding(0) var<uniform> global: Global;

struct Material {
  base_color: vec3<f32>,
  opacity: f32,

  specular_color: vec3<f32>,
  shininess: f32,
}
@group(2) @binding(0) var<uniform> material: Material;

@group(3) @binding(0) var decal_texture: texture_2d<f32>;
@group(3) @binding(1) var decal_sampler: sampler;
@group(3) @binding(2) var normal_texture: texture_2d<f32>;
@group(3) @binding(3) var normal_sampler: sampler;

struct VertexOutput {
  @builtin(position) clip_position: vec4<f32>,
  @location(0) light: vec3<f32>,     // espaco tangente
  @location(1) view: vec3<f32>,      // espaco tangente
  @location(2) texcoord: vec2<f32>,
  @location(3) vpos: vec3<f32>,      // posicao no espaco da camera (para o corte)
}

@vertex
fn vs_main (@builtin(instance_index) instance_index: u32,
            @location(0) coord: vec3<f32>,
            @location(1) normal: vec3<f32>,
            @location(2) tangent: vec3<f32>,
            @location(3) texcoord: vec2<f32>) -> VertexOutput {
  let m = matrix[instance_index];
  var out: VertexOutput;
  let vpos = (m.vertex * vec4<f32>(coord, 1.0)).xyz;

  // base do espaco tangente: t (tangente), b = n x t, n (normal)
  let n = normalize((m.normal * vec4<f32>(normal, 0.0)).xyz);
  let t = normalize((m.normal * vec4<f32>(tangent, 0.0)).xyz);
  let b = cross(n, t);
  // o construtor de mat3x3 recebe colunas: as linhas ficam t, b, n
  let to_tangent_space = mat3x3<f32>(
    vec3<f32>(t.x, b.x, n.x),
    vec3<f32>(t.y, b.y, n.y),
    vec3<f32>(t.z, b.z, n.z),
  );

  // no espaco da camera a origem e o observador
  out.light = to_tangent_space * (global.light_position.xyz - vpos);
  out.view = to_tangent_space * (-vpos);
  out.texcoord = texcoord;
  out.vpos = vpos;
  out.clip_position = global.projection * vec4<f32>(vpos, 1.0);
  return out;
}

@fragment
fn fs_main (in: VertexOutput, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
  // amostragens antes de qualquer descarte (fluxo uniforme)
  let decal = textureSample(decal_texture, decal_sampler, in.texcoord).rgb;
  let texel = textureSample(normal_texture, normal_sampler, in.texcoord).rgb;

  // plano de corte: descarta o lado negativo
  if (dot(global.clip_plane.xyz, in.vpos) + global.clip_plane.w < 0.0) {
    discard;
  }
  // face de dentro da esfera cortada: pinta a secao com cor lisa
  if (!front_facing) {
    return vec4<f32>(global.clip_plane_color.rgb, 1.0);
  }

  // decodifica o mapa de normais; bump_amount escala a inclinacao (x, y) da
  // micro-normal, permitindo desligar (0) ou aprofundar (> 1) os relevos
  let raw = 2.0 * texel - vec3<f32>(1.0);
  let normal = normalize(vec3<f32>(raw.xy * global.bump_amount, max(raw.z, 0.05)));
  let view = normalize(in.view);
  let light = normalize(in.light);
  let albedo = decal * material.base_color;

  // teste da normal macro: no espaco tangente ela e sempre (0, 0, 1)
  let ndotl = select(0.0, dot(normal, light), light.z > 0.0);

  var color = albedo * global.light_ambient.rgb;
  if (ndotl > 0.0) {
    color += albedo * global.light_diffuse.rgb * ndotl;
    let refl = normalize(reflect(-light, normal));
    color += material.specular_color * global.light_specular.rgb
           * pow(max(0.0, dot(refl, view)), material.shininess);
  }
  return vec4<f32>(color, 1.0);
}
