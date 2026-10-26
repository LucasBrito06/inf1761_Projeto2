// =============================================================================
// main_solar3d.cpp  -  Projeto 2: Mini-sistema solar 3D
// INF1761 Computacao Grafica | PUC-Rio | Prof. Waldemar Celes
// Aluno: Lucas Brito - 2411907
// =============================================================================
// Grafo de cena (todas as esferas instanciam a mesma Sphere do framework):
//
//   root
//   |- sky_node        [SkyBoxTransform]      SkyBox + cubemap do espaco
//   |- sun_pivot       [spin do Sol]          esfera emissiva (pipeline do Sol)
//   `- planets (pipeline dos planetas, luz pontual na posicao do Sol)
//       |- mercury_orbit -> mercury_pos -> mercury_spin -> mercury_body
//       `- earth_orbit -> earth_pos
//           |- earth_tilt -> earth_spin -> earth_body
//           `- moon_orbit -> moon_pos -> moon_spin -> moon_body
//
// Os nos de "orbit" e "spin" sao pivos: a SolarEngine acumula uma rotacao
// neles a cada quadro. A Lua herda o movimento da Terra pela pilha de
// matrizes, mas e irma do no de rotacao propria da Terra para nao girar junto
// com os continentes.
//
// Cameras (tecla C cicla entre elas, ou teclas 1, 2, 3, 4):
//   1. Global (visao geral do sistema, com arcball + zoom no scroll)
//   2. Terra -> Lua (na superficie da Terra olhando para a Lua no centro da
//      tela, atualizada a cada quadro pela EarthCameraEngine para ver as fases)
//   3. Close-up na Terra (acompanha a Terra com arcball proprio + zoom)
//   4. Close-up em Mercurio (acompanha Mercurio com arcball proprio + zoom)
//
// Controles:
//   C ou 1..4 : troca de camera       | Scroll do mouse : zoom in / zoom out
//   B         : relevo liga/desliga   | [ / ]           : intensidade do relevo
//   X         : plano de corte        | P               : pausa
//   + / -     : velocidade da animacao| Q / Esc         : sai
// =============================================================================

#include <GLFW/glfw3.h>

#include "wgpucontext.h"
#include "arcball.h"
#include "scene.h"
#include "state.h"
#include "camera3d.h"
#include "phongmaterial.h"
#include "pointlight.h"
#include "transform.h"
#include "node.h"
#include "sphere.h"
#include "skybox.h"
#include "texture.h"
#include "texcube.h"
#include "sampler.h"
#include "textureset.h"
#include "shader.h"
#include "pipeline.h"
#include "renderer.h"
#include "clipplane.h"
#include "engine.h"
#include "error.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

// ---------------------------------------------------------------------------
// Constantes do sistema (unidades arbitrarias; angulos em graus, tempo em s)
// ---------------------------------------------------------------------------
static constexpr float R_SUN      = 1.20f;
static constexpr float R_MERCURY  = 0.14f;
static constexpr float R_EARTH    = 0.38f;
static constexpr float R_MOON     = 0.10f;

static constexpr float D_MERCURY  = 2.40f;   // Sol - Mercurio
static constexpr float D_EARTH    = 4.20f;   // Sol - Terra
static constexpr float D_MOON     = 0.95f;   // Terra - Lua

static constexpr float W_MERCURY_ORBIT = 80.0f;   // translacao de Mercurio
static constexpr float W_MERCURY_SPIN  = 25.0f;
static constexpr float W_EARTH_ORBIT   = 20.0f;   // translacao da Terra
static constexpr float W_EARTH_SPIN    = 120.0f;  // rotacao da Terra
static constexpr float W_MOON_ORBIT    = 70.0f;   // translacao da Lua
static constexpr float W_SUN_SPIN      = 6.0f;

static constexpr float EARTH_TILT       = 23.4f;  // inclinacao do eixo da Terra
static constexpr float MOON_INCLINATION = 5.0f;   // inclinacao da orbita da Lua

// ---------------------------------------------------------------------------
// Globais da aplicacao
// ---------------------------------------------------------------------------
static WgpuContextPtr ctx;
static RendererPtr    renderer;
static ScenePtr       scene;

static constexpr int  NUM_CAMERAS = 4;
static Camera3DPtr    cameras[NUM_CAMERAS];
static const char*    cameraNames[NUM_CAMERAS] = {
  "1: global (arcball)",
  "2: Terra -> Lua (fases)",
  "3: foco na Terra (arcball)",
  "4: foco em Mercurio (arcball)"
};
static int            activeCamIdx = 0;

static ShaderPtr      planetShader;
static ShaderPtr      sunShader;
static std::unique_ptr<ClipPlane> planetClipPlane;
static std::unique_ptr<ClipPlane> sunClipPlane;
static bool           bumpOn = true;
static float          bumpStrength = 2.2f;   // relevo mais profundo por padrao
static bool           clipOn = false;

static float          timeScale = 1.0f;
static bool           paused = false;

// nos de posicao no mundo usados pelas cameras
static NodePtr        mercuryPosNode;
static NodePtr        earthPosNode;
static NodePtr        moonPosNode;

// texturas, samplers e conjuntos precisam viver tanto quanto a cena
static std::vector<std::shared_ptr<TextureItem>> keepAlive;

// ---------------------------------------------------------------------------
// Engines
// ---------------------------------------------------------------------------

// Gira os pivos de translacao e rotacao de cada astro.
class SolarEngine : public Engine {
  TransformPtr m_sunSpin, m_mercuryOrbit, m_mercurySpin;
  TransformPtr m_earthOrbit, m_earthSpin, m_moonOrbit, m_moonSpin;
protected:
  SolarEngine (TransformPtr sunSpin, TransformPtr mercuryOrbit, TransformPtr mercurySpin,
               TransformPtr earthOrbit, TransformPtr earthSpin,
               TransformPtr moonOrbit, TransformPtr moonSpin)
  : m_sunSpin(sunSpin), m_mercuryOrbit(mercuryOrbit), m_mercurySpin(mercurySpin),
    m_earthOrbit(earthOrbit), m_earthSpin(earthSpin), m_moonOrbit(moonOrbit), m_moonSpin(moonSpin)
  {}
public:
  static EnginePtr Make (TransformPtr sunSpin, TransformPtr mercuryOrbit, TransformPtr mercurySpin,
                         TransformPtr earthOrbit, TransformPtr earthSpin,
                         TransformPtr moonOrbit, TransformPtr moonSpin)
  {
    return EnginePtr(new SolarEngine(sunSpin, mercuryOrbit, mercurySpin, earthOrbit, earthSpin, moonOrbit, moonSpin));
  }

  void Update (float dt) override
  {
    if (paused) return;
    dt *= timeScale;
    m_sunSpin->Rotate(W_SUN_SPIN * dt, 0, 1, 0);
    m_mercuryOrbit->Rotate(W_MERCURY_ORBIT * dt, 0, 1, 0);
    m_mercurySpin->Rotate(W_MERCURY_SPIN * dt, 0, 1, 0);
    m_earthOrbit->Rotate(W_EARTH_ORBIT * dt, 0, 1, 0);
    m_earthSpin->Rotate(W_EARTH_SPIN * dt, 0, 1, 0);
    m_moonOrbit->Rotate(W_MOON_ORBIT * dt, 0, 1, 0);
    // acoplamento de mare: a Lua gira na mesma taxa da orbita
    m_moonSpin->Rotate(W_MOON_ORBIT * dt, 0, 1, 0);
  }
};

static Vec3 WorldPosition (NodePtr node)
{
  Mat4 m = node->GetModelMatrix();
  return Vec3{Mat4At(m, 0, 3), Mat4At(m, 1, 3), Mat4At(m, 2, 3)};
}

// Atualiza a camera posicionada na Terra olhando para a Lua a cada quadro:
// o olho fica logo acima da superficie da Terra na direcao da Lua, e o alvo
// (center) e o centro da Lua, mantendo-a sempre no centro da tela para
// permitir observar suas mudancas de fases e seu relevo.
class EarthCameraEngine : public Engine {
  Camera3DPtr m_camera;
protected:
  EarthCameraEngine (Camera3DPtr camera) : m_camera(camera) {}
public:
  static EnginePtr Make (Camera3DPtr camera) { return EnginePtr(new EarthCameraEngine(camera)); }

  void Update (float dt) override
  {
    (void) dt;
    Vec3 earth = WorldPosition(earthPosNode);
    Vec3 moon = WorldPosition(moonPosNode);
    Vec3 dir = Vec3Normalize(moon - earth);
    Vec3 eye = earth + dir * (R_EARTH + 0.06f);
    m_camera->SetEye(eye.x, eye.y, eye.z);
    m_camera->SetCenter(moon.x, moon.y, moon.z);
    m_camera->SetUpDir(0.0f, 1.0f, 0.0f);
  }
};

// ---------------------------------------------------------------------------
// Auxiliares de construcao
// ---------------------------------------------------------------------------
static std::vector<VertexBufferSpec> SphereVertexBuffers ()
{
  // a esfera do framework entrega 4 buffers: coord, normal, tangente, texcoord
  VertexBufferSpec coord, normal, tangent, texcoord;
  coord.arrayStride = 3 * sizeof(float);
  coord.attributes.push_back({std::string("coord"), std::nullopt, WGPUVertexFormat_Float32x3, 0});
  normal.arrayStride = 3 * sizeof(float);
  normal.attributes.push_back({std::string("normal"), std::nullopt, WGPUVertexFormat_Float32x3, 0});
  tangent.arrayStride = 3 * sizeof(float);
  tangent.attributes.push_back({std::string("tangent"), std::nullopt, WGPUVertexFormat_Float32x3, 0});
  texcoord.arrayStride = 2 * sizeof(float);
  texcoord.attributes.push_back({std::string("texcoord"), std::nullopt, WGPUVertexFormat_Float32x2, 0});
  return {coord, normal, tangent, texcoord};
}

static TransformPtr ScaleTransform (float r)
{
  TransformPtr trf = Transform::Make();
  trf->Scale(r, r, r);
  return trf;
}

static TransformPtr TranslateTransform (float x, float y, float z)
{
  TransformPtr trf = Transform::Make();
  trf->Translate(x, y, z);
  return trf;
}

// ---------------------------------------------------------------------------
// initialize
// ---------------------------------------------------------------------------
static void initialize (void)
{
  WGPUDevice device = ctx->GetDevice();

  // --- 1. Luz: pontual, na posicao do Sol (origem do mundo) ---
  PointLightPtr sunLight = PointLight::Make(0.0f, 0.0f, 0.0f, "world");
  sunLight->SetAmbient(0.05f, 0.05f, 0.07f);
  sunLight->SetDiffuse(1.75f, 1.70f, 1.58f);
  sunLight->SetSpecular(1.0f, 1.0f, 1.0f);

  // --- 2. Shaders e pipelines ---
  planetShader = Shader::Make(device, "../shaders/solar/planet.wgsl", sunLight, "camera");
  planetShader->SetVertexBuffers(SphereVertexBuffers());
  planetShader->SetValue("bump_amount", bumpStrength);

  WGPUPrimitiveState planetPrim = {};
  planetPrim.cullMode = WGPUCullMode_None;   // a face interna mostra a secao do corte
  PipelineDesc planetDesc;
  planetDesc.primitive = planetPrim;
  PipelinePtr planetPipeline = Pipeline::Make(planetShader, ctx->GetSurfaceFormat(), planetDesc);

  planetClipPlane = std::make_unique<ClipPlane>(planetShader, 0.0f, 0.0f, 0.0f, 0.0f);
  planetClipPlane->SetColor(0.85f, 0.45f, 0.20f);

  // Sol: emissivo, sem iluminacao, tambem com suporte a plano de corte
  sunShader = Shader::Make(device, "../shaders/solar/sun.wgsl", nullptr, "camera");
  sunShader->SetVertexBuffers(SphereVertexBuffers());
  WGPUPrimitiveState sunPrim = {};
  sunPrim.cullMode = WGPUCullMode_None;
  PipelineDesc sunDesc;
  sunDesc.primitive = sunPrim;
  PipelinePtr sunPipeline = Pipeline::Make(sunShader, ctx->GetSurfaceFormat(), sunDesc);

  sunClipPlane = std::make_unique<ClipPlane>(sunShader, 0.0f, 0.0f, 0.0f, 0.0f);
  sunClipPlane->SetColor(1.00f, 0.72f, 0.18f);

  // Fundo: cubo recentrado no olho; nao escreve profundidade
  ShaderPtr skyShader = Shader::Make(device, "../shaders/solar/skybox.wgsl", nullptr, "camera");
  VertexBufferSpec skyCoord;
  skyCoord.arrayStride = 3 * sizeof(float);
  skyCoord.attributes.push_back({std::string("coord"), std::nullopt, WGPUVertexFormat_Float32x3, 0});
  skyShader->SetVertexBuffers({skyCoord});
  WGPUPrimitiveState skyPrim = {};
  skyPrim.cullMode = WGPUCullMode_None;
  PipelineDesc skyDesc;
  skyDesc.primitive = skyPrim;
  skyDesc.depthStencilMode = DepthStencilMode::Explicit;
  skyDesc.depthStencil.format = WGPUTextureFormat_Depth24Plus;
  skyDesc.depthStencil.depthWriteEnabled = WGPUOptionalBool_False;
  skyDesc.depthStencil.depthCompare = WGPUCompareFunction_LessEqual;
  skyDesc.depthStencil.stencilReadMask = 0xFFFFFFFF;
  skyDesc.depthStencil.stencilWriteMask = 0xFFFFFFFF;
  PipelinePtr skyPipeline = Pipeline::Make(skyShader, ctx->GetSurfaceFormat(), skyDesc);

  // --- 3. Texturas ---
  // cor: rgba8unorm-srgb (padrao). Mapas de normais: DADO, nao cor -> rgba8unorm.
  auto color = [&](const char* var, const char* file) {
    TexturePtr t = Texture::Make(device, var, file);
    keepAlive.push_back(t);
    return t;
  };
  auto normalMap = [&](const char* var, const char* file) {
    TexturePtr t = Texture::Make(device, var, file, WGPUTextureFormat_RGBA8Unorm);
    keepAlive.push_back(t);
    return t;
  };
  SamplerPtr decalSampler = Sampler::Make(device, "decal_sampler");
  SamplerPtr normalSampler = Sampler::Make(device, "normal_sampler");
  SamplerPtr skySampler = Sampler::Make(device, "sky_sampler",
                                        WGPUAddressMode_ClampToEdge, WGPUAddressMode_ClampToEdge);
  keepAlive.push_back(decalSampler);
  keepAlive.push_back(normalSampler);
  keepAlive.push_back(skySampler);

  TexturePtr sunColor = color("decal_texture", "../images/sun.jpg");
  TexturePtr earthColor = color("decal_texture", "../images/earth.jpg");
  TexturePtr moonColor = color("decal_texture", "../images/moon.jpg");
  TexturePtr mercuryColor = color("decal_texture", "../images/mercury.jpg");
  TexturePtr earthNormal = normalMap("normal_texture", "../images/earth-normal.png");
  TexturePtr moonNormal = normalMap("normal_texture", "../images/moon_normal.png");
  TexturePtr mercuryNormal = normalMap("normal_texture", "../images/mercury_normal.png");
  TexCubePtr spaceCube = TexCube::Make(device, "sky_texture", "../images/space_cross.png");
  keepAlive.push_back(spaceCube);

  TextureSetPtr sunSet = TextureSet::Make({sunColor.get(), decalSampler.get()});
  TextureSetPtr earthSet = TextureSet::Make({earthColor.get(), decalSampler.get(), earthNormal.get(), normalSampler.get()});
  TextureSetPtr moonSet = TextureSet::Make({moonColor.get(), decalSampler.get(), moonNormal.get(), normalSampler.get()});
  TextureSetPtr mercurySet = TextureSet::Make({mercuryColor.get(), decalSampler.get(), mercuryNormal.get(), normalSampler.get()});
  TextureSetPtr skySet = TextureSet::Make({spaceCube.get(), skySampler.get()});

  sunShader->AddTextureSet(sunSet.get());
  planetShader->AddTextureSet(earthSet.get());
  planetShader->AddTextureSet(moonSet.get());
  planetShader->AddTextureSet(mercurySet.get());
  skyShader->AddTextureSet(skySet.get());

  // --- 4. Materiais (a cor vem da textura: base branca) ---
  PhongMaterialPtr matEarth = PhongMaterial::Make(1.0f, 1.0f, 1.0f);
  matEarth->SetSpecular(0.28f, 0.28f, 0.34f);
  matEarth->SetShininess(24.0f);
  PhongMaterialPtr matMoon = PhongMaterial::Make(1.0f, 1.0f, 1.0f);
  matMoon->SetSpecular(0.06f, 0.06f, 0.06f);
  matMoon->SetShininess(10.0f);
  PhongMaterialPtr matMercury = PhongMaterial::Make(1.0f, 1.0f, 1.0f);
  matMercury->SetSpecular(0.10f, 0.09f, 0.08f);
  matMercury->SetShininess(12.0f);
  planetShader->AddMaterial(matEarth.get());
  planetShader->AddMaterial(matMoon.get());
  planetShader->AddMaterial(matMercury.get());

  // --- 5. Geometrias ---
  SpherePtr sphere = Sphere::Make(device, 64, 128);
  SkyBoxPtr skybox = SkyBox::Make(device);

  // =========================================================================
  // Grafo de cena
  // =========================================================================

  // Fundo
  NodePtr skyNode = Node::Make(skyPipeline, SkyBoxTransform::Make(), {skySet}, {skybox});

  // Sol (fixo no centro, gira devagar)
  TransformPtr trfSunSpin = Transform::Make();
  NodePtr sunBody = Node::Make(ScaleTransform(R_SUN), {}, {sphere});
  NodePtr sunNode = Node::Make(sunPipeline, trfSunSpin, {sunSet}, {sunBody});

  // Mercurio: orbit -> pos -> spin -> corpo
  TransformPtr trfMercuryOrbit = Transform::Make();
  TransformPtr trfMercurySpin = Transform::Make();
  NodePtr mercuryBody = Node::Make(ScaleTransform(R_MERCURY), {matMercury, mercurySet}, {sphere});
  NodePtr mercurySpinNode = Node::Make(trfMercurySpin, {mercuryBody});
  mercuryPosNode = Node::Make(TranslateTransform(D_MERCURY, 0.0f, 0.0f), {mercurySpinNode});
  NodePtr mercuryOrbitNode = Node::Make(trfMercuryOrbit, {mercuryPosNode});

  // Terra: orbit -> pos -> { tilt -> spin -> corpo ; Lua }
  TransformPtr trfEarthOrbit = Transform::Make();
  TransformPtr trfEarthSpin = Transform::Make();
  TransformPtr trfEarthTilt = Transform::Make();
  trfEarthTilt->Rotate(EARTH_TILT, 0.0f, 0.0f, 1.0f);
  NodePtr earthBody = Node::Make(ScaleTransform(R_EARTH), {matEarth, earthSet}, {sphere});
  NodePtr earthSpinNode = Node::Make(trfEarthSpin, {earthBody});
  NodePtr earthTiltNode = Node::Make(trfEarthTilt, {earthSpinNode});

  // Lua: orbita levemente inclinada (a inclinacao vem antes da rotacao do pivo)
  TransformPtr trfMoonOrbit = Transform::Make();
  trfMoonOrbit->Rotate(MOON_INCLINATION, 1.0f, 0.0f, 0.0f);
  TransformPtr trfMoonSpin = Transform::Make();
  NodePtr moonBody = Node::Make(ScaleTransform(R_MOON), {matMoon, moonSet}, {sphere});
  NodePtr moonSpinNode = Node::Make(trfMoonSpin, {moonBody});
  moonPosNode = Node::Make(TranslateTransform(D_MOON, 0.0f, 0.0f), {moonSpinNode});
  NodePtr moonOrbitNode = Node::Make(trfMoonOrbit, {moonPosNode});

  earthPosNode = Node::Make(TranslateTransform(D_EARTH, 0.0f, 0.0f), {earthTiltNode, moonOrbitNode});
  NodePtr earthOrbitNode = Node::Make(trfEarthOrbit, {earthPosNode});

  NodePtr planets = Node::Make(planetPipeline, {mercuryOrbitNode, earthOrbitNode});

  NodePtr root = Node::Make({skyNode, sunNode, planets});
  scene = Scene::Make(root);

  // --- 6. Cameras ---
  // Camera 0: Global (com arcball ao redor da origem)
  cameras[0] = Camera3D::Make(0.0f, 5.5f, 11.0f);
  cameras[0]->SetCenter(0.0f, 0.0f, 0.0f);
  cameras[0]->SetZPlanes(0.1f, 200.0f);
  cameras[0]->CreateArcball();

  // Camera 1: Posicionada na Terra olhando para a Lua (atualizada pela EarthCameraEngine)
  cameras[1] = Camera3D::Make(D_EARTH, 0.0f, 0.0f);
  cameras[1]->SetAngle(24.0f);
  cameras[1]->SetZPlanes(0.02f, 200.0f);

  // Camera 2: Close-up focado na Terra (referenciada ao no de posicao da Terra + arcball local)
  cameras[2] = Camera3D::Make(-0.85f, 0.32f, 1.15f);
  cameras[2]->SetCenter(0.0f, 0.0f, 0.0f);
  cameras[2]->SetAngle(35.0f);
  cameras[2]->SetZPlanes(0.02f, 200.0f);
  cameras[2]->SetReference(earthPosNode);
  cameras[2]->CreateArcball();

  // Camera 3: Close-up focado em Mercurio (referenciada ao no de posicao de Mercurio + arcball local)
  cameras[3] = Camera3D::Make(-0.32f, 0.12f, 0.42f);
  cameras[3]->SetCenter(0.0f, 0.0f, 0.0f);
  cameras[3]->SetAngle(35.0f);
  cameras[3]->SetZPlanes(0.02f, 200.0f);
  cameras[3]->SetReference(mercuryPosNode);
  cameras[3]->CreateArcball();

  // --- 7. Engines ---
  scene->AddEngine(SolarEngine::Make(trfSunSpin, trfMercuryOrbit, trfMercurySpin,
                                     trfEarthOrbit, trfEarthSpin, trfMoonOrbit, trfMoonSpin));
  scene->AddEngine(EarthCameraEngine::Make(cameras[1]));
}

// ---------------------------------------------------------------------------
// Quadro e interacao
// ---------------------------------------------------------------------------
static Camera3DPtr activeCamera ()
{
  return cameras[activeCamIdx];
}

static void updateEffects ()
{
  planetShader->SetValue("bump_amount", bumpOn ? bumpStrength : 0.0f);

  if (!clipOn) {
    planetClipPlane->SetPlane(0.0f, 0.0f, 0.0f, 0.0f);
    sunClipPlane->SetPlane(0.0f, 0.0f, 0.0f, 0.0f);
    return;
  }
  // plano do mundo y = 0 (plano das orbitas), descartando a metade de cima
  // (n.p + d < 0): transformado para o espaco da camera ativa por (V^-1)^T
  Mat4 view = activeCamera()->GetViewMatrix();
  Mat4 invT = Mat4Transpose(Mat4Inverse(view));
  Vec4 p = Mat4MulVec4(invT, Vec4{0.0f, -1.0f, 0.0f, 0.0f});
  planetClipPlane->SetPlane(p.x, p.y, p.z, p.w);
  sunClipPlane->SetPlane(p.x, p.y, p.z, p.w);
}

static void display ()
{
  WGPUSurfaceTexture surfaceTex = ctx->GetCurrentTexture();
  if (surfaceTex.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
      surfaceTex.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal)
    return;
  updateEffects();
  renderer->Render(surfaceTex.texture, scene, activeCamera());
  ctx->Present();
  wgpuTextureRelease(surfaceTex.texture);
}

static void error (int code, const char* msg)
{
  printf("GLFW error %d: %s\n", code, msg);
  glfwTerminate();
  exit(0);
}

static void updateTitle (GLFWwindow* win)
{
  char title[200];
  snprintf(title, sizeof(title),
           "INF1761 - Projeto 2: Sistema Solar 3D [%s] [relevo: %s (%.1fx)]%s",
           cameraNames[activeCamIdx],
           bumpOn ? "ON" : "OFF", bumpStrength,
           clipOn ? " [corte ATIVO]" : "");
  glfwSetWindowTitle(win, title);
}

static void keyboard (GLFWwindow* window, int key, int scancode, int action, int mods)
{
  (void) scancode; (void) mods;
  if (action == GLFW_RELEASE) return;
  bool press = (action == GLFW_PRESS);

  if ((key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE) && press)
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  else if (key == GLFW_KEY_C && press)
    activeCamIdx = (activeCamIdx + 1) % NUM_CAMERAS;
  else if (key == GLFW_KEY_1 && press) activeCamIdx = 0;
  else if (key == GLFW_KEY_2 && press) activeCamIdx = 1;
  else if (key == GLFW_KEY_3 && press) activeCamIdx = 2;
  else if (key == GLFW_KEY_4 && press) activeCamIdx = 3;
  else if (key == GLFW_KEY_B && press) bumpOn = !bumpOn;
  else if (key == GLFW_KEY_LEFT_BRACKET)
    bumpStrength = std::max(0.4f, bumpStrength - 0.3f);
  else if (key == GLFW_KEY_RIGHT_BRACKET)
    bumpStrength = std::min(5.0f, bumpStrength + 0.3f);
  else if (key == GLFW_KEY_X && press) clipOn = !clipOn;
  else if (key == GLFW_KEY_P && press) paused = !paused;
  else if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD)
    timeScale = std::min(timeScale * 1.25f, 20.0f);
  else if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT)
    timeScale = std::max(timeScale / 1.25f, 0.05f);
  updateTitle(window);
}

static void scroll (GLFWwindow* win, double xoffset, double yoffset)
{
  (void) xoffset;
  Camera3DPtr cam = activeCamera();
  float fovy = cam->GetAngle() - (float) yoffset * 2.5f;
  cam->SetAngle(std::clamp(fovy, 6.0f, 75.0f));
  updateTitle(win);
}

static void resize (GLFWwindow* win, int width, int height)
{
  (void) win;
  if (width > 0 && height > 0) ctx->Configure(width, height);
}

// Arcball opera sobre a camera ativa (quando ela possui arcball)
static void cursorpos (GLFWwindow* win, double x, double y)
{
  ArcballPtr ab = activeCamera()->GetArcball();
  if (!ab) return;
  int wn_w, wn_h, fb_w, fb_h;
  glfwGetWindowSize(win, &wn_w, &wn_h);
  glfwGetFramebufferSize(win, &fb_w, &fb_h);
  x = x * fb_w / wn_w;
  y = (wn_h - y) * fb_h / wn_h;
  ab->AccumulateMouseMotion((float) x, (float) y, (float) fb_w, (float) fb_h);
}

static void cursorinit (GLFWwindow* win, double x, double y)
{
  ArcballPtr ab = activeCamera()->GetArcball();
  if (!ab) return;
  int wn_w, wn_h, fb_w, fb_h;
  glfwGetWindowSize(win, &wn_w, &wn_h);
  glfwGetFramebufferSize(win, &fb_w, &fb_h);
  x = x * fb_w / wn_w;
  y = (wn_h - y) * fb_h / wn_h;
  ab->InitMouseMotion((float) x, (float) y);
  glfwSetCursorPosCallback(win, cursorpos);
}

static void mousebutton (GLFWwindow* win, int button, int action, int mods)
{
  (void) button; (void) mods;
  if (action == GLFW_PRESS)
    glfwSetCursorPosCallback(win, cursorinit);
  else
    glfwSetCursorPosCallback(win, nullptr);
}

int main ()
{
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
#ifdef __APPLE__
  glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_TRUE);
#endif

  glfwSetErrorCallback(error);

  GLFWwindow* win = glfwCreateWindow(1000, 700, "INF1761 - Projeto 2: Sistema Solar 3D", nullptr, nullptr);
  assert(win);
  glfwSetFramebufferSizeCallback(win, resize);
  glfwSetKeyCallback(win, keyboard);
  glfwSetMouseButtonCallback(win, mousebutton);
  glfwSetScrollCallback(win, scroll);

  int fbw, fbh;
  glfwGetFramebufferSize(win, &fbw, &fbh);
  ctx = WgpuContext::Make(win, fbw, fbh);

  renderer = Renderer::Make(ctx->GetDevice(), true, {0.0f, 0.0f, 0.0f, 1.0f});

  initialize();
  updateTitle(win);

  printf("Controles:\n");
  printf("  C ou 1..4 : troca de camera (1=Global, 2=Terra->Lua, 3=Terra, 4=Mercurio)\n");
  printf("  Mouse     : arrastar gira o arcball | Scroll da zoom in/out\n");
  printf("  B         : liga/desliga relevo     | [ e ] ajustam a profundidade do relevo\n");
  printf("  X         : liga/desliga plano de corte\n");
  printf("  P         : pausa | +/- : velocidade | Q/Esc : sai\n");

  float t0 = (float) glfwGetTime();
  while (!glfwWindowShouldClose(win)) {
    float t = (float) glfwGetTime();
    float dt = std::min(t - t0, 0.1f);
    t0 = t;
    scene->Update(dt);
    display();
    glfwPollEvents();
  }

  ctx = nullptr;
  glfwDestroyWindow(win);
  glfwTerminate();
  return 0;
}
