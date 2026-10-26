#include "texture.h"
#include "error.h"
#include <vector>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Texture::Texture (WGPUDevice device, const std::string& varname, int width, int height, const uint8_t* rgba,
                  WGPUTextureFormat format)
  : m_varname(varname), m_width(width), m_height(height)
{
  WGPUTextureDescriptor texDesc = {};
  texDesc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
  texDesc.dimension = WGPUTextureDimension_2D;
  texDesc.size = {(uint32_t) width, (uint32_t) height, 1};
  texDesc.format = format;
  texDesc.mipLevelCount = 1;
  texDesc.sampleCount = 1;
  m_tex = wgpuDeviceCreateTexture(device, &texDesc);

  WGPUTexelCopyTextureInfo dst = {};
  dst.texture = m_tex;
  WGPUTexelCopyBufferLayout layout = {};
  layout.bytesPerRow = (uint32_t) width * 4;
  layout.rowsPerImage = (uint32_t) height;
  WGPUExtent3D writeSize = {(uint32_t) width, (uint32_t) height, 1};

  WGPUQueue queue = wgpuDeviceGetQueue(device);
  wgpuQueueWriteTexture(queue, &dst, rgba, (size_t) width * height * 4, &layout, &writeSize);
  wgpuQueueRelease(queue);

  m_view = wgpuTextureCreateView(m_tex, nullptr);
}

TexturePtr Texture::Make (WGPUDevice device, const std::string& varname, const std::string& filename,
                          WGPUTextureFormat format)
{
  int w, h, channels;
  unsigned char* pixels = stbi_load(filename.c_str(), &w, &h, &channels, 4);
  if (!pixels) Error::Fatal("could not load image: " + filename);
  // sem inverter: a linha 0 da imagem e a linha 0 da textura, ou seja
  // (s,t) = (0,0) e o canto SUPERIOR esquerdo, como manda a convencao da
  // WebGPU. Toda geometria daqui (Disk, Square, Quad, Cube, Sphere) gera
  // t crescendo para baixo, de acordo.
  TexturePtr tex(new Texture(device, varname, w, h, pixels, format));
  stbi_image_free(pixels);
  return tex;
}

TexturePtr Texture::Make (WGPUDevice device, const std::string& varname, const Vec3& texel,
                          WGPUTextureFormat format)
{
  uint8_t rgba[4] = {(uint8_t) (texel.x * 255), (uint8_t) (texel.y * 255), (uint8_t) (texel.z * 255), 255};
  return TexturePtr(new Texture(device, varname, 1, 1, rgba, format));
}

TexturePtr Texture::Make (WGPUDevice device, const std::string& varname, const Vec4& texel)
{
  uint8_t rgba[4] = {(uint8_t) (texel.x * 255), (uint8_t) (texel.y * 255), (uint8_t) (texel.z * 255), (uint8_t) (texel.w * 255)};
  return TexturePtr(new Texture(device, varname, 1, 1, rgba, WGPUTextureFormat_RGBA8UnormSrgb));
}

TexturePtr Texture::Make (WGPUDevice device, const std::string& varname, int width, int height)
{
  std::vector<uint8_t> blank((size_t) width * height * 4, 0);
  return TexturePtr(new Texture(device, varname, width, height, blank.data(), WGPUTextureFormat_RGBA8UnormSrgb));
}

Texture::~Texture ()
{
  wgpuTextureViewRelease(m_view);
  wgpuTextureRelease(m_tex);
}

WGPUTexture Texture::GetTexture () const { return m_tex; }
int Texture::GetWidth () const { return m_width; }
int Texture::GetHeight () const { return m_height; }
const std::string& Texture::GetVarName () const { return m_varname; }

WGPUBindGroupEntry Texture::MakeEntry (uint32_t binding) const
{
  WGPUBindGroupEntry entry = {};
  entry.binding = binding;
  entry.textureView = m_view;
  return entry;
}
