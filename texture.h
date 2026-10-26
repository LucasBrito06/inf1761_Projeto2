#include <memory>
class Texture;
using TexturePtr = std::shared_ptr<Texture>;

#ifndef TEXTURE_H
#define TEXTURE_H

#include <wgpu.h>
#include <string>
#include "textureset.h"
#include "graphicsmath.h"

// 2D texture bound to varname. By default it is stored as
// rgba8unorm-srgb (right for color images); pass WGPUTextureFormat_RGBA8Unorm
// for data textures such as normal maps, which must NOT go through the sRGB
// curve. Content comes from one source: a filename (loads an image), a solid
// texel color, or a blank width x height texture. A pure resource-holder -
// no Load/Unload of its own, see TextureSet.
class Texture : public TextureItem {
  std::string m_varname;
  int m_width, m_height;
  WGPUTexture m_tex;
  WGPUTextureView m_view;

protected:
  Texture (WGPUDevice device, const std::string& varname, int width, int height, const uint8_t* rgba,
           WGPUTextureFormat format);

public:
  static TexturePtr Make (WGPUDevice device, const std::string& varname, const std::string& filename,
                          WGPUTextureFormat format = WGPUTextureFormat_RGBA8UnormSrgb);
  static TexturePtr Make (WGPUDevice device, const std::string& varname, const Vec3& texel,
                          WGPUTextureFormat format = WGPUTextureFormat_RGBA8UnormSrgb);
  static TexturePtr Make (WGPUDevice device, const std::string& varname, const Vec4& texel);
  static TexturePtr Make (WGPUDevice device, const std::string& varname, int width = 1, int height = 1);
  ~Texture ();

  WGPUTexture GetTexture () const;
  int GetWidth () const;
  int GetHeight () const;

  const std::string& GetVarName () const override;
  WGPUBindGroupEntry MakeEntry (uint32_t binding) const override;
};

#endif
