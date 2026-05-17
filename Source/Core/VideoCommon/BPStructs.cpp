// Copyright 2009 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/BPStructs.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "Common/CommonTypes.h"
#include "Common/EnumMap.h"
#include "Common/FileUtil.h"
#include "Common/Logging/Log.h"

#include "Core/DolphinAnalytics.h"
#include "Core/FifoPlayer/FifoPlayer.h"
#include "Core/FifoPlayer/FifoRecorder.h"
#include "Core/HW/Memmap.h"
#include "Core/HW/VideoInterface.h"
#include "Core/System.h"

#include "VideoCommon/BPFunctions.h"
#include "VideoCommon/BPMemory.h"
#include "VideoCommon/BoundingBox.h"
#include "VideoCommon/Fifo.h"
#include "VideoCommon/FramebufferManager.h"
#include "VideoCommon/GeometryShaderManager.h"
#include "VideoCommon/OpcodeDecoding.h"
#include "VideoCommon/PerfQueryBase.h"
#include "VideoCommon/PixelEngine.h"
#include "VideoCommon/PixelShaderManager.h"
#include "VideoCommon/Present.h"
#include "VideoCommon/SMGPCParityTrace.h"
#include "VideoCommon/Statistics.h"
#include "VideoCommon/TMEM.h"
#include "VideoCommon/TextureCacheBase.h"
#include "VideoCommon/TextureDecoder.h"
#include "VideoCommon/TextureInfo.h"
#include "VideoCommon/VideoBackendBase.h"
#include "VideoCommon/VideoCommon.h"
#include "VideoCommon/VideoConfig.h"
#include "VideoCommon/VideoEvents.h"
#include "VideoCommon/XFStateManager.h"

using namespace BPFunctions;

static constexpr Common::EnumMap<float, GammaCorrection::Invalid2_2> s_gammaLUT = {1.0f, 1.7f, 2.2f,
                                                                                   2.2f};

namespace
{
struct SMGPCDolphinCopyTraceEvent
{
  u64 event_index = 0;
  int presenter_frame_count = 0;
  bool copy_to_xfb = false;
  bool depth_copy = false;
  bool clear = false;
  bool half_scale = false;
  bool scale_invert = false;
  bool clamp_top = false;
  bool clamp_bottom = false;
  bool intensity_format = false;
  bool auto_conversion = false;
  u32 dest_addr = 0;
  u32 dest_stride = 0;
  u32 src_left = 0;
  u32 src_top = 0;
  u32 src_right = 0;
  u32 src_bottom = 0;
  u32 copy_width = 0;
  u32 copy_height = 0;
  u32 output_width = 0;
  u32 output_height = 0;
  u32 target_pixel_format = 0;
  u32 real_format = 0;
  u32 frame_to_field = 0;
  u32 gamma_index = 0;
  float gamma_value = 1.0f;
  float y_scale = 1.0f;
  u32 dispcopyyscale = 0;
  u32 scissor_tl_raw = 0;
  u32 scissor_br_raw = 0;
  u32 scissor_offset_raw = 0;
  float viewport_left = 0.0f;
  float viewport_right = 0.0f;
  float viewport_top = 0.0f;
  float viewport_bottom = 0.0f;
  float viewport_near_depth = 0.0f;
  float viewport_far_depth = 0.0f;
  int backbuffer_width = 0;
  int backbuffer_height = 0;
  int target_left = 0;
  int target_top = 0;
  int target_right = 0;
  int target_bottom = 0;
};

struct SMGPCDolphinTevOrderTraceState
{
  u32 stage = 0;
  bool texture_enabled = false;
  u32 tex_coord = 0;
  u32 tex_map = 0;
  u32 color_channel = 0;
};

struct SMGPCDolphinTevStageTraceState
{
  u32 stage = 0;
  u32 color_raw = 0;
  u32 alpha_raw = 0;
};

struct SMGPCDolphinColorChannelTraceState
{
  u32 channel = 0;
  u32 color_raw = 0;
  u32 alpha_raw = 0;
  u32 color_light_mask = 0;
  u32 alpha_light_mask = 0;
};

struct SMGPCDolphinLightTraceState
{
  bool active = false;
  u32 index = 0;
  std::array<u8, 4> color_rgba{};
  std::array<u8, 4> color_abgr{};
  std::array<float, 3> cosine_attenuation{};
  std::array<float, 3> distance_attenuation{};
  std::array<float, 3> position{};
  std::array<float, 3> direction{};
};

struct SMGPCDolphinTextureTraceState
{
  bool active = false;
  u32 slot = 0;
  u32 address = 0;
  u32 width = 0;
  u32 height = 0;
  u32 expanded_width = 0;
  u32 expanded_height = 0;
  u32 texture_size = 0;
  u32 texture_format = 0;
  u32 tlut_format = 0;
  u32 wrap_s = 0;
  u32 wrap_t = 0;
  u32 min_filter = 0;
  u32 mag_filter = 0;
  u32 mipmap_filter = 0;
  s32 lod_bias = 0;
  u32 min_lod = 0;
  u32 max_lod = 0;
  bool from_tmem = false;
  bool data_valid = false;
  bool has_mipmaps = false;
  u32 tex_mode0_raw = 0;
  u32 tex_mode1_raw = 0;
  u32 tex_image0_raw = 0;
  u32 tex_image1_raw = 0;
  u32 tex_image2_raw = 0;
  u32 tex_image3_raw = 0;
  u32 tex_tlut_raw = 0;
  std::string texture_format_name;
  std::string tlut_format_name;
  std::string identity_name;
};

struct SMGPCDolphinDrawTraceEvent
{
  u32 draw_index = 0;
  int presenter_frame_count = 0;
  PrimitiveType primitive_type = PrimitiveType::Triangles;
  u32 num_vertices = 0;
  u32 num_indices = 0;
  u32 base_vertex = 0;
  u32 base_index = 0;
  u32 used_textures_mask = 0;
  u32 texgen_count = 0;
  u32 color_channel_count = 0;
  u32 tev_stage_count = 0;
  u32 indirect_stage_count = 0;
  u32 gen_mode_raw = 0;
  u32 cull_mode = 0;
  u32 z_mode_raw = 0;
  bool z_compare_enable = false;
  u32 z_function = 0;
  bool z_update_enable = false;
  u32 blend_mode_raw = 0;
  bool blend_enable = false;
  bool logic_op_enable = false;
  bool color_update = false;
  bool alpha_update = false;
  u32 src_blend_factor = 0;
  u32 dst_blend_factor = 0;
  u32 alpha_compare_raw = 0;
  u32 alpha_comp0 = 0;
  u32 alpha_ref0 = 0;
  u32 alpha_op = 0;
  u32 alpha_comp1 = 0;
  u32 alpha_ref1 = 0;
  u32 fog_raw[5] = {};
  std::array<SMGPCDolphinTevOrderTraceState, 16> tev_orders{};
  std::array<SMGPCDolphinTevStageTraceState, 16> tev_stages{};
  std::array<SMGPCDolphinColorChannelTraceState, 2> color_channels{};
  std::array<SMGPCDolphinLightTraceState, 8> lights{};
  std::array<SMGPCDolphinTextureTraceState, 8> textures{};
  u32 requested_light_mask = 0;
  u32 loaded_light_mask = 0;
};

std::vector<SMGPCDolphinCopyTraceEvent> s_smgpc_dolphin_copy_trace_events;
std::vector<SMGPCDolphinDrawTraceEvent> s_smgpc_dolphin_draw_trace_events;
u64 s_smgpc_dolphin_copy_trace_event_index = 0;
bool s_smgpc_dolphin_copy_trace_completed = false;

std::optional<int> GetSMGPCDolphinTraceFrame()
{
  const char* value = std::getenv("SMGPC_DOLPHIN_TRACE_FRAME");
  if (value == nullptr || value[0] == '\0')
    return std::nullopt;

  const auto text = std::string_view(value);
  int frame = 0;
  const auto* begin = text.data();
  const auto* end = begin + text.size();
  const auto result = std::from_chars(begin, end, frame);
  if (result.ec != std::errc{} || result.ptr != end || frame < 0)
    return std::nullopt;

  return frame;
}

std::optional<int> GetSMGPCDolphinTraceWindow()
{
  const char* value = std::getenv("SMGPC_DOLPHIN_TRACE_WINDOW");
  if (value == nullptr || value[0] == '\0')
    return 0;

  const auto text = std::string_view(value);
  int window = 0;
  const auto* begin = text.data();
  const auto* end = begin + text.size();
  const auto result = std::from_chars(begin, end, window);
  if (result.ec != std::errc{} || result.ptr != end || window < 0)
    return 0;

  return window;
}

std::optional<std::string> GetSMGPCDolphinTracePath()
{
  const char* value = std::getenv("SMGPC_DOLPHIN_TRACE_PATH");
  if (value == nullptr || value[0] == '\0')
    return std::nullopt;

  return std::string(value);
}

std::optional<std::string> GetSMGPCDolphinSemanticAnchorName()
{
  const char* value = std::getenv("SMGPC_DOLPHIN_SEMANTIC_ANCHOR_NAME");
  if (value == nullptr || value[0] == '\0')
    return std::nullopt;

  return std::string(value);
}

std::string GetSMGPCDolphinSemanticAnchorCategory()
{
  const char* value = std::getenv("SMGPC_DOLPHIN_SEMANTIC_ANCHOR_CATEGORY");
  if (value == nullptr || value[0] == '\0')
    return "capture";

  return std::string(value);
}

std::string GetSMGPCDolphinSemanticAnchorDetail(int requested_frame)
{
  const char* value = std::getenv("SMGPC_DOLPHIN_SEMANTIC_ANCHOR_DETAIL");
  if (value != nullptr && value[0] != '\0')
    return std::string(value);

  return "requested_frame=" + std::to_string(requested_frame);
}

const char* BoolJson(bool value)
{
  return value ? "true" : "false";
}

const char* PrimitiveTypeName(PrimitiveType primitive_type)
{
  switch (primitive_type)
  {
  case PrimitiveType::Points:
    return "points";
  case PrimitiveType::Lines:
    return "lines";
  case PrimitiveType::Triangles:
    return "triangles";
  case PrimitiveType::TriangleStrip:
    return "triangle_strip";
  }

  return "unknown";
}

const char* TextureFormatName(TextureFormat format)
{
  switch (format)
  {
  case TextureFormat::I4:
    return "I4";
  case TextureFormat::I8:
    return "I8";
  case TextureFormat::IA4:
    return "IA4";
  case TextureFormat::IA8:
    return "IA8";
  case TextureFormat::RGB565:
    return "RGB565";
  case TextureFormat::RGB5A3:
    return "RGB5A3";
  case TextureFormat::RGBA8:
    return "RGBA8";
  case TextureFormat::C4:
    return "C4";
  case TextureFormat::C8:
    return "C8";
  case TextureFormat::C14X2:
    return "C14X2";
  case TextureFormat::CMPR:
    return "CMPR";
  case TextureFormat::XFB:
    return "XFB";
  }

  return "Unknown";
}

const char* TlutFormatName(TLUTFormat format)
{
  switch (format)
  {
  case TLUTFormat::IA8:
    return "IA8";
  case TLUTFormat::RGB565:
    return "RGB565";
  case TLUTFormat::RGB5A3:
    return "RGB5A3";
  }

  return "Unknown";
}

void WriteUsedTextureSlots(std::ostream& out, u32 used_textures_mask)
{
  bool needs_comma = false;
  for (u32 slot = 0; slot < 8; ++slot)
  {
    if ((used_textures_mask & (1u << slot)) == 0)
      continue;

    if (needs_comma)
      out << ", ";
    out << slot;
    needs_comma = true;
  }
}

void WriteJsonFloat(std::ostream& out, float value)
{
  if (!std::isfinite(value))
  {
    out << "null";
    return;
  }

  out << value;
}

void WriteFloat3(std::ostream& out, const std::array<float, 3>& values)
{
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i)
  {
    if (i != 0)
      out << ", ";
    WriteJsonFloat(out, values[i]);
  }
  out << ']';
}

void WriteColor4(std::ostream& out, const std::array<u8, 4>& values)
{
  out << '[' << static_cast<u32>(values[0]) << ", " << static_cast<u32>(values[1]) << ", "
      << static_cast<u32>(values[2]) << ", " << static_cast<u32>(values[3]) << ']';
}

void WriteJsonString(std::ostream& out, std::string_view value)
{
  out << '"';
  for (const char c : value)
  {
    switch (c)
    {
    case '\\':
      out << "\\\\";
      break;
    case '"':
      out << "\\\"";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      out << c;
      break;
    }
  }
  out << '"';
}

void WriteTextureBinding(const SMGPCDolphinTextureTraceState& texture, std::ostream& out)
{
  out << "        {\"slot\": " << texture.slot
      << ", \"address\": " << texture.address
      << ", \"width\": " << texture.width
      << ", \"height\": " << texture.height
      << ", \"expanded_width\": " << texture.expanded_width
      << ", \"expanded_height\": " << texture.expanded_height
      << ", \"texture_size\": " << texture.texture_size
      << ", \"format\": \"" << texture.texture_format_name << "\""
      << ", \"format_raw\": " << texture.texture_format
      << ", \"tlut_format\": \"" << texture.tlut_format_name << "\""
      << ", \"tlut_format_raw\": " << texture.tlut_format
      << ", \"wrap_s\": " << texture.wrap_s
      << ", \"wrap_t\": " << texture.wrap_t
      << ", \"min_filter\": " << texture.min_filter
      << ", \"mag_filter\": " << texture.mag_filter
      << ", \"mipmap_filter\": " << texture.mipmap_filter
      << ", \"lod_bias\": " << texture.lod_bias
      << ", \"min_lod\": " << texture.min_lod
      << ", \"max_lod\": " << texture.max_lod
      << ", \"from_tmem\": " << BoolJson(texture.from_tmem)
      << ", \"data_valid\": " << BoolJson(texture.data_valid)
      << ", \"has_mipmaps\": " << BoolJson(texture.has_mipmaps)
      << ", \"tex_mode0_raw\": " << texture.tex_mode0_raw
      << ", \"tex_mode1_raw\": " << texture.tex_mode1_raw
      << ", \"tex_image0_raw\": " << texture.tex_image0_raw
      << ", \"tex_image1_raw\": " << texture.tex_image1_raw
      << ", \"tex_image2_raw\": " << texture.tex_image2_raw
      << ", \"tex_image3_raw\": " << texture.tex_image3_raw
      << ", \"tex_tlut_raw\": " << texture.tex_tlut_raw
      << ", \"identity_name\": ";
  WriteJsonString(out, texture.identity_name);
  out << '}';
}

void WriteSMGPCTraceRecordPrefix(std::ostream& out, std::string_view record_type, int requested_frame)
{
  out << "{\"schema\":\"smgpc-trace-ndjson-v1\",\"record_type\":";
  WriteJsonString(out, record_type);
  out << ",\"emulator\":\"dolphin\",\"frame_index\":" << requested_frame;
}

void WriteSMGPCDolphinDrawTraceEventPayload(std::ostream& out,
                                            const SMGPCDolphinDrawTraceEvent& event,
                                            std::size_t event_index)
{
  const auto tev_stage_count = std::min<std::size_t>(event.tev_stage_count, 16);
  const auto color_channel_count = std::min<std::size_t>(event.color_channel_count, 2);

  out << "{\"index\": " << event_index
      << ", \"draw_index\": " << event.draw_index
      << ", \"presenter_frame_count\": " << event.presenter_frame_count
      << ", \"frame_index\": " << event.presenter_frame_count
      << ", \"model_name\": null"
      << ", \"material_name\": null"
      << ", \"render_pass\": \"GXDraw\""
      << ", \"view_id\": 0"
      << ", \"primitive_type\": ";
  WriteJsonString(out, PrimitiveTypeName(event.primitive_type));
  out << ", \"primitive_type_raw\": " << static_cast<u32>(event.primitive_type)
      << ", \"num_vertices\": " << event.num_vertices
      << ", \"num_indices\": " << event.num_indices
      << ", \"base_vertex\": " << event.base_vertex
      << ", \"base_index\": " << event.base_index
      << ", \"used_textures_mask\": " << event.used_textures_mask
      << ", \"used_texture_slots\": [";
  WriteUsedTextureSlots(out, event.used_textures_mask);
  out << "], \"texture_bindings\": [";
  bool needs_texture_comma = false;
  for (const auto& texture : event.textures)
  {
    if (!texture.active)
      continue;

    if (needs_texture_comma)
      out << ", ";
    WriteTextureBinding(texture, out);
    needs_texture_comma = true;
  }
  out << "]"
      << ", \"gen_mode\": {\"raw\": " << event.gen_mode_raw
      << ", \"texgen_count\": " << event.texgen_count
      << ", \"color_channel_count\": " << event.color_channel_count
      << ", \"tev_stage_count\": " << event.tev_stage_count
      << ", \"indirect_stage_count\": " << event.indirect_stage_count
      << ", \"cull_mode\": " << event.cull_mode << "}"
      << ", \"gx_z_mode\": {\"raw\": " << event.z_mode_raw
      << ", \"compare_enable\": " << BoolJson(event.z_compare_enable)
      << ", \"function\": " << event.z_function
      << ", \"update_enable\": " << BoolJson(event.z_update_enable) << "}"
      << ", \"gx_blend\": {\"raw\": " << event.blend_mode_raw
      << ", \"enabled\": " << BoolJson(event.blend_enable)
      << ", \"logic_op_enable\": " << BoolJson(event.logic_op_enable)
      << ", \"color_update\": " << BoolJson(event.color_update)
      << ", \"alpha_update\": " << BoolJson(event.alpha_update)
      << ", \"src_factor\": " << event.src_blend_factor
      << ", \"dst_factor\": " << event.dst_blend_factor << "}"
      << ", \"gx_alpha_compare\": {\"raw\": " << event.alpha_compare_raw
      << ", \"comp0\": " << event.alpha_comp0
      << ", \"ref0\": " << event.alpha_ref0
      << ", \"op\": " << event.alpha_op
      << ", \"comp1\": " << event.alpha_comp1
      << ", \"ref1\": " << event.alpha_ref1 << "}"
      << ", \"gx_fog\": {\"a_raw\": " << event.fog_raw[0]
      << ", \"b_magnitude\": " << event.fog_raw[1]
      << ", \"b_shift\": " << event.fog_raw[2]
      << ", \"c_proj_fsel_raw\": " << event.fog_raw[3]
      << ", \"color_raw\": " << event.fog_raw[4] << "}"
      << ", \"tev_orders\": [";
  for (std::size_t stage = 0; stage < tev_stage_count; ++stage)
  {
    const auto& order = event.tev_orders[stage];
    if (stage != 0)
      out << ", ";
    out << "{\"stage\": " << order.stage
        << ", \"texture_enabled\": " << BoolJson(order.texture_enabled)
        << ", \"tex_coord\": " << order.tex_coord
        << ", \"tex_map\": " << order.tex_map
        << ", \"color_channel\": " << order.color_channel << "}";
  }
  out << "], \"tev_stages\": [";
  for (std::size_t stage = 0; stage < tev_stage_count; ++stage)
  {
    const auto& tev_stage = event.tev_stages[stage];
    if (stage != 0)
      out << ", ";
    out << "{\"stage\": " << tev_stage.stage
        << ", \"color_raw\": " << tev_stage.color_raw
        << ", \"alpha_raw\": " << tev_stage.alpha_raw << "}";
  }
  out << "], \"color_channels\": [";
  for (std::size_t channel = 0; channel < color_channel_count; ++channel)
  {
    const auto& color_channel = event.color_channels[channel];
    if (channel != 0)
      out << ", ";
    out << "{\"channel\": " << color_channel.channel
        << ", \"color_raw\": " << color_channel.color_raw
        << ", \"alpha_raw\": " << color_channel.alpha_raw
        << ", \"color_light_mask\": " << color_channel.color_light_mask
        << ", \"alpha_light_mask\": " << color_channel.alpha_light_mask << "}";
  }
  out << "], \"requested_light_mask\": " << event.requested_light_mask
      << ", \"loaded_light_mask\": " << event.loaded_light_mask
      << ", \"lights\": [";
  bool needs_light_comma = false;
  for (const auto& light : event.lights)
  {
    if (!light.active)
      continue;

    if (needs_light_comma)
      out << ", ";
    out << "{\"index\": " << light.index << ", \"color\": ";
    WriteColor4(out, light.color_rgba);
    out << ", \"color_abgr\": ";
    WriteColor4(out, light.color_abgr);
    out << ", \"cosine_attenuation\": ";
    WriteFloat3(out, light.cosine_attenuation);
    out << ", \"distance_attenuation\": ";
    WriteFloat3(out, light.distance_attenuation);
    out << ", \"position\": ";
    WriteFloat3(out, light.position);
    out << ", \"direction\": ";
    WriteFloat3(out, light.direction);
    out << "}";
    needs_light_comma = true;
  }
  out << "]}";
}

void WriteSMGPCDolphinDrawTraceRecords(std::ostream& out, int requested_frame)
{
  for (std::size_t i = 0; i < s_smgpc_dolphin_draw_trace_events.size(); ++i)
  {
    const auto& event = s_smgpc_dolphin_draw_trace_events[i];
    WriteSMGPCTraceRecordPrefix(out, "render_packet", requested_frame);
    out << ",\"record_index\":" << i << ",\"payload\":";
    WriteSMGPCDolphinDrawTraceEventPayload(out, event, i);
    out << "}\n";
  }
}

void WriteSMGPCDolphinTopLevelRecord(std::ostream& out, int requested_frame, std::string_view key,
                                     std::string_view payload_json)
{
  WriteSMGPCTraceRecordPrefix(out, "top_level", requested_frame);
  out << ",\"key\":";
  WriteJsonString(out, key);
  out << ",\"payload\":" << payload_json << "}\n";
}

void WriteSMGPCDolphinSemanticEventRecord(std::ostream& out, int requested_frame,
                                          std::size_t record_index, std::string_view category,
                                          std::string_view name, std::string_view detail,
                                          std::string_view source)
{
  WriteSMGPCTraceRecordPrefix(out, "semantic_event", requested_frame);
  out << ",\"record_index\":" << record_index << ",\"payload\":{\"index\":" << record_index
      << ",\"frame_index\":" << requested_frame << ",\"category\":";
  WriteJsonString(out, category);
  out << ",\"name\":";
  WriteJsonString(out, name);
  out << ",\"detail\":";
  WriteJsonString(out, detail);
  out << ",\"stage\":\"\",\"source\":";
  WriteJsonString(out, source);
  out << "}}\n";
}

void WriteSMGPCDolphinSemanticTraceRecords(std::ostream& out, int requested_frame)
{
  auto record_index = std::size_t{0};
  WriteSMGPCDolphinSemanticEventRecord(out, requested_frame, record_index++, "capture",
                                       "dolphin_trace_requested_frame",
                                       "requested_frame=" + std::to_string(requested_frame),
                                       "dolphin_trace");

  const auto anchor_name = GetSMGPCDolphinSemanticAnchorName();
  if (!anchor_name.has_value())
    return;

  WriteSMGPCDolphinSemanticEventRecord(out, requested_frame, record_index++,
                                       GetSMGPCDolphinSemanticAnchorCategory(), *anchor_name,
                                       GetSMGPCDolphinSemanticAnchorDetail(requested_frame),
                                       "parity_capture");
}

void WriteSMGPCDolphinCopyEventPayload(std::ostream& out,
                                       const SMGPCDolphinCopyTraceEvent& event,
                                       std::size_t event_index)
{
  out << "{\"index\": " << event_index
      << ", \"event_index\": " << event.event_index
      << ", \"presenter_frame_count\": " << event.presenter_frame_count
      << ", \"kind\": \"" << (event.copy_to_xfb ? "xfb" : "texture") << "\""
      << ", \"copy_to_xfb\": " << BoolJson(event.copy_to_xfb)
      << ", \"depth_copy\": " << BoolJson(event.depth_copy)
      << ", \"clear\": " << BoolJson(event.clear)
      << ", \"half_scale\": " << BoolJson(event.half_scale)
      << ", \"scale_invert\": " << BoolJson(event.scale_invert)
      << ", \"clamp_top\": " << BoolJson(event.clamp_top)
      << ", \"clamp_bottom\": " << BoolJson(event.clamp_bottom)
      << ", \"intensity_format\": " << BoolJson(event.intensity_format)
      << ", \"auto_conversion\": " << BoolJson(event.auto_conversion)
      << ", \"dest_addr\": " << event.dest_addr
      << ", \"dest_stride\": " << event.dest_stride
      << ", \"source_rect\": {\"left\": " << event.src_left
      << ", \"top\": " << event.src_top
      << ", \"right\": " << event.src_right
      << ", \"bottom\": " << event.src_bottom
      << ", \"width\": " << event.copy_width
      << ", \"height\": " << event.copy_height << "}"
      << ", \"output_size\": {\"width\": " << event.output_width
      << ", \"height\": " << event.output_height << "}"
      << ", \"target_pixel_format\": " << event.target_pixel_format
      << ", \"real_format\": " << event.real_format
      << ", \"frame_to_field\": " << event.frame_to_field
      << ", \"gamma_index\": " << event.gamma_index
      << ", \"gamma_value\": " << event.gamma_value
      << ", \"y_scale\": " << event.y_scale
      << ", \"dispcopyyscale\": " << event.dispcopyyscale
      << ", \"scissor\": {\"tl_raw\": " << event.scissor_tl_raw
      << ", \"br_raw\": " << event.scissor_br_raw
      << ", \"offset_raw\": " << event.scissor_offset_raw << "}"
      << ", \"viewport\": {\"left\": " << event.viewport_left
      << ", \"right\": " << event.viewport_right
      << ", \"top\": " << event.viewport_top
      << ", \"bottom\": " << event.viewport_bottom
      << ", \"near_depth\": " << event.viewport_near_depth
      << ", \"far_depth\": " << event.viewport_far_depth << "}"
      << ", \"backbuffer\": {\"width\": " << event.backbuffer_width
      << ", \"height\": " << event.backbuffer_height << "}"
      << ", \"target_rect\": {\"left\": " << event.target_left
      << ", \"top\": " << event.target_top
      << ", \"right\": " << event.target_right
      << ", \"bottom\": " << event.target_bottom << "}}";
}

void WriteSMGPCDolphinCopyTraceRecords(std::ostream& out, int requested_frame)
{
  for (std::size_t i = 0; i < s_smgpc_dolphin_copy_trace_events.size(); ++i)
  {
    const auto& event = s_smgpc_dolphin_copy_trace_events[i];
    WriteSMGPCTraceRecordPrefix(out, "copy_event", requested_frame);
    out << ",\"record_index\":" << i << ",\"payload\":";
    WriteSMGPCDolphinCopyEventPayload(out, event, i);
    out << "}\n";
  }
}

void WriteSMGPCDolphinCopyTrace(const std::string& path, int requested_frame, int window)
{
  if (!File::CreateFullPath(path))
    return;

  std::ofstream out(path);
  if (!out)
    return;

  const auto current_frame = g_presenter ? g_presenter->FrameCount() : 0;
  WriteSMGPCTraceRecordPrefix(out, "trace_meta", requested_frame);
  out << ",\"payload\":{\"source_schema\":\"smgpc-dolphin-parity-trace-v1\",\"emulator\":\"dolphin\",\"requested_frame\":"
      << requested_frame << ",\"trace_window\":" << window << "}}\n";
  WriteSMGPCTraceRecordPrefix(out, "frame", requested_frame);
  out << ",\"payload\":{\"index\":" << requested_frame
      << ",\"dolphin_presenter_frame_count\":" << current_frame
      << ",\"framebuffer\":{\"width\":" << (g_presenter ? g_presenter->GetBackbufferWidth() : 0)
      << ",\"height\":" << (g_presenter ? g_presenter->GetBackbufferHeight() : 0) << "}}}\n";
  WriteSMGPCDolphinTopLevelRecord(out, requested_frame, "camera_pose", "null");
  WriteSMGPCDolphinTopLevelRecord(out, requested_frame, "scene_snapshot", "[]");
  WriteSMGPCDolphinTopLevelRecord(out, requested_frame, "scene_trace", "[]");
  WriteSMGPCDolphinTopLevelRecord(out, requested_frame, "layout_runtime", "[]");
  WriteSMGPCDolphinSemanticTraceRecords(out, requested_frame);
  WriteSMGPCDolphinDrawTraceRecords(out, requested_frame);
  WriteSMGPCDolphinCopyTraceRecords(out, requested_frame);
}

void RecordSMGPCDolphinCopyTrace(bool copy_to_xfb, const MathUtil::Rectangle<s32>& src_rect,
                                 u32 output_height, u32 dest_addr, u32 dest_stride,
                                 bool is_depth_copy, float y_scale, const UPE_Copy& pe_copy)
{
  ++s_smgpc_dolphin_copy_trace_event_index;

  const auto target_frame = GetSMGPCDolphinTraceFrame();
  const auto trace_path = GetSMGPCDolphinTracePath();
  const auto trace_window = GetSMGPCDolphinTraceWindow();
  if (!target_frame.has_value() || !trace_path.has_value() || !trace_window.has_value() ||
      s_smgpc_dolphin_copy_trace_completed)
  {
    return;
  }

  const auto current_frame = g_presenter ? g_presenter->FrameCount() : 0;
  const auto min_frame = std::max(0, *target_frame - *trace_window);
  const auto max_frame = *target_frame + *trace_window;
  if (current_frame < min_frame)
    return;

  if (current_frame <= max_frame)
  {
    const auto viewport_left = xfmem.viewport.xOrig - xfmem.viewport.wd;
    const auto viewport_right = xfmem.viewport.xOrig + xfmem.viewport.wd;
    const auto viewport_top = xfmem.viewport.yOrig - xfmem.viewport.ht;
    const auto viewport_bottom = xfmem.viewport.yOrig + xfmem.viewport.ht;
    const auto near_depth = (xfmem.viewport.farZ - xfmem.viewport.zRange) / 16777216.0f;
    const auto far_depth = xfmem.viewport.farZ / 16777216.0f;
    const auto target_rect =
        g_presenter ? g_presenter->GetTargetRectangle() : MathUtil::Rectangle<int>();
    s_smgpc_dolphin_copy_trace_events.push_back(SMGPCDolphinCopyTraceEvent{
        .event_index = s_smgpc_dolphin_copy_trace_event_index,
        .presenter_frame_count = current_frame,
        .copy_to_xfb = copy_to_xfb,
        .depth_copy = is_depth_copy,
        .clear = pe_copy.clear,
        .half_scale = pe_copy.half_scale,
        .scale_invert = pe_copy.scale_invert,
        .clamp_top = pe_copy.clamp_top,
        .clamp_bottom = pe_copy.clamp_bottom,
        .intensity_format = pe_copy.intensity_fmt,
        .auto_conversion = pe_copy.auto_conv,
        .dest_addr = dest_addr,
        .dest_stride = dest_stride,
        .src_left = static_cast<u32>(src_rect.left),
        .src_top = static_cast<u32>(src_rect.top),
        .src_right = static_cast<u32>(src_rect.right),
        .src_bottom = static_cast<u32>(src_rect.bottom),
        .copy_width = static_cast<u32>(src_rect.GetWidth()),
        .copy_height = static_cast<u32>(src_rect.GetHeight()),
        .output_width = static_cast<u32>(src_rect.GetWidth()),
        .output_height = output_height,
        .target_pixel_format = pe_copy.target_pixel_format,
        .real_format = static_cast<u32>(pe_copy.tp_realFormat()),
        .frame_to_field = static_cast<u32>(pe_copy.frame_to_field.Value()),
        .gamma_index = static_cast<u32>(pe_copy.gamma.Value()),
        .gamma_value = s_gammaLUT[pe_copy.gamma],
        .y_scale = y_scale,
        .dispcopyyscale = bpmem.dispcopyyscale,
        .scissor_tl_raw = bpmem.scissorTL.hex,
        .scissor_br_raw = bpmem.scissorBR.hex,
        .scissor_offset_raw = bpmem.scissorOffset.hex,
        .viewport_left = viewport_left,
        .viewport_right = viewport_right,
        .viewport_top = viewport_top,
        .viewport_bottom = viewport_bottom,
        .viewport_near_depth = near_depth,
        .viewport_far_depth = far_depth,
        .backbuffer_width = g_presenter ? g_presenter->GetBackbufferWidth() : 0,
        .backbuffer_height = g_presenter ? g_presenter->GetBackbufferHeight() : 0,
        .target_left = target_rect.left,
        .target_top = target_rect.top,
        .target_right = target_rect.right,
        .target_bottom = target_rect.bottom,
    });
  }

  if (!s_smgpc_dolphin_copy_trace_events.empty() && current_frame >= *target_frame)
    WriteSMGPCDolphinCopyTrace(*trace_path, *target_frame, *trace_window);

  if (current_frame > max_frame)
    s_smgpc_dolphin_copy_trace_completed = true;
}
}  // namespace

void RecordSMGPCDolphinDrawTrace(u32 draw_index, PrimitiveType primitive_type, u32 num_vertices,
                                 u32 num_indices, u32 base_vertex, u32 base_index,
                                 u32 used_textures_mask)
{
  const auto target_frame = GetSMGPCDolphinTraceFrame();
  const auto trace_path = GetSMGPCDolphinTracePath();
  const auto trace_window = GetSMGPCDolphinTraceWindow();
  if (!target_frame.has_value() || !trace_path.has_value() || !trace_window.has_value() ||
      s_smgpc_dolphin_copy_trace_completed)
  {
    return;
  }

  const auto current_frame = g_presenter ? g_presenter->FrameCount() : 0;
  const auto min_frame = std::max(0, *target_frame - *trace_window);
  const auto max_frame = *target_frame + *trace_window;
  if (current_frame < min_frame || current_frame > max_frame)
    return;

  SMGPCDolphinDrawTraceEvent event = {};
  event.draw_index = draw_index;
  event.presenter_frame_count = current_frame;
  event.primitive_type = primitive_type;
  event.num_vertices = num_vertices;
  event.num_indices = num_indices;
  event.base_vertex = base_vertex;
  event.base_index = base_index;
  event.used_textures_mask = used_textures_mask;
  event.texgen_count = bpmem.genMode.numtexgens.Value();
  event.color_channel_count = bpmem.genMode.numcolchans.Value();
  event.tev_stage_count = bpmem.genMode.numtevstages.Value() + 1;
  event.indirect_stage_count = bpmem.genMode.numindstages.Value();
  event.gen_mode_raw = bpmem.genMode.hex;
  event.cull_mode = static_cast<u32>(bpmem.genMode.cull_mode.Value());
  event.z_mode_raw = bpmem.zmode.hex;
  event.z_compare_enable = bpmem.zmode.test_enable.Value();
  event.z_function = static_cast<u32>(bpmem.zmode.func.Value());
  event.z_update_enable = bpmem.zmode.update_enable.Value();
  event.blend_mode_raw = bpmem.blendmode.hex;
  event.blend_enable = bpmem.blendmode.blend_enable.Value();
  event.logic_op_enable = bpmem.blendmode.logic_op_enable.Value();
  event.color_update = bpmem.blendmode.color_update.Value();
  event.alpha_update = bpmem.blendmode.alpha_update.Value();
  event.src_blend_factor = static_cast<u32>(bpmem.blendmode.src_factor.Value());
  event.dst_blend_factor = static_cast<u32>(bpmem.blendmode.dst_factor.Value());
  event.alpha_compare_raw = bpmem.alpha_test.hex;
  event.alpha_comp0 = static_cast<u32>(bpmem.alpha_test.comp0.Value());
  event.alpha_ref0 = bpmem.alpha_test.ref0.Value();
  event.alpha_op = static_cast<u32>(bpmem.alpha_test.logic.Value());
  event.alpha_comp1 = static_cast<u32>(bpmem.alpha_test.comp1.Value());
  event.alpha_ref1 = bpmem.alpha_test.ref1.Value();
  event.fog_raw[0] = bpmem.fog.a.hex;
  event.fog_raw[1] = bpmem.fog.b_magnitude;
  event.fog_raw[2] = bpmem.fog.b_shift;
  event.fog_raw[3] = bpmem.fog.c_proj_fsel.hex;
  event.fog_raw[4] = bpmem.fog.color.hex;

  for (u32 slot = 0; slot < event.textures.size(); ++slot)
  {
    if ((used_textures_mask & (1u << slot)) == 0)
      continue;

    const auto& tex = bpmem.tex.GetUnit(slot);
    const auto texture_info = TextureInfo::FromStage(slot);
    std::string identity_name;
    if (texture_info.IsDataValid())
      identity_name = texture_info.CalculateTextureName().GetFullName();

    event.textures[slot] = SMGPCDolphinTextureTraceState{
        .active = true,
        .slot = slot,
        .address = texture_info.GetRawAddress(),
        .width = texture_info.GetRawWidth(),
        .height = texture_info.GetRawHeight(),
        .expanded_width = texture_info.GetExpandedWidth(),
        .expanded_height = texture_info.GetExpandedHeight(),
        .texture_size = texture_info.GetTextureSize(),
        .texture_format = static_cast<u32>(texture_info.GetTextureFormat()),
        .tlut_format = static_cast<u32>(texture_info.GetTlutFormat()),
        .wrap_s = static_cast<u32>(tex.texMode0.wrap_s.Value()),
        .wrap_t = static_cast<u32>(tex.texMode0.wrap_t.Value()),
        .min_filter = static_cast<u32>(tex.texMode0.min_filter.Value()),
        .mag_filter = static_cast<u32>(tex.texMode0.mag_filter.Value()),
        .mipmap_filter = static_cast<u32>(tex.texMode0.mipmap_filter.Value()),
        .lod_bias = tex.texMode0.lod_bias.Value(),
        .min_lod = tex.texMode1.min_lod.Value(),
        .max_lod = tex.texMode1.max_lod.Value(),
        .from_tmem = texture_info.IsFromTmem(),
        .data_valid = texture_info.IsDataValid(),
        .has_mipmaps = texture_info.HasMipMaps(),
        .tex_mode0_raw = tex.texMode0.hex,
        .tex_mode1_raw = tex.texMode1.hex,
        .tex_image0_raw = tex.texImage0.hex,
        .tex_image1_raw = tex.texImage1.hex,
        .tex_image2_raw = tex.texImage2.hex,
        .tex_image3_raw = tex.texImage3.hex,
        .tex_tlut_raw = tex.texTlut.hex,
        .texture_format_name = TextureFormatName(texture_info.GetTextureFormat()),
        .tlut_format_name = TlutFormatName(texture_info.GetTlutFormat()),
        .identity_name = std::move(identity_name),
    };
  }

  const auto tev_stage_count = std::min<u32>(event.tev_stage_count, 16);
  for (u32 stage = 0; stage < tev_stage_count; ++stage)
  {
    const auto& order = bpmem.tevorders[stage / 2];
    const auto order_slot = static_cast<int>(stage & 1);
    event.tev_orders[stage] = SMGPCDolphinTevOrderTraceState{
        .stage = stage,
        .texture_enabled = order.getEnable(order_slot) != 0,
        .tex_coord = order.getTexCoord(order_slot),
        .tex_map = order.getTexMap(order_slot),
        .color_channel = static_cast<u32>(order.getColorChan(order_slot)),
    };
    event.tev_stages[stage] = SMGPCDolphinTevStageTraceState{
        .stage = stage,
        .color_raw = bpmem.combiners[stage].colorC.hex,
        .alpha_raw = bpmem.combiners[stage].alphaC.hex,
    };
  }

  const auto color_channel_count = std::min<u32>(event.color_channel_count, 2);
  for (u32 channel = 0; channel < color_channel_count; ++channel)
  {
    const auto color_light_mask = xfmem.color[channel].GetFullLightMask();
    const auto alpha_light_mask = xfmem.alpha[channel].GetFullLightMask();
    event.color_channels[channel] = SMGPCDolphinColorChannelTraceState{
        .channel = channel,
        .color_raw = xfmem.color[channel].hex,
        .alpha_raw = xfmem.alpha[channel].hex,
        .color_light_mask = color_light_mask,
        .alpha_light_mask = alpha_light_mask,
    };
    event.requested_light_mask |= color_light_mask | alpha_light_mask;
  }

  event.loaded_light_mask = event.requested_light_mask;
  for (std::size_t light_index = 0; light_index < event.lights.size(); ++light_index)
  {
    if ((event.requested_light_mask & (1u << light_index)) == 0)
      continue;

    const auto& light = xfmem.lights[light_index];
    event.lights[light_index] = SMGPCDolphinLightTraceState{
        .active = true,
        .index = static_cast<u32>(light_index),
        .color_rgba = {light.color[3], light.color[2], light.color[1], light.color[0]},
        .color_abgr = {light.color[0], light.color[1], light.color[2], light.color[3]},
        .cosine_attenuation = {light.cosatt[0], light.cosatt[1], light.cosatt[2]},
        .distance_attenuation = {light.distatt[0], light.distatt[1], light.distatt[2]},
        .position = {light.dpos[0], light.dpos[1], light.dpos[2]},
        .direction = {light.ddir[0], light.ddir[1], light.ddir[2]},
    };
  }

  s_smgpc_dolphin_draw_trace_events.push_back(event);
}

void BPInit()
{
  memset(reinterpret_cast<u8*>(&bpmem), 0, sizeof(bpmem));
  bpmem.bpMask = 0xFFFFFF;
}

static void BPWritten(PixelShaderManager& pixel_shader_manager, XFStateManager& xf_state_manager,
                      GeometryShaderManager& geometry_shader_manager, const BPCmd& bp,
                      int cycles_into_future)
{
  /*
  ----------------------------------------------------------------------------------------------------------------
  Purpose: Writes to the BP registers
  Called: At the end of every: OpcodeDecoding.cpp ExecuteDisplayList > Decode() > LoadBPReg
  How It Works: First the pipeline is flushed then update the bpmem with the new value.
          Some of the BP cases have to call certain functions while others just update the bpmem.
          some bp cases check the changes variable, because they might not have to be updated all
  the time
  NOTE: it seems not all bp cases like checking changes, so calling if (bp.changes == 0 ? false :
  true)
      had to be ditched and the games seem to work fine with out it.
  NOTE2: Yet Another GameCube Documentation calls them Bypass Raster State Registers but possibly
  completely wrong
  NOTE3: This controls the register groups: RAS1/2, SU, TF, TEV, C/Z, PEC
  TODO: Turn into function table. The (future) DisplayList (DL) jit can then call the functions
  directly,
      getting rid of dynamic dispatch. Unfortunately, few games use DLs properly - most\
      just stuff geometry in them and don't put state changes there
  ----------------------------------------------------------------------------------------------------------------
  */

  if (((s32*)&bpmem)[bp.address] == bp.newvalue)
  {
    if (!(bp.address == BPMEM_TRIGGER_EFB_COPY || bp.address == BPMEM_CLEARBBOX1 ||
          bp.address == BPMEM_CLEARBBOX2 || bp.address == BPMEM_SETDRAWDONE ||
          bp.address == BPMEM_PE_TOKEN_ID || bp.address == BPMEM_PE_TOKEN_INT_ID ||
          bp.address == BPMEM_LOADTLUT0 || bp.address == BPMEM_LOADTLUT1 ||
          bp.address == BPMEM_TEXINVALIDATE || bp.address == BPMEM_PRELOAD_MODE ||
          bp.address == BPMEM_CLEAR_PIXEL_PERF))
    {
      return;
    }
  }

  FlushPipeline();

  ((u32*)&bpmem)[bp.address] = bp.newvalue;

  switch (bp.address)
  {
  case BPMEM_GENMODE:  // Set the Generation Mode
    PRIM_LOG(
        "genmode: texgen={}, col={}, multisampling={}, tev={}, cull_mode={}, ind={}, zfeeze={}",
        bpmem.genMode.numtexgens, bpmem.genMode.numcolchans, bpmem.genMode.multisampling,
        bpmem.genMode.numtevstages + 1, bpmem.genMode.cull_mode, bpmem.genMode.numindstages,
        bpmem.genMode.zfreeze);

    if (bp.changes)
      pixel_shader_manager.SetGenModeChanged();

    // Only call SetGenerationMode when cull mode changes.
    if (bp.changes & 0xC000)
      SetGenerationMode();
    return;
  case BPMEM_IND_MTXA:  // Index Matrix Changed
  case BPMEM_IND_MTXB:
  case BPMEM_IND_MTXC:
  case BPMEM_IND_MTXA + 3:
  case BPMEM_IND_MTXB + 3:
  case BPMEM_IND_MTXC + 3:
  case BPMEM_IND_MTXA + 6:
  case BPMEM_IND_MTXB + 6:
  case BPMEM_IND_MTXC + 6:
    if (bp.changes)
      pixel_shader_manager.SetIndMatrixChanged((bp.address - BPMEM_IND_MTXA) / 3);
    return;
  case BPMEM_RAS1_SS0:  // Index Texture Coordinate Scale 0
    if (bp.changes)
      pixel_shader_manager.SetIndTexScaleChanged(false);
    return;
  case BPMEM_RAS1_SS1:  // Index Texture Coordinate Scale 1
    if (bp.changes)
      pixel_shader_manager.SetIndTexScaleChanged(true);
    return;
  // ----------------
  // Scissor Control
  // ----------------
  case BPMEM_SCISSORTL:      // Scissor Rectable Top, Left
  case BPMEM_SCISSORBR:      // Scissor Rectable Bottom, Right
  case BPMEM_SCISSOROFFSET:  // Scissor Offset
    xf_state_manager.SetViewportChanged();
    geometry_shader_manager.SetViewportChanged();
    return;
  case BPMEM_LINEPTWIDTH:  // Line Width
    geometry_shader_manager.SetLinePtWidthChanged();
    return;
  case BPMEM_ZMODE:  // Depth Control
    PRIM_LOG("zmode: test={}, func={}, upd={}", bpmem.zmode.test_enable, bpmem.zmode.func,
             bpmem.zmode.update_enable);
    SetDepthMode();
    pixel_shader_manager.SetZModeControl();
    return;
  case BPMEM_BLENDMODE:  // Blending Control
    if (bp.changes & 0xFFFF)
    {
      PRIM_LOG("blendmode: en={}, open={}, colupd={}, alphaupd={}, dst={}, src={}, sub={}, mode={}",
               bpmem.blendmode.blend_enable, bpmem.blendmode.logic_op_enable,
               bpmem.blendmode.color_update, bpmem.blendmode.alpha_update,
               bpmem.blendmode.dst_factor, bpmem.blendmode.src_factor, bpmem.blendmode.subtract,
               bpmem.blendmode.logic_mode);

      SetBlendMode();

      pixel_shader_manager.SetBlendModeChanged();
    }
    return;
  case BPMEM_CONSTANTALPHA:  // Set Destination Alpha
    PRIM_LOG("constalpha: alp={}, en={}", bpmem.dstalpha.alpha, bpmem.dstalpha.enable);
    if (bp.changes)
    {
      pixel_shader_manager.SetAlpha();
      pixel_shader_manager.SetDestAlphaChanged();
    }
    if (bp.changes & 0x100)
      SetBlendMode();
    return;

  // This is called when the game is done drawing the new frame (eg: like in DX: Begin(); Draw();
  // End();)
  // Triggers an interrupt on the PPC side so that the game knows when the GPU has finished drawing.
  // Tokens are similar.
  case BPMEM_SETDRAWDONE:
    switch (bp.newvalue & 0xFF)
    {
    case 0x02:
    {
      INCSTAT(g_stats.this_frame.num_draw_done);
      g_texture_cache->FlushEFBCopies();
      g_texture_cache->FlushStaleBinds();
      g_framebuffer_manager->InvalidatePeekCache(false);
      g_framebuffer_manager->RefreshPeekCache();
      auto& system = Core::System::GetInstance();
      if (!system.GetFifo().UseDeterministicGPUThread())
        system.GetPixelEngine().SetFinish(cycles_into_future);  // may generate interrupt
      DEBUG_LOG_FMT(VIDEO, "GXSetDrawDone SetPEFinish (value: {:#04X})", bp.newvalue & 0xFFFF);
      return;
    }

    default:
      WARN_LOG_FMT(VIDEO, "GXSetDrawDone ??? (value {:#04X})", bp.newvalue & 0xFFFF);
      return;
    }
    return;
  case BPMEM_PE_TOKEN_ID:  // Pixel Engine Token ID
  {
    INCSTAT(g_stats.this_frame.num_token);
    g_texture_cache->FlushEFBCopies();
    g_texture_cache->FlushStaleBinds();
    g_framebuffer_manager->InvalidatePeekCache(false);
    g_framebuffer_manager->RefreshPeekCache();
    auto& system = Core::System::GetInstance();
    if (!system.GetFifo().UseDeterministicGPUThread())
    {
      system.GetPixelEngine().SetToken(static_cast<u16>(bp.newvalue & 0xFFFF), false,
                                       cycles_into_future);
    }
    DEBUG_LOG_FMT(VIDEO, "SetPEToken {:#06X}", bp.newvalue & 0xFFFF);
    return;
  }
  case BPMEM_PE_TOKEN_INT_ID:  // Pixel Engine Interrupt Token ID
  {
    INCSTAT(g_stats.this_frame.num_token_int);
    g_texture_cache->FlushEFBCopies();
    g_texture_cache->FlushStaleBinds();
    g_framebuffer_manager->InvalidatePeekCache(false);
    g_framebuffer_manager->RefreshPeekCache();
    auto& system = Core::System::GetInstance();
    if (!system.GetFifo().UseDeterministicGPUThread())
    {
      system.GetPixelEngine().SetToken(static_cast<u16>(bp.newvalue & 0xFFFF), true,
                                       cycles_into_future);
    }
    DEBUG_LOG_FMT(VIDEO, "SetPEToken + INT {:#06X}", bp.newvalue & 0xFFFF);
    return;
  }

  // ------------------------
  // EFB copy command. This copies a rectangle from the EFB to either RAM in a texture format or to
  // XFB as YUYV.
  // It can also optionally clear the EFB while copying from it. To emulate this, we of course copy
  // first and clear afterwards.
  case BPMEM_TRIGGER_EFB_COPY:  // Copy EFB Region or Render to the XFB or Clear the screen.
  {
    // The bottom right is within the rectangle
    // The values in bpmem.copyTexSrcXY and bpmem.copyTexSrcWH are updated in case 0x49 and 0x4a in
    // this function

    u32 destAddr = bpmem.copyTexDest << 5;
    u32 destStride = bpmem.copyDestStride << 5;

    MathUtil::Rectangle<s32> srcRect;
    srcRect.left = bpmem.copyTexSrcXY.x;
    srcRect.top = bpmem.copyTexSrcXY.y;

    // Here Width+1 like Height, otherwise some textures are corrupted already since the native
    // resolution.
    srcRect.right = bpmem.copyTexSrcXY.x + bpmem.copyTexSrcWH.x + 1;
    srcRect.bottom = bpmem.copyTexSrcXY.y + bpmem.copyTexSrcWH.y + 1;

    const UPE_Copy PE_copy = bpmem.triggerEFBCopy;

    // Since the copy X and Y coordinates/sizes are 10-bit, the game can configure a copy region up
    // to 1024x1024. Hardware tests have found that the number of bytes written does not depend on
    // the configured stride, instead it is based on the size registers, writing beyond the length
    // of a single row. The data written for the pixels which lie outside the EFB bounds does not
    // wrap around instead returning different colors based on the pixel format of the EFB. This
    // suggests it's not based on coordinates, but instead on memory addresses. The effect of a
    // within-bounds size but out-of-bounds offset (e.g. offset 320,0, size 640,480) are the same.

    // As it would be difficult to emulate the exact behavior of out-of-bounds reads, instead of
    // writing the junk data, we don't write anything to RAM at all for over-sized copies, and clamp
    // to the EFB borders for over-offset copies. The arcade virtual console games (e.g. 1942) are
    // known for configuring these out-of-range copies.

    if (u32(srcRect.right) > EFB_WIDTH || u32(srcRect.bottom) > EFB_HEIGHT)
    {
      WARN_LOG_FMT(VIDEO, "Oversized EFB copy: {}x{} (offset {},{} stride {})", srcRect.GetWidth(),
                   srcRect.GetHeight(), srcRect.left, srcRect.top, destStride);

      if (u32(srcRect.left) >= EFB_WIDTH || u32(srcRect.top) >= EFB_HEIGHT)
      {
        // This is not a sane src rectangle, it doesn't touch any valid image data at all
        // Just ignore it
        // Apparently Mario Kart Wii in wifi mode can generate a deformed EFB copy of size 4x4
        // at offset (328,1020)
        if (PE_copy.copy_to_xfb == 1)
        {
          // Make sure we disable Bounding box to match the side effects of the non-failure path
          g_bounding_box->Disable(pixel_shader_manager);
        }

        return;
      }

      // Clamp the copy region to fit within EFB. So that we don't end up with a stretched image.
      srcRect.right = std::clamp<int>(srcRect.right, 0, EFB_WIDTH);
      srcRect.bottom = std::clamp<int>(srcRect.bottom, 0, EFB_HEIGHT);
    }

    const u32 copy_width = srcRect.GetWidth();
    const u32 copy_height = srcRect.GetHeight();

    if (destStride != 0)
    {
      // Check if we are to copy from the EFB or draw to the XFB
      if (PE_copy.copy_to_xfb == 0)
      {
        // bpmem.zcontrol.pixel_format to PixelFormat::Z24 is when the game wants to copy from
        // ZBuffer (Zbuffer uses 24-bit Format)
        bool is_depth_copy = bpmem.zcontrol.pixel_format == PixelFormat::Z24;
        RecordSMGPCDolphinCopyTrace(false, srcRect, copy_height, destAddr, destStride,
                                    is_depth_copy, 1.0f, PE_copy);
        g_texture_cache->CopyRenderTargetToTexture(
            destAddr, PE_copy.tp_realFormat(), copy_width, copy_height, destStride, is_depth_copy,
            srcRect, PE_copy.intensity_fmt && PE_copy.auto_conv, PE_copy.half_scale, 1.0f,
            s_gammaLUT[PE_copy.gamma], bpmem.triggerEFBCopy.clamp_top,
            bpmem.triggerEFBCopy.clamp_bottom, bpmem.copyfilter.GetCoefficients());
      }
      else
      {
        // We should be able to get away with deactivating the current bbox tracking
        // here. Not sure if there's a better spot to put this.
        // the number of lines copied is determined by the y scale * source efb height
        g_bounding_box->Disable(pixel_shader_manager);

        float yScale;
        if (PE_copy.scale_invert)
          yScale = 256.0f / static_cast<float>(bpmem.dispcopyyscale);
        else
          yScale = static_cast<float>(bpmem.dispcopyyscale) / 256.0f;

        float num_xfb_lines = 1.0f + bpmem.copyTexSrcWH.y * yScale;

        u32 height = static_cast<u32>(num_xfb_lines);

        DEBUG_LOG_FMT(VIDEO,
                      "RenderToXFB: destAddr: {:08x} | srcRect [{} {} {} {}] | fbWidth: {} | "
                      "fbStride: {} | fbHeight: {} | yScale: {}",
                      destAddr, srcRect.left, srcRect.top, srcRect.right, srcRect.bottom,
                      bpmem.copyTexSrcWH.x + 1, destStride, height, yScale);

        bool is_depth_copy = bpmem.zcontrol.pixel_format == PixelFormat::Z24;
        RecordSMGPCDolphinCopyTrace(true, srcRect, height, destAddr, destStride, is_depth_copy,
                                    yScale, PE_copy);
        g_texture_cache->CopyRenderTargetToTexture(
            destAddr, EFBCopyFormat::XFB, copy_width, height, destStride, is_depth_copy, srcRect,
            false, false, yScale, s_gammaLUT[PE_copy.gamma], bpmem.triggerEFBCopy.clamp_top,
            bpmem.triggerEFBCopy.clamp_bottom, bpmem.copyfilter.GetCoefficients());

        auto& system = Core::System::GetInstance();

        // This is as closest as we have to an "end of the frame"
        // It works 99% of the time.
        // But sometimes games want to render an XFB larger than the EFB's 640x528 pixel resolution
        // (especially when using the 3xMSAA mode, which cuts EFB resolution to 640x264). So they
        // render multiple sub-frames and arrange the XFB copies in next to each-other in main
        // memory so they form a single completed XFB. See
        // https://dolphin-emu.org/blog/2017/11/19/hybridxfb/ for examples and more detail.
        system.GetVideoEvents().after_frame_event.Trigger(system);

        // Note: Theoretically, in the future we could track the VI configuration and try to detect
        //       when an XFB is the last XFB copy of a frame. Not only would we get a clean "end of
        //       the frame", but we would also be able to use ImmediateXFB even for these games.
        //       Might also clean up some issues with games doing XFB copies they don't intend to
        //       display.

        if (g_ActiveConfig.bImmediateXFB)
        {
          // below div two to convert from bytes to pixels - it expects width, not stride
          g_presenter->ImmediateSwap(destAddr, destStride / 2, destStride, height);
        }
        else
        {
          if (system.GetFifoPlayer().IsRunningWithFakeVideoInterfaceUpdates())
          {
            auto& vi = system.GetVideoInterface();
            vi.FakeVIUpdate(destAddr, srcRect.GetWidth(), destStride, height);
          }
        }
      }
    }

    // Clear the rectangular region after copying it.
    if (PE_copy.clear)
    {
      const bool color_enable = bpmem.blendmode.color_update != 0;
      const bool alpha_enable = bpmem.blendmode.alpha_update != 0;
      const bool z_enable = bpmem.zmode.update_enable != 0;
      const auto pixel_format = bpmem.zcontrol.pixel_format;
      const auto color_ar = bpmem.clearcolorAR;
      const auto color_gb = bpmem.clearcolorGB;
      const auto z_value = bpmem.clearZValue;
      ClearScreen(g_framebuffer_manager.get(), srcRect, color_enable, alpha_enable, z_enable,
                  pixel_format, color_ar, color_gb, z_value);

      // Scissor rect must be restored.
      BPFunctions::SetScissorAndViewport(g_framebuffer_manager.get(), bpmem.scissorTL,
                                         bpmem.scissorBR, bpmem.scissorOffset, xfmem.viewport);
    }

    return;
  }
  case BPMEM_LOADTLUT0:  // This updates bpmem.tmem_config.tlut_src, no need to do anything here.
    return;
  case BPMEM_LOADTLUT1:  // Load a Texture Look Up Table
  {
    u32 tmem_addr = bpmem.tmem_config.tlut_dest.tmem_addr << 9;
    u32 tmem_transfer_count = bpmem.tmem_config.tlut_dest.tmem_line_count * TMEM_LINE_SIZE;
    u32 addr = bpmem.tmem_config.tlut_src << 5;

    // The GameCube ignores the upper bits of this address. Some games (WW, MKDD) set them.
    auto& system = Core::System::GetInstance();
    if (!system.IsWii())
      addr = addr & 0x01FFFFFF;

    // The copy below will always be in bounds as tmem is bigger than the maximum address a TLUT can
    // be loaded to.
    static constexpr u32 MAX_LOADABLE_TMEM_ADDR =
        (1 << bpmem.tmem_config.tlut_dest.tmem_addr.NumBits()) << 9;
    static constexpr u32 MAX_TMEM_LINE_COUNT =
        (1 << bpmem.tmem_config.tlut_dest.tmem_line_count.NumBits()) * TMEM_LINE_SIZE;
    static_assert(MAX_LOADABLE_TMEM_ADDR + MAX_TMEM_LINE_COUNT < TMEM_SIZE);

    auto& memory = system.GetMemory();
    memory.CopyFromEmu(s_tex_mem.data() + tmem_addr, addr, tmem_transfer_count);

    if (OpcodeDecoder::g_record_fifo_data)
      system.GetFifoRecorder().UseMemory(addr, tmem_transfer_count, MemoryUpdate::Type::TMEM);

    TMEM::InvalidateAll();

    return;
  }
  case BPMEM_FOGRANGE:  // Fog Settings Control
  case BPMEM_FOGRANGE + 1:
  case BPMEM_FOGRANGE + 2:
  case BPMEM_FOGRANGE + 3:
  case BPMEM_FOGRANGE + 4:
  case BPMEM_FOGRANGE + 5:
    if (bp.changes)
      pixel_shader_manager.SetFogRangeAdjustChanged();
    return;
  case BPMEM_FOGPARAM0:
  case BPMEM_FOGBMAGNITUDE:
  case BPMEM_FOGBEXPONENT:
  case BPMEM_FOGPARAM3:
    if (bp.changes)
      pixel_shader_manager.SetFogParamChanged();
    return;
  case BPMEM_FOGCOLOR:  // Fog Color
    if (bp.changes)
      pixel_shader_manager.SetFogColorChanged();
    return;
  case BPMEM_ALPHACOMPARE:  // Compare Alpha Values
    PRIM_LOG("alphacmp: ref0={}, ref1={}, comp0={}, comp1={}, logic={}", bpmem.alpha_test.ref0,
             bpmem.alpha_test.ref1, bpmem.alpha_test.comp0, bpmem.alpha_test.comp1,
             bpmem.alpha_test.logic);
    if (bp.changes & 0xFFFF)
      pixel_shader_manager.SetAlpha();
    if (bp.changes)
    {
      pixel_shader_manager.SetAlphaTestChanged();
      SetBlendMode();
    }
    return;
  case BPMEM_BIAS:  // BIAS
    PRIM_LOG("ztex bias={:#x}", bpmem.ztex1.bias);
    if (bp.changes)
      pixel_shader_manager.SetZTextureBias();
    return;
  case BPMEM_ZTEX2:  // Z Texture type
  {
    if (bp.changes & 3)
      pixel_shader_manager.SetZTextureTypeChanged();
    if (bp.changes & 12)
      pixel_shader_manager.SetZTextureOpChanged();
    PRIM_LOG("ztex op={}, type={}", bpmem.ztex2.op, bpmem.ztex2.type);
  }
    return;
  // ----------------------------------
  // Display Copy Filtering Control - GX_SetCopyFilter(u8 aa,u8 sample_pattern[12][2],u8 vf,u8
  // vfilter[7])
  // Fields: Destination, Frame2Field, Gamma, Source
  // ----------------------------------
  case BPMEM_DISPLAYCOPYFILTER:      // if (aa) { use sample_pattern } else { use 666666 }
  case BPMEM_DISPLAYCOPYFILTER + 1:  // if (aa) { use sample_pattern } else { use 666666 }
  case BPMEM_DISPLAYCOPYFILTER + 2:  // if (aa) { use sample_pattern } else { use 666666 }
  case BPMEM_DISPLAYCOPYFILTER + 3:  // if (aa) { use sample_pattern } else { use 666666 }
  case BPMEM_COPYFILTER0:            // if (vf) { use vfilter } else { use 595000 }
  case BPMEM_COPYFILTER1:            // if (vf) { use vfilter } else { use 000015 }
    return;
  // -----------------------------------
  // Interlacing Control
  // -----------------------------------
  case BPMEM_FIELDMASK:  // GX_SetFieldMask(u8 even_mask,u8 odd_mask)
  case BPMEM_FIELDMODE:  // GX_SetFieldMode(u8 field_mode,u8 half_aspect_ratio)
    // TODO
    return;
  // ----------------------------------------
  // Unimportant regs (Clock, Perf, ...)
  // ----------------------------------------
  case BPMEM_BUSCLOCK0:   // TB Bus Clock ?
  case BPMEM_BUSCLOCK1:   // TB Bus Clock ?
  case BPMEM_PERF0_TRI:   // Perf: Triangles
  case BPMEM_PERF0_QUAD:  // Perf: Quads
  case BPMEM_PERF1:       // Perf: Some Clock, Texels, TX, TC
    return;
  // ----------------
  // EFB Copy config
  // ----------------
  case BPMEM_EFB_TL:    // EFB Source Rect. Top, Left
  case BPMEM_EFB_WH:    // EFB Source Rect. Width, Height - 1
  case BPMEM_EFB_ADDR:  // EFB Target Address
    return;
  // --------------
  // Clear Config
  // --------------
  case BPMEM_CLEAR_AR:  // Alpha and Red Components
  case BPMEM_CLEAR_GB:  // Green and Blue Components
  case BPMEM_CLEAR_Z:   // Z Components (24-bit Zbuffer)
    return;
  // -------------------------
  // Bounding Box Control
  // -------------------------
  case BPMEM_CLEARBBOX1:
  case BPMEM_CLEARBBOX2:
  {
    const u8 offset = bp.address & 2;
    g_bounding_box->Enable(pixel_shader_manager);

    g_bounding_box->Set(offset, bp.newvalue & 0x3ff);
    g_bounding_box->Set(offset + 1, bp.newvalue >> 10);
  }
    return;
  case BPMEM_TEXINVALIDATE:
    TMEM::Invalidate(bp.newvalue);
    return;

  case BPMEM_ZCOMPARE:  // Set the Z-Compare and EFB pixel format
    OnPixelFormatChange(g_framebuffer_manager.get(), bpmem.zcontrol.pixel_format,
                        bpmem.zcontrol.zformat);
    if (bp.changes & 7)
      SetBlendMode();  // dual source could be activated by changing to PIXELFMT_RGBA6_Z24
    pixel_shader_manager.SetZModeControl();
    return;

  case BPMEM_EFB_STRIDE:  // Display Copy Stride
  case BPMEM_COPYYSCALE:  // Display Copy Y Scale
    return;

  /* 24 RID
   * 21 BC3 - Ind. Tex Stage 3 NTexCoord
   * 18 BI3 - Ind. Tex Stage 3 NTexMap
   * 15 BC2 - Ind. Tex Stage 2 NTexCoord
   * 12 BI2 - Ind. Tex Stage 2 NTexMap
   * 9 BC1 - Ind. Tex Stage 1 NTexCoord
   * 6 BI1 - Ind. Tex Stage 1 NTexMap
   * 3 BC0 - Ind. Tex Stage 0 NTexCoord
   * 0 BI0 - Ind. Tex Stage 0 NTexMap */
  case BPMEM_IREF:
  {
    if (bp.changes)
      pixel_shader_manager.SetTevIndirectChanged();
    return;
  }

  case BPMEM_TEV_KSEL:      // Texture Environment Swap Mode Table 0
  case BPMEM_TEV_KSEL + 1:  // Texture Environment Swap Mode Table 1
  case BPMEM_TEV_KSEL + 2:  // Texture Environment Swap Mode Table 2
  case BPMEM_TEV_KSEL + 3:  // Texture Environment Swap Mode Table 3
  case BPMEM_TEV_KSEL + 4:  // Texture Environment Swap Mode Table 4
  case BPMEM_TEV_KSEL + 5:  // Texture Environment Swap Mode Table 5
  case BPMEM_TEV_KSEL + 6:  // Texture Environment Swap Mode Table 6
  case BPMEM_TEV_KSEL + 7:  // Texture Environment Swap Mode Table 7
    pixel_shader_manager.SetTevKSel(bp.address - BPMEM_TEV_KSEL, bp.newvalue);
    return;

  /* This Register can be used to limit to which bits of BP registers is
   * actually written to. The mask is only valid for the next BP write,
   * and will reset itself afterwards. It's handled as a special case in
   * LoadBPReg. */
  case BPMEM_BP_MASK:

  case BPMEM_IND_IMASK:  // Index Mask ?
  case BPMEM_REVBITS:    // Always set to 0x0F when GX_InitRevBits() is called.
    return;

  case BPMEM_CLEAR_PIXEL_PERF:
    // GXClearPixMetric writes 0xAAA here, Sunshine alternates this register between values 0x000
    // and 0xAAA
    if (PerfQueryBase::ShouldEmulate())
      g_perf_query->ResetQuery();
    return;

  case BPMEM_PRELOAD_ADDR:
  case BPMEM_PRELOAD_TMEMEVEN:
  case BPMEM_PRELOAD_TMEMODD:  // Used when PRELOAD_MODE is set
    return;

  case BPMEM_PRELOAD_MODE:  // Set to 0 when GX_TexModeSync() is called.
    // if this is different from 0, manual TMEM management is used (GX_PreloadEntireTexture).
    if (bp.newvalue != 0)
    {
      // TODO: Not quite sure if this is completely correct (likely not)
      // NOTE: libogc's implementation of GX_PreloadEntireTexture seems flawed, so it's not
      // necessarily a good reference for RE'ing this feature.

      BPS_TmemConfig& tmem_cfg = bpmem.tmem_config;
      u32 src_addr = tmem_cfg.preload_addr << 5;  // TODO: Should we add mask here on GC?
      u32 bytes_read = 0;
      u32 tmem_addr_even = tmem_cfg.preload_tmem_even * TMEM_LINE_SIZE;

      if (tmem_cfg.preload_tile_info.type != 3)
      {
        if (tmem_addr_even < TMEM_SIZE)
        {
          bytes_read = tmem_cfg.preload_tile_info.count * TMEM_LINE_SIZE;
          if (tmem_addr_even + bytes_read > TMEM_SIZE)
            bytes_read = TMEM_SIZE - tmem_addr_even;

          auto& system = Core::System::GetInstance();
          auto& memory = system.GetMemory();
          memory.CopyFromEmu(s_tex_mem.data() + tmem_addr_even, src_addr, bytes_read);
        }
      }
      else  // RGBA8 tiles (and CI14, but that might just be stupid libogc!)
      {
        auto& system = Core::System::GetInstance();
        auto& memory = system.GetMemory();

        // AR and GB tiles are stored in separate TMEM banks => can't use a single memcpy for
        // everything
        u32 tmem_addr_odd = tmem_cfg.preload_tmem_odd * TMEM_LINE_SIZE;

        for (u32 i = 0; i < tmem_cfg.preload_tile_info.count; ++i)
        {
          if (tmem_addr_even + TMEM_LINE_SIZE > TMEM_SIZE ||
              tmem_addr_odd + TMEM_LINE_SIZE > TMEM_SIZE)
          {
            break;
          }

          memory.CopyFromEmu(s_tex_mem.data() + tmem_addr_even, src_addr + bytes_read,
                             TMEM_LINE_SIZE);
          memory.CopyFromEmu(s_tex_mem.data() + tmem_addr_odd,
                             src_addr + bytes_read + TMEM_LINE_SIZE, TMEM_LINE_SIZE);
          tmem_addr_even += TMEM_LINE_SIZE;
          tmem_addr_odd += TMEM_LINE_SIZE;
          bytes_read += TMEM_LINE_SIZE * 2;
        }
      }

      if (OpcodeDecoder::g_record_fifo_data)
      {
        Core::System::GetInstance().GetFifoRecorder().UseMemory(src_addr, bytes_read,
                                                                MemoryUpdate::Type::TMEM);
      }

      TMEM::InvalidateAll();
    }
    return;

  // ---------------------------------------------------
  // Set the TEV Color
  // ---------------------------------------------------
  //
  // NOTE: Each of these registers actually maps to two variables internally.
  //       There's a bit that specifies which one is currently written to.
  //
  // NOTE: Some games write only to the RA register (or only to the BG register).
  //       We may not assume that the unwritten register holds a valid value, hence
  //       both component pairs need to be loaded individually.
  case BPMEM_TEV_COLOR_RA:
  case BPMEM_TEV_COLOR_RA + 2:
  case BPMEM_TEV_COLOR_RA + 4:
  case BPMEM_TEV_COLOR_RA + 6:
  {
    int num = (bp.address >> 1) & 0x3;
    if (bpmem.tevregs[num].ra.type == TevRegType::Constant)
    {
      pixel_shader_manager.SetTevKonstColor(num, 0, bpmem.tevregs[num].ra.red);
      pixel_shader_manager.SetTevKonstColor(num, 3, bpmem.tevregs[num].ra.alpha);
    }
    else
    {
      pixel_shader_manager.SetTevColor(num, 0, bpmem.tevregs[num].ra.red);
      pixel_shader_manager.SetTevColor(num, 3, bpmem.tevregs[num].ra.alpha);
    }
    return;
  }

  case BPMEM_TEV_COLOR_BG:
  case BPMEM_TEV_COLOR_BG + 2:
  case BPMEM_TEV_COLOR_BG + 4:
  case BPMEM_TEV_COLOR_BG + 6:
  {
    int num = (bp.address >> 1) & 0x3;
    if (bpmem.tevregs[num].bg.type == TevRegType::Constant)
    {
      pixel_shader_manager.SetTevKonstColor(num, 1, bpmem.tevregs[num].bg.green);
      pixel_shader_manager.SetTevKonstColor(num, 2, bpmem.tevregs[num].bg.blue);
    }
    else
    {
      pixel_shader_manager.SetTevColor(num, 1, bpmem.tevregs[num].bg.green);
      pixel_shader_manager.SetTevColor(num, 2, bpmem.tevregs[num].bg.blue);
    }
    return;
  }

  default:
    break;
  }

  switch (bp.address & 0xFC)  // Texture sampler filter
  {
  // -------------------------
  // Texture Environment Order
  // -------------------------
  case BPMEM_TREF:
  case BPMEM_TREF + 4:
    pixel_shader_manager.SetTevOrder(bp.address - BPMEM_TREF, bp.newvalue);
    return;
  // ----------------------
  // Set wrap size
  // ----------------------
  case BPMEM_SU_SSIZE:  // Matches BPMEM_SU_TSIZE too
  case BPMEM_SU_SSIZE + 4:
  case BPMEM_SU_SSIZE + 8:
  case BPMEM_SU_SSIZE + 12:
    if (bp.changes)
    {
      pixel_shader_manager.SetTexCoordChanged((bp.address - BPMEM_SU_SSIZE) >> 1);
      geometry_shader_manager.SetTexCoordChanged((bp.address - BPMEM_SU_SSIZE) >> 1);
    }
    return;
  }

  if ((bp.address & 0xc0) == 0x80)
  {
    auto tex_address = TexUnitAddress::FromBPAddress(bp.address);

    switch (tex_address.Reg)
    {
    // ------------------------
    // BPMEM_TX_SETMODE0 - (Texture lookup and filtering mode) LOD/BIAS Clamp, MaxAnsio, LODBIAS,
    // DiagLoad, Min Filter, Mag Filter, Wrap T, S
    // BPMEM_TX_SETMODE1 - (LOD Stuff) - Max LOD, Min LOD
    // ------------------------
    case TexUnitAddress::Register::SETMODE0:
    case TexUnitAddress::Register::SETMODE1:
      TMEM::ConfigurationChanged(tex_address, bp.newvalue);
      return;

    // --------------------------------------------
    // BPMEM_TX_SETIMAGE0 - Texture width, height, format
    // BPMEM_TX_SETIMAGE1 - even LOD address in TMEM - Image Type, Cache Height, Cache Width,
    //                      TMEM Offset
    // BPMEM_TX_SETIMAGE2 - odd LOD address in TMEM - Cache Height, Cache Width, TMEM Offset
    // BPMEM_TX_SETIMAGE3 - Address of Texture in main memory
    // --------------------------------------------
    case TexUnitAddress::Register::SETIMAGE0:
    case TexUnitAddress::Register::SETIMAGE1:
    case TexUnitAddress::Register::SETIMAGE2:
    case TexUnitAddress::Register::SETIMAGE3:
      TMEM::ConfigurationChanged(tex_address, bp.newvalue);
      return;

    // -------------------------------
    // Set a TLUT
    // BPMEM_TX_SETTLUT - Format, TMEM Offset (offset of TLUT from start of TMEM high bank > > 5)
    // -------------------------------
    case TexUnitAddress::Register::SETTLUT:
      TMEM::ConfigurationChanged(tex_address, bp.newvalue);
      return;
    case TexUnitAddress::Register::UNKNOWN:
      break;  // Not handled
    }
  }

  switch (bp.address & 0xF0)
  {
  // --------------
  // Indirect Tev
  // --------------
  case BPMEM_IND_CMD:
    pixel_shader_manager.SetTevIndirectChanged();
    return;
  // --------------------------------------------------
  // Set Color/Alpha of a Tev
  // BPMEM_TEV_COLOR_ENV - Dest, Shift, Clamp, Sub, Bias, Sel A, Sel B, Sel C, Sel D
  // BPMEM_TEV_ALPHA_ENV - Dest, Shift, Clamp, Sub, Bias, Sel A, Sel B, Sel C, Sel D, T Swap, R Swap
  // --------------------------------------------------
  case BPMEM_TEV_COLOR_ENV:  // Texture Environment 1
  case BPMEM_TEV_COLOR_ENV + 16:
    pixel_shader_manager.SetTevCombiner((bp.address - BPMEM_TEV_COLOR_ENV) >> 1,
                                        (bp.address - BPMEM_TEV_COLOR_ENV) & 1, bp.newvalue);
    return;
  default:
    break;
  }

  DolphinAnalytics::Instance().ReportGameQuirk(GameQuirk::UsesUnknownBPCommand);
  WARN_LOG_FMT(VIDEO, "Unknown BP opcode: address = {:#010x} value = {:#010x}", bp.address,
               bp.newvalue);
}

// Call browser: OpcodeDecoding.cpp RunCallback::OnBP()
void LoadBPReg(u8 reg, u32 value, int cycles_into_future)
{
  auto& system = Core::System::GetInstance();

  int oldval = ((u32*)&bpmem)[reg];
  int newval = (oldval & ~bpmem.bpMask) | (value & bpmem.bpMask);
  int changes = (oldval ^ newval) & 0xFFFFFF;

  BPCmd bp = {reg, changes, newval};

  // Reset the mask register if we're not trying to set it ourselves.
  if (reg != BPMEM_BP_MASK)
    bpmem.bpMask = 0xFFFFFF;

  BPWritten(system.GetPixelShaderManager(), system.GetXFStateManager(),
            system.GetGeometryShaderManager(), bp, cycles_into_future);
}

void LoadBPRegPreprocess(u8 reg, u32 value, int cycles_into_future)
{
  auto& system = Core::System::GetInstance();

  // masking via BPMEM_BP_MASK could hypothetically be a problem
  u32 newval = value & 0xffffff;
  switch (reg)
  {
  case BPMEM_SETDRAWDONE:
    if ((newval & 0xff) == 0x02)
      system.GetPixelEngine().SetFinish(cycles_into_future);
    break;
  case BPMEM_PE_TOKEN_ID:
    system.GetPixelEngine().SetToken(newval & 0xffff, false, cycles_into_future);
    break;
  case BPMEM_PE_TOKEN_INT_ID:  // Pixel Engine Interrupt Token ID
    system.GetPixelEngine().SetToken(newval & 0xffff, true, cycles_into_future);
    break;
  }
}

std::pair<std::string, std::string> GetBPRegInfo(u8 cmd, u32 cmddata)
{
// Macro to set the register name and make sure it was written correctly via compile time assertion
#define RegName(reg) ((void)(reg), #reg)
#define DescriptionlessReg(reg) std::make_pair(RegName(reg), "");

  switch (cmd)
  {
  case BPMEM_GENMODE:  // 0x00
    return std::make_pair(RegName(BPMEM_GENMODE), fmt::to_string(GenMode{.hex = cmddata}));

  case BPMEM_DISPLAYCOPYFILTER:  // 0x01
  case BPMEM_DISPLAYCOPYFILTER + 1:
  case BPMEM_DISPLAYCOPYFILTER + 2:
  case BPMEM_DISPLAYCOPYFILTER + 3:
    // TODO: This is actually the sample pattern used for copies from an antialiased EFB
    return DescriptionlessReg(BPMEM_DISPLAYCOPYFILTER);
    // TODO: Description

  case BPMEM_IND_MTXA:  // 0x06
  case BPMEM_IND_MTXA + 3:
  case BPMEM_IND_MTXA + 6:
    return std::make_pair(fmt::format("BPMEM_IND_MTXA Matrix {}", (cmd - BPMEM_IND_MTXA) / 3),
                          fmt::format("Matrix {} column A\n{}", (cmd - BPMEM_IND_MTXA) / 3,
                                      IND_MTXA{.hex = cmddata}));

  case BPMEM_IND_MTXB:  // 0x07
  case BPMEM_IND_MTXB + 3:
  case BPMEM_IND_MTXB + 6:
    return std::make_pair(fmt::format("BPMEM_IND_MTXB Matrix {}", (cmd - BPMEM_IND_MTXB) / 3),
                          fmt::format("Matrix {} column B\n{}", (cmd - BPMEM_IND_MTXB) / 3,
                                      IND_MTXB{.hex = cmddata}));

  case BPMEM_IND_MTXC:  // 0x08
  case BPMEM_IND_MTXC + 3:
  case BPMEM_IND_MTXC + 6:
    return std::make_pair(fmt::format("BPMEM_IND_MTXC Matrix {}", (cmd - BPMEM_IND_MTXC) / 3),
                          fmt::format("Matrix {} column C\n{}", (cmd - BPMEM_IND_MTXC) / 3,
                                      IND_MTXC{.hex = cmddata}));

  case BPMEM_IND_IMASK:  // 0x0F
    return DescriptionlessReg(BPMEM_IND_IMASK);
    // TODO: Description

  case BPMEM_IND_CMD:  // 0x10
  case BPMEM_IND_CMD + 1:
  case BPMEM_IND_CMD + 2:
  case BPMEM_IND_CMD + 3:
  case BPMEM_IND_CMD + 4:
  case BPMEM_IND_CMD + 5:
  case BPMEM_IND_CMD + 6:
  case BPMEM_IND_CMD + 7:
  case BPMEM_IND_CMD + 8:
  case BPMEM_IND_CMD + 9:
  case BPMEM_IND_CMD + 10:
  case BPMEM_IND_CMD + 11:
  case BPMEM_IND_CMD + 12:
  case BPMEM_IND_CMD + 13:
  case BPMEM_IND_CMD + 14:
  case BPMEM_IND_CMD + 15:
    return std::make_pair(fmt::format("BPMEM_IND_CMD number {}", cmd - BPMEM_IND_CMD),
                          fmt::to_string(TevStageIndirect{.fullhex = cmddata}));

  case BPMEM_SCISSORTL:  // 0x20
    return std::make_pair(RegName(BPMEM_SCISSORTL), fmt::to_string(ScissorPos{.hex = cmddata}));

  case BPMEM_SCISSORBR:  // 0x21
    return std::make_pair(RegName(BPMEM_SCISSORBR), fmt::to_string(ScissorPos{.hex = cmddata}));

  case BPMEM_LINEPTWIDTH:  // 0x22
    return std::make_pair(RegName(BPMEM_LINEPTWIDTH), fmt::to_string(LPSize{.hex = cmddata}));

  case BPMEM_PERF0_TRI:  // 0x23
    return DescriptionlessReg(BPMEM_PERF0_TRI);
    // TODO: Description

  case BPMEM_PERF0_QUAD:  // 0x24
    return DescriptionlessReg(BPMEM_PERF0_QUAD);
    // TODO: Description

  case BPMEM_RAS1_SS0:  // 0x25
    return std::make_pair(RegName(BPMEM_RAS1_SS0),
                          fmt::to_string(std::make_pair(cmd, TEXSCALE{.hex = cmddata})));

  case BPMEM_RAS1_SS1:  // 0x26
    return std::make_pair(RegName(BPMEM_RAS1_SS1),
                          fmt::to_string(std::make_pair(cmd, TEXSCALE{.hex = cmddata})));

  case BPMEM_IREF:  // 0x27
    return std::make_pair(RegName(BPMEM_IREF), fmt::to_string(RAS1_IREF{.hex = cmddata}));

  case BPMEM_TREF:  // 0x28
  case BPMEM_TREF + 1:
  case BPMEM_TREF + 2:
  case BPMEM_TREF + 3:
  case BPMEM_TREF + 4:
  case BPMEM_TREF + 5:
  case BPMEM_TREF + 6:
  case BPMEM_TREF + 7:
    return std::make_pair(fmt::format("BPMEM_TREF number {}", cmd - BPMEM_TREF),
                          fmt::to_string(std::make_pair(cmd, TwoTevStageOrders{.hex = cmddata})));

  case BPMEM_SU_SSIZE:  // 0x30
  case BPMEM_SU_SSIZE + 2:
  case BPMEM_SU_SSIZE + 4:
  case BPMEM_SU_SSIZE + 6:
  case BPMEM_SU_SSIZE + 8:
  case BPMEM_SU_SSIZE + 10:
  case BPMEM_SU_SSIZE + 12:
  case BPMEM_SU_SSIZE + 14:
    return std::make_pair(fmt::format("BPMEM_SU_SSIZE number {}", (cmd - BPMEM_SU_SSIZE) / 2),
                          fmt::to_string(std::make_pair(true, TCInfo{.hex = cmddata})));

  case BPMEM_SU_TSIZE:  // 0x31
  case BPMEM_SU_TSIZE + 2:
  case BPMEM_SU_TSIZE + 4:
  case BPMEM_SU_TSIZE + 6:
  case BPMEM_SU_TSIZE + 8:
  case BPMEM_SU_TSIZE + 10:
  case BPMEM_SU_TSIZE + 12:
  case BPMEM_SU_TSIZE + 14:
    return std::make_pair(fmt::format("BPMEM_SU_TSIZE number {}", (cmd - BPMEM_SU_TSIZE) / 2),
                          fmt::to_string(std::make_pair(false, TCInfo{.hex = cmddata})));

  case BPMEM_ZMODE:  // 0x40
    return std::make_pair(RegName(BPMEM_ZMODE), fmt::format("Z mode: {}", ZMode{.hex = cmddata}));

  case BPMEM_BLENDMODE:  // 0x41
    return std::make_pair(RegName(BPMEM_BLENDMODE), fmt::to_string(BlendMode{.hex = cmddata}));

  case BPMEM_CONSTANTALPHA:  // 0x42
    return std::make_pair(RegName(BPMEM_CONSTANTALPHA),
                          fmt::to_string(ConstantAlpha{.hex = cmddata}));

  case BPMEM_ZCOMPARE:  // 0x43
    return std::make_pair(RegName(BPMEM_ZCOMPARE), fmt::to_string(PEControl{.hex = cmddata}));

  case BPMEM_FIELDMASK:  // 0x44
    return std::make_pair(RegName(BPMEM_FIELDMASK), fmt::to_string(FieldMask{.hex = cmddata}));

  case BPMEM_SETDRAWDONE:  // 0x45
    return DescriptionlessReg(BPMEM_SETDRAWDONE);
    // TODO: Description

  case BPMEM_BUSCLOCK0:  // 0x46
    return DescriptionlessReg(BPMEM_BUSCLOCK0);
    // TODO: Description

  case BPMEM_PE_TOKEN_ID:  // 0x47
    return DescriptionlessReg(BPMEM_PE_TOKEN_ID);
    // TODO: Description

  case BPMEM_PE_TOKEN_INT_ID:  // 0x48
    return DescriptionlessReg(BPMEM_PE_TOKEN_INT_ID);
    // TODO: Description

  case BPMEM_EFB_TL:  // 0x49
  {
    const X10Y10 left_top{.hex = cmddata};
    return std::make_pair(RegName(BPMEM_EFB_TL),
                          fmt::format("EFB Left: {}\nEFB Top: {}", left_top.x, left_top.y));
  }

  case BPMEM_EFB_WH:  // 0x4A
  {
    const X10Y10 width_height{.hex = cmddata};
    return std::make_pair(
        RegName(BPMEM_EFB_WH),
        fmt::format("EFB Width: {}\nEFB Height: {}", width_height.x + 1, width_height.y + 1));
  }

  case BPMEM_EFB_ADDR:  // 0x4B
    return std::make_pair(
        RegName(BPMEM_EFB_ADDR),
        fmt::format("EFB Target address (32 byte aligned): 0x{:06X}", cmddata << 5));

  case BPMEM_EFB_STRIDE:  // 0x4D
    return std::make_pair(
        RegName(BPMEM_EFB_STRIDE),
        fmt::format("EFB destination stride (32 byte aligned): 0x{:06X}", cmddata << 5));

  case BPMEM_COPYYSCALE:  // 0x4E
    return std::make_pair(
        RegName(BPMEM_COPYYSCALE),
        fmt::format("Y scaling factor (XFB copy only): 0x{:X} ({}, reciprocal {})", cmddata,
                    static_cast<float>(cmddata) / 256.f, 256.f / static_cast<float>(cmddata)));

  case BPMEM_CLEAR_AR:  // 0x4F
    return std::make_pair(RegName(BPMEM_CLEAR_AR),
                          fmt::format("Clear color alpha: 0x{:02X}\nClear color red: 0x{:02X}",
                                      (cmddata & 0xFF00) >> 8, cmddata & 0xFF));

  case BPMEM_CLEAR_GB:  // 0x50
    return std::make_pair(RegName(BPMEM_CLEAR_GB),
                          fmt::format("Clear color green: 0x{:02X}\nClear color blue: 0x{:02X}",
                                      (cmddata & 0xFF00) >> 8, cmddata & 0xFF));

  case BPMEM_CLEAR_Z:  // 0x51
    return std::make_pair(RegName(BPMEM_CLEAR_Z), fmt::format("Clear Z value: 0x{:06X}", cmddata));

  case BPMEM_TRIGGER_EFB_COPY:  // 0x52
    return std::make_pair(RegName(BPMEM_TRIGGER_EFB_COPY),
                          fmt::to_string(UPE_Copy{.Hex = cmddata}));

  case BPMEM_COPYFILTER0:  // 0x53
  {
    const u32 w0 = (cmddata & 0x00003f);
    const u32 w1 = (cmddata & 0x000fc0) >> 6;
    const u32 w2 = (cmddata & 0x03f000) >> 12;
    const u32 w3 = (cmddata & 0xfc0000) >> 18;
    return std::make_pair(RegName(BPMEM_COPYFILTER0),
                          fmt::format("w0: {}\nw1: {}\nw2: {}\nw3: {}", w0, w1, w2, w3));
  }

  case BPMEM_COPYFILTER1:  // 0x54
  {
    const u32 w4 = (cmddata & 0x00003f);
    const u32 w5 = (cmddata & 0x000fc0) >> 6;
    const u32 w6 = (cmddata & 0x03f000) >> 12;
    // There is no w7
    return std::make_pair(RegName(BPMEM_COPYFILTER1),
                          fmt::format("w4: {}\nw5: {}\nw6: {}", w4, w5, w6));
  }

  case BPMEM_CLEARBBOX1:  // 0x55
    return std::make_pair(RegName(BPMEM_CLEARBBOX1),
                          fmt::format("Bounding Box index 0: {}\nBounding Box index 1: {}",
                                      cmddata & 0x3ff, (cmddata >> 10) & 0x3ff));

  case BPMEM_CLEARBBOX2:  // 0x56
    return std::make_pair(RegName(BPMEM_CLEARBBOX2),
                          fmt::format("Bounding Box index 2: {}\nBounding Box index 3: {}",
                                      cmddata & 0x3ff, (cmddata >> 10) & 0x3ff));

  case BPMEM_CLEAR_PIXEL_PERF:  // 0x57
    return DescriptionlessReg(BPMEM_CLEAR_PIXEL_PERF);
    // TODO: Description

  case BPMEM_REVBITS:  // 0x58
    return DescriptionlessReg(BPMEM_REVBITS);
    // TODO: Description

  case BPMEM_SCISSOROFFSET:  // 0x59
    return std::make_pair(RegName(BPMEM_SCISSOROFFSET),
                          fmt::to_string(ScissorOffset{.hex = cmddata}));

  case BPMEM_PRELOAD_ADDR:  // 0x60
    return std::make_pair(
        RegName(BPMEM_PRELOAD_ADDR),
        fmt::format("Tmem preload address (32 byte aligned, in main memory): 0x{:06x}",
                    cmddata << 5));

  case BPMEM_PRELOAD_TMEMEVEN:  // 0x61
    return std::make_pair(RegName(BPMEM_PRELOAD_TMEMEVEN),
                          fmt::format("Tmem preload even line: 0x{:04x} (byte 0x{:05x})", cmddata,
                                      cmddata * TMEM_LINE_SIZE));

  case BPMEM_PRELOAD_TMEMODD:  // 0x62
    return std::make_pair(RegName(BPMEM_PRELOAD_TMEMODD),
                          fmt::format("Tmem preload odd line: 0x{:04x} (byte 0x{:05x})", cmddata,
                                      cmddata * TMEM_LINE_SIZE));

  case BPMEM_PRELOAD_MODE:  // 0x63
    return std::make_pair(RegName(BPMEM_PRELOAD_MODE),
                          fmt::to_string(BPU_PreloadTileInfo{.hex = cmddata}));

  case BPMEM_LOADTLUT0:  // 0x64
    return std::make_pair(
        RegName(BPMEM_LOADTLUT0),
        fmt::format("TLUT load address (32 byte aligned, in main memory): 0x{:06x}", cmddata << 5));

  case BPMEM_LOADTLUT1:  // 0x65
    return std::make_pair(RegName(BPMEM_LOADTLUT1),
                          fmt::to_string(BPU_LoadTlutInfo{.hex = cmddata}));

  case BPMEM_TEXINVALIDATE:  // 0x66
    return DescriptionlessReg(BPMEM_TEXINVALIDATE);
    // TODO: Description

  case BPMEM_PERF1:  // 0x67
    return DescriptionlessReg(BPMEM_PERF1);
    // TODO: Description

  case BPMEM_FIELDMODE:  // 0x68
    return std::make_pair(RegName(BPMEM_FIELDMODE), fmt::to_string(FieldMode{.hex = cmddata}));

  case BPMEM_BUSCLOCK1:  // 0x69
    return DescriptionlessReg(BPMEM_BUSCLOCK1);
    // TODO: Description

  case BPMEM_TX_SETMODE0:  // 0x80
  case BPMEM_TX_SETMODE0 + 1:
  case BPMEM_TX_SETMODE0 + 2:
  case BPMEM_TX_SETMODE0 + 3:
    return std::make_pair(fmt::format("BPMEM_TX_SETMODE0 Texture Unit {}", cmd - BPMEM_TX_SETMODE0),
                          fmt::to_string(TexMode0{.hex = cmddata}));

  case BPMEM_TX_SETMODE1:  // 0x84
  case BPMEM_TX_SETMODE1 + 1:
  case BPMEM_TX_SETMODE1 + 2:
  case BPMEM_TX_SETMODE1 + 3:
    return std::make_pair(fmt::format("BPMEM_TX_SETMODE1 Texture Unit {}", cmd - BPMEM_TX_SETMODE1),
                          fmt::to_string(TexMode1{.hex = cmddata}));

  case BPMEM_TX_SETIMAGE0:  // 0x88
  case BPMEM_TX_SETIMAGE0 + 1:
  case BPMEM_TX_SETIMAGE0 + 2:
  case BPMEM_TX_SETIMAGE0 + 3:
    return std::make_pair(
        fmt::format("BPMEM_TX_SETIMAGE0 Texture Unit {}", cmd - BPMEM_TX_SETIMAGE0),
        fmt::to_string(TexImage0{.hex = cmddata}));

  case BPMEM_TX_SETIMAGE1:  // 0x8C
  case BPMEM_TX_SETIMAGE1 + 1:
  case BPMEM_TX_SETIMAGE1 + 2:
  case BPMEM_TX_SETIMAGE1 + 3:
    return std::make_pair(
        fmt::format("BPMEM_TX_SETIMAGE1 Texture Unit {}", cmd - BPMEM_TX_SETIMAGE1),
        fmt::to_string(TexImage1{.hex = cmddata}));

  case BPMEM_TX_SETIMAGE2:  // 0x90
  case BPMEM_TX_SETIMAGE2 + 1:
  case BPMEM_TX_SETIMAGE2 + 2:
  case BPMEM_TX_SETIMAGE2 + 3:
    return std::make_pair(
        fmt::format("BPMEM_TX_SETIMAGE2 Texture Unit {}", cmd - BPMEM_TX_SETIMAGE2),
        fmt::to_string(TexImage2{.hex = cmddata}));

  case BPMEM_TX_SETIMAGE3:  // 0x94
  case BPMEM_TX_SETIMAGE3 + 1:
  case BPMEM_TX_SETIMAGE3 + 2:
  case BPMEM_TX_SETIMAGE3 + 3:
    return std::make_pair(
        fmt::format("BPMEM_TX_SETIMAGE3 Texture Unit {}", cmd - BPMEM_TX_SETIMAGE3),
        fmt::to_string(TexImage3{.hex = cmddata}));

  case BPMEM_TX_SETTLUT:  // 0x98
  case BPMEM_TX_SETTLUT + 1:
  case BPMEM_TX_SETTLUT + 2:
  case BPMEM_TX_SETTLUT + 3:
    return std::make_pair(fmt::format("BPMEM_TX_SETTLUT Texture Unit {}", cmd - BPMEM_TX_SETTLUT),
                          fmt::to_string(TexTLUT{.hex = cmddata}));

  case BPMEM_TX_SETMODE0_4:  // 0xA0
  case BPMEM_TX_SETMODE0_4 + 1:
  case BPMEM_TX_SETMODE0_4 + 2:
  case BPMEM_TX_SETMODE0_4 + 3:
    return std::make_pair(
        fmt::format("BPMEM_TX_SETMODE0_4 Texture Unit {}", cmd - BPMEM_TX_SETMODE0_4 + 4),
        fmt::to_string(TexMode0{.hex = cmddata}));

  case BPMEM_TX_SETMODE1_4:  // 0xA4
  case BPMEM_TX_SETMODE1_4 + 1:
  case BPMEM_TX_SETMODE1_4 + 2:
  case BPMEM_TX_SETMODE1_4 + 3:
    return std::make_pair(
        fmt::format("BPMEM_TX_SETMODE1_4 Texture Unit {}", cmd - BPMEM_TX_SETMODE1_4 + 4),
        fmt::to_string(TexMode1{.hex = cmddata}));

  case BPMEM_TX_SETIMAGE0_4:  // 0xA8
  case BPMEM_TX_SETIMAGE0_4 + 1:
  case BPMEM_TX_SETIMAGE0_4 + 2:
  case BPMEM_TX_SETIMAGE0_4 + 3:
    return std::make_pair(
        fmt::format("BPMEM_TX_SETIMAGE0_4 Texture Unit {}", cmd - BPMEM_TX_SETIMAGE0_4 + 4),
        fmt::to_string(TexImage0{.hex = cmddata}));

  case BPMEM_TX_SETIMAGE1_4:  // 0xAC
  case BPMEM_TX_SETIMAGE1_4 + 1:
  case BPMEM_TX_SETIMAGE1_4 + 2:
  case BPMEM_TX_SETIMAGE1_4 + 3:
    return std::make_pair(
        fmt::format("BPMEM_TX_SETIMAGE1_4 Texture Unit {}", cmd - BPMEM_TX_SETIMAGE1_4 + 4),
        fmt::to_string(TexImage1{.hex = cmddata}));

  case BPMEM_TX_SETIMAGE2_4:  // 0xB0
  case BPMEM_TX_SETIMAGE2_4 + 1:
  case BPMEM_TX_SETIMAGE2_4 + 2:
  case BPMEM_TX_SETIMAGE2_4 + 3:
    return std::make_pair(
        fmt::format("BPMEM_TX_SETIMAGE2_4 Texture Unit {}", cmd - BPMEM_TX_SETIMAGE2_4 + 4),
        fmt::to_string(TexImage2{.hex = cmddata}));

  case BPMEM_TX_SETIMAGE3_4:  // 0xB4
  case BPMEM_TX_SETIMAGE3_4 + 1:
  case BPMEM_TX_SETIMAGE3_4 + 2:
  case BPMEM_TX_SETIMAGE3_4 + 3:
    return std::make_pair(
        fmt::format("BPMEM_TX_SETIMAGE3_4 Texture Unit {}", cmd - BPMEM_TX_SETIMAGE3_4 + 4),
        fmt::to_string(TexImage3{.hex = cmddata}));

  case BPMEM_TX_SETTLUT_4:  // 0xB8
  case BPMEM_TX_SETTLUT_4 + 1:
  case BPMEM_TX_SETTLUT_4 + 2:
  case BPMEM_TX_SETTLUT_4 + 3:
    return std::make_pair(
        fmt::format("BPMEM_TX_SETTLUT_4 Texture Unit {}", cmd - BPMEM_TX_SETTLUT_4 + 4),
        fmt::to_string(TexTLUT{.hex = cmddata}));

  case BPMEM_TEV_COLOR_ENV:  // 0xC0
  case BPMEM_TEV_COLOR_ENV + 2:
  case BPMEM_TEV_COLOR_ENV + 4:
  case BPMEM_TEV_COLOR_ENV + 6:
  case BPMEM_TEV_COLOR_ENV + 8:
  case BPMEM_TEV_COLOR_ENV + 10:
  case BPMEM_TEV_COLOR_ENV + 12:
  case BPMEM_TEV_COLOR_ENV + 14:
  case BPMEM_TEV_COLOR_ENV + 16:
  case BPMEM_TEV_COLOR_ENV + 18:
  case BPMEM_TEV_COLOR_ENV + 20:
  case BPMEM_TEV_COLOR_ENV + 22:
  case BPMEM_TEV_COLOR_ENV + 24:
  case BPMEM_TEV_COLOR_ENV + 26:
  case BPMEM_TEV_COLOR_ENV + 28:
  case BPMEM_TEV_COLOR_ENV + 30:
    return std::make_pair(
        fmt::format("BPMEM_TEV_COLOR_ENV Tev stage {}", (cmd - BPMEM_TEV_COLOR_ENV) / 2),
        fmt::to_string(TevStageCombiner::ColorCombiner{.hex = cmddata}));

  case BPMEM_TEV_ALPHA_ENV:  // 0xC1
  case BPMEM_TEV_ALPHA_ENV + 2:
  case BPMEM_TEV_ALPHA_ENV + 4:
  case BPMEM_TEV_ALPHA_ENV + 6:
  case BPMEM_TEV_ALPHA_ENV + 8:
  case BPMEM_TEV_ALPHA_ENV + 10:
  case BPMEM_TEV_ALPHA_ENV + 12:
  case BPMEM_TEV_ALPHA_ENV + 14:
  case BPMEM_TEV_ALPHA_ENV + 16:
  case BPMEM_TEV_ALPHA_ENV + 18:
  case BPMEM_TEV_ALPHA_ENV + 20:
  case BPMEM_TEV_ALPHA_ENV + 22:
  case BPMEM_TEV_ALPHA_ENV + 24:
  case BPMEM_TEV_ALPHA_ENV + 26:
  case BPMEM_TEV_ALPHA_ENV + 28:
  case BPMEM_TEV_ALPHA_ENV + 30:
    return std::make_pair(
        fmt::format("BPMEM_TEV_ALPHA_ENV Tev stage {}", (cmd - BPMEM_TEV_ALPHA_ENV) / 2),
        fmt::to_string(TevStageCombiner::AlphaCombiner{.hex = cmddata}));

  case BPMEM_TEV_COLOR_RA:      // 0xE0
  case BPMEM_TEV_COLOR_RA + 2:  // 0xE2
  case BPMEM_TEV_COLOR_RA + 4:  // 0xE4
  case BPMEM_TEV_COLOR_RA + 6:  // 0xE6
    return std::make_pair(
        fmt::format("BPMEM_TEV_COLOR_RA Tev register {}", (cmd - BPMEM_TEV_COLOR_RA) / 2),
        fmt::to_string(TevReg::RA{.hex = cmddata}));

  case BPMEM_TEV_COLOR_BG:      // 0xE1
  case BPMEM_TEV_COLOR_BG + 2:  // 0xE3
  case BPMEM_TEV_COLOR_BG + 4:  // 0xE5
  case BPMEM_TEV_COLOR_BG + 6:  // 0xE7
    return std::make_pair(
        fmt::format("BPMEM_TEV_COLOR_BG Tev register {}", (cmd - BPMEM_TEV_COLOR_BG) / 2),
        fmt::to_string(TevReg::BG{.hex = cmddata}));

  case BPMEM_FOGRANGE:  // 0xE8
    return std::make_pair("BPMEM_FOGRANGE Base",
                          fmt::to_string(FogRangeParams::RangeBase{.hex = cmddata}));

  case BPMEM_FOGRANGE + 1:
  case BPMEM_FOGRANGE + 2:
  case BPMEM_FOGRANGE + 3:
  case BPMEM_FOGRANGE + 4:
  case BPMEM_FOGRANGE + 5:
    return std::make_pair(fmt::format("BPMEM_FOGRANGE K element {}", cmd - BPMEM_FOGRANGE),
                          fmt::to_string(FogRangeKElement{.HEX = cmddata}));

  case BPMEM_FOGPARAM0:  // 0xEE
    return std::make_pair(RegName(BPMEM_FOGPARAM0), fmt::to_string(FogParam0{.hex = cmddata}));

  case BPMEM_FOGBMAGNITUDE:  // 0xEF
    return std::make_pair(RegName(BPMEM_FOGBMAGNITUDE), fmt::format("B magnitude: {}", cmddata));

  case BPMEM_FOGBEXPONENT:  // 0xF0
    return std::make_pair(RegName(BPMEM_FOGBEXPONENT),
                          fmt::format("B shift: 1>>{} (1/{})", cmddata, 1 << cmddata));

  case BPMEM_FOGPARAM3:  // 0xF1
    return std::make_pair(RegName(BPMEM_FOGPARAM3), fmt::to_string(FogParam3{.hex = cmddata}));

  case BPMEM_FOGCOLOR:  // 0xF2
    return std::make_pair(RegName(BPMEM_FOGCOLOR),
                          fmt::to_string(FogParams::FogColor{.hex = cmddata}));

  case BPMEM_ALPHACOMPARE:  // 0xF3
    return std::make_pair(RegName(BPMEM_ALPHACOMPARE), fmt::to_string(AlphaTest{.hex = cmddata}));

  case BPMEM_BIAS:  // 0xF4
    return std::make_pair(RegName(BPMEM_BIAS), fmt::to_string(ZTex1{.hex = cmddata}));

  case BPMEM_ZTEX2:  // 0xF5
    return std::make_pair(RegName(BPMEM_ZTEX2), fmt::to_string(ZTex2{.hex = cmddata}));

  case BPMEM_TEV_KSEL:  // 0xF6
  case BPMEM_TEV_KSEL + 1:
  case BPMEM_TEV_KSEL + 2:
  case BPMEM_TEV_KSEL + 3:
  case BPMEM_TEV_KSEL + 4:
  case BPMEM_TEV_KSEL + 5:
  case BPMEM_TEV_KSEL + 6:
  case BPMEM_TEV_KSEL + 7:
    return std::make_pair(fmt::format("BPMEM_TEV_KSEL number {}", cmd - BPMEM_TEV_KSEL),
                          fmt::to_string(std::make_pair(cmd, TevKSel{.hex = cmddata})));

  case BPMEM_BP_MASK:  // 0xFE
    return std::make_pair(RegName(BPMEM_BP_MASK),
                          fmt::format("The next BP command will only update these bits; others "
                                      "will retain their prior values: {:06x}",
                                      cmddata));

  default:
    return std::make_pair(fmt::format("Unknown BP Reg: {:02x}={:06x}", cmd, cmddata), "");

#undef DescriptionlessReg
#undef RegName
  }
}

// Called when loading a saved state.
void BPReload()
{
  // restore anything that goes straight to the renderer.
  // let's not risk actually replaying any writes.
  // note that PixelShaderManager is already covered since it has its own DoState.
  SetGenerationMode();
  BPFunctions::SetScissorAndViewport(g_framebuffer_manager.get(), bpmem.scissorTL, bpmem.scissorBR,
                                     bpmem.scissorOffset, xfmem.viewport);
  SetDepthMode();
  SetBlendMode();
  OnPixelFormatChange(g_framebuffer_manager.get(), bpmem.zcontrol.pixel_format,
                      bpmem.zcontrol.zformat);
}
