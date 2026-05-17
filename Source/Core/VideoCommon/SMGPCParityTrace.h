// Copyright 2009 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "Common/CommonTypes.h"
#include "VideoCommon/RenderState.h"

void RecordSMGPCDolphinDrawTrace(u32 draw_index, PrimitiveType primitive_type, u32 num_vertices,
                                 u32 num_indices, u32 base_vertex, u32 base_index,
                                 u32 used_textures_mask);
