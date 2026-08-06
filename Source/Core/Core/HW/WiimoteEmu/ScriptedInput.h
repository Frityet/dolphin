// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "Common/CommonTypes.h"

#include "InputCommon/ControllerEmu/ControlGroup/ControlGroup.h"

namespace WiimoteEmu
{
// Returns an input override driven by optional environment-provided frame scripts. An empty
// function means that scripted input was not requested for this Wii Remote.
ControllerEmu::InputOverrideFunction CreateScriptedInputOverride(unsigned int wiimote_index);

// Keeps frame scripts on the same unique-presented-frame clock used by frame capture and render
// traces. This is called from the presenter and is safe to read from the emulation thread.
void SetScriptedInputFrame(u64 frame);
}  // namespace WiimoteEmu
