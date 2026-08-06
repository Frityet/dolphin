// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>

#include "Core/HW/WiimoteEmu/Extension/Nunchuk.h"
#include "Core/HW/WiimoteEmu/ScriptedInput.h"

namespace
{
class ScopedEnvironment final
{
public:
  ScopedEnvironment(const char* name, std::string_view value) : m_name(name)
  {
    if (const char* const previous = std::getenv(name))
      m_previous = previous;
    Set(value);
  }

  ~ScopedEnvironment()
  {
    if (m_previous)
      Set(*m_previous);
    else
      Unset();
  }

private:
  void Set(std::string_view value)
  {
#ifdef _WIN32
    _putenv_s(m_name.c_str(), std::string(value).c_str());
#else
    setenv(m_name.c_str(), std::string(value).c_str(), 1);
#endif
  }

  void Unset()
  {
#ifdef _WIN32
    _putenv_s(m_name.c_str(), "");
#else
    unsetenv(m_name.c_str());
#endif
  }

  std::string m_name;
  std::optional<std::string> m_previous;
};

std::optional<ControlState> ReadControl(const ControllerEmu::InputOverrideFunction& input,
                                        std::string_view group, std::string_view control,
                                        ControlState state = 0.0)
{
  return input(group, control, state);
}
}  // namespace

TEST(ScriptedInput, NunchukStickUsesInclusiveRangesAndLastMatch)
{
  ScopedEnvironment stick_script("SMGPC_DOLPHIN_NUNCHUK_STICK_SCRIPT",
                                 "10-20:-0.25,0.75;15-17:0.5,-0.5;broken;30-:1,0");
  ScopedEnvironment button_script("SMGPC_DOLPHIN_NUNCHUK_BUTTON_SCRIPT", "12-13:C;13:Z");
  const auto input = WiimoteEmu::CreateScriptedNunchukInputOverride(0);
  ASSERT_TRUE(input);

  WiimoteEmu::SetScriptedInputFrame(10);
  EXPECT_EQ(-0.25, *ReadControl(input, WiimoteEmu::Nunchuk::STICK_GROUP,
                                ControllerEmu::ReshapableInput::X_INPUT_OVERRIDE));
  EXPECT_EQ(0.75, *ReadControl(input, WiimoteEmu::Nunchuk::STICK_GROUP,
                               ControllerEmu::ReshapableInput::Y_INPUT_OVERRIDE));

  WiimoteEmu::SetScriptedInputFrame(15);
  EXPECT_EQ(0.5, *ReadControl(input, WiimoteEmu::Nunchuk::STICK_GROUP,
                              ControllerEmu::ReshapableInput::X_INPUT_OVERRIDE));
  EXPECT_EQ(-0.5, *ReadControl(input, WiimoteEmu::Nunchuk::STICK_GROUP,
                               ControllerEmu::ReshapableInput::Y_INPUT_OVERRIDE));

  WiimoteEmu::SetScriptedInputFrame(21);
  EXPECT_EQ(0.125, *ReadControl(input, WiimoteEmu::Nunchuk::STICK_GROUP,
                                ControllerEmu::ReshapableInput::X_INPUT_OVERRIDE, 0.125));

  WiimoteEmu::SetScriptedInputFrame(30);
  EXPECT_EQ(1.0, *ReadControl(input, WiimoteEmu::Nunchuk::STICK_GROUP,
                              ControllerEmu::ReshapableInput::X_INPUT_OVERRIDE));
}

TEST(ScriptedInput, NunchukButtonsComposeAndPreserveOrdinaryInput)
{
  ScopedEnvironment stick_script("SMGPC_DOLPHIN_NUNCHUK_STICK_SCRIPT", "10-20:0,1");
  ScopedEnvironment button_script("SMGPC_DOLPHIN_NUNCHUK_BUTTON_SCRIPT", "12-13:C;13:Z");
  const auto input = WiimoteEmu::CreateScriptedNunchukInputOverride(0);
  ASSERT_TRUE(input);

  WiimoteEmu::SetScriptedInputFrame(12);
  EXPECT_EQ(1.0,
            *ReadControl(input, WiimoteEmu::Nunchuk::BUTTONS_GROUP, WiimoteEmu::Nunchuk::C_BUTTON));
  EXPECT_EQ(0.0,
            *ReadControl(input, WiimoteEmu::Nunchuk::BUTTONS_GROUP, WiimoteEmu::Nunchuk::Z_BUTTON));

  WiimoteEmu::SetScriptedInputFrame(13);
  EXPECT_EQ(1.0,
            *ReadControl(input, WiimoteEmu::Nunchuk::BUTTONS_GROUP, WiimoteEmu::Nunchuk::C_BUTTON));
  EXPECT_EQ(1.0,
            *ReadControl(input, WiimoteEmu::Nunchuk::BUTTONS_GROUP, WiimoteEmu::Nunchuk::Z_BUTTON));

  WiimoteEmu::SetScriptedInputFrame(14);
  EXPECT_EQ(0.25, *ReadControl(input, WiimoteEmu::Nunchuk::BUTTONS_GROUP,
                               WiimoteEmu::Nunchuk::C_BUTTON, 0.25));
  EXPECT_FALSE(ReadControl(input, "Other", "Other"));
}

TEST(ScriptedInput, NunchukRejectsInvalidOnlyScriptsAndOtherControllers)
{
  ScopedEnvironment stick_script("SMGPC_DOLPHIN_NUNCHUK_STICK_SCRIPT", "bad;20-10:0,1;10:1.1,0");
  ScopedEnvironment button_script("SMGPC_DOLPHIN_NUNCHUK_BUTTON_SCRIPT", "10:A");
  EXPECT_FALSE(WiimoteEmu::CreateScriptedNunchukInputOverride(0));
  EXPECT_FALSE(WiimoteEmu::CreateScriptedNunchukInputOverride(1));
}
