// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Core/HW/WiimoteEmu/ScriptedInput.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "Common/CommonTypes.h"
#include "Common/Logging/Log.h"

#include "Core/HW/WiimoteEmu/Extension/Nunchuk.h"
#include "Core/HW/WiimoteEmu/WiimoteEmu.h"
#include "InputCommon/ControllerEmu/ControllerEmu.h"

namespace WiimoteEmu
{
namespace
{
constexpr std::string_view BUTTON_SCRIPT_ENV = "SMGPC_DOLPHIN_WPAD_BUTTON_SCRIPT";
constexpr std::string_view POINTER_SCRIPT_ENV = "SMGPC_DOLPHIN_WPAD_POINTER_SCRIPT";
constexpr std::string_view POINTER_WIDTH_ENV = "SMGPC_DOLPHIN_WPAD_POINTER_WIDTH";
constexpr std::string_view POINTER_HEIGHT_ENV = "SMGPC_DOLPHIN_WPAD_POINTER_HEIGHT";
constexpr std::string_view NUNCHUK_STICK_SCRIPT_ENV = "SMGPC_DOLPHIN_NUNCHUK_STICK_SCRIPT";
constexpr std::string_view NUNCHUK_BUTTON_SCRIPT_ENV = "SMGPC_DOLPHIN_NUNCHUK_BUTTON_SCRIPT";

constexpr double DEFAULT_POINTER_WIDTH = 640.0;
constexpr double DEFAULT_POINTER_HEIGHT = 456.0;

std::atomic<u64> s_current_frame{0};

struct FrameRange
{
  u64 first = 0;
  u64 last = 0;
};

struct ButtonSpan
{
  FrameRange range;
  u16 mask = 0;
};

struct PointerSpan
{
  FrameRange range;
  double x = 0.0;
  double y = 0.0;
  bool visible = true;
};

struct StickSpan
{
  FrameRange range;
  ControlState x = 0.0;
  ControlState y = 0.0;
};

struct ScriptedInput
{
  std::vector<ButtonSpan> button_spans;
  std::vector<PointerSpan> pointer_spans;
  double pointer_width = DEFAULT_POINTER_WIDTH;
  double pointer_height = DEFAULT_POINTER_HEIGHT;
};

struct ScriptedNunchukInput
{
  std::vector<ButtonSpan> button_spans;
  std::vector<StickSpan> stick_spans;
};

std::string_view Trim(std::string_view text)
{
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0)
    text.remove_prefix(1);
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0)
    text.remove_suffix(1);
  return text;
}

std::optional<std::string> ReadEnvironment(std::string_view name)
{
  const std::string key{name};
  const char* const value = std::getenv(key.c_str());
  if (value == nullptr || value[0] == '\0')
    return std::nullopt;
  return std::string{value};
}

template <typename T>
std::optional<T> ParseNumber(std::string_view text)
{
  text = Trim(text);
  if (text.empty())
    return std::nullopt;

  T value{};
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc{} || result.ptr != end)
    return std::nullopt;
  return value;
}

std::optional<FrameRange> ParseFrameRange(std::string_view text)
{
  text = Trim(text);
  const std::size_t dash = text.find('-');
  if (dash == std::string_view::npos)
  {
    const auto frame = ParseNumber<u64>(text);
    if (!frame)
      return std::nullopt;
    return FrameRange{*frame, *frame};
  }

  const auto first = ParseNumber<u64>(text.substr(0, dash));
  if (!first)
    return std::nullopt;

  const std::string_view last_text = Trim(text.substr(dash + 1));
  if (last_text.empty())
    return FrameRange{*first, std::numeric_limits<u64>::max()};

  const auto last = ParseNumber<u64>(last_text);
  if (!last || *last < *first)
    return std::nullopt;
  return FrameRange{*first, *last};
}

std::string Uppercase(std::string_view text)
{
  std::string result{Trim(text)};
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char value) { return static_cast<char>(std::toupper(value)); });
  return result;
}

std::optional<u16> ParseButton(std::string_view text)
{
  const std::string token = Uppercase(text);
  if (token == "A")
    return Wiimote::BUTTON_A;
  if (token == "B")
    return Wiimote::BUTTON_B;
  if (token == "ONE" || token == "1")
    return Wiimote::BUTTON_ONE;
  if (token == "TWO" || token == "2")
    return Wiimote::BUTTON_TWO;
  if (token == "MINUS" || token == "-")
    return Wiimote::BUTTON_MINUS;
  if (token == "PLUS")
    return Wiimote::BUTTON_PLUS;
  if (token == "HOME")
    return Wiimote::BUTTON_HOME;
  if (token == "UP")
    return Wiimote::PAD_UP;
  if (token == "DOWN")
    return Wiimote::PAD_DOWN;
  if (token == "LEFT")
    return Wiimote::PAD_LEFT;
  if (token == "RIGHT")
    return Wiimote::PAD_RIGHT;
  return std::nullopt;
}

std::optional<u16> ParseNunchukButton(std::string_view text)
{
  const std::string token = Uppercase(text);
  if (token == "C")
    return Nunchuk::BUTTON_C;
  if (token == "Z")
    return Nunchuk::BUTTON_Z;
  return std::nullopt;
}

std::optional<bool> ParseBool(std::string_view text)
{
  const std::string value = Uppercase(text);
  if (value == "1" || value == "TRUE" || value == "ON")
    return true;
  if (value == "0" || value == "FALSE" || value == "OFF")
    return false;
  return std::nullopt;
}

bool IsActive(const FrameRange& range, u64 frame)
{
  return frame >= range.first && frame <= range.last;
}

template <typename ButtonParser>
void ParseButtonScript(std::string_view text, std::vector<ButtonSpan>* spans,
                       ButtonParser parse_button, std::string_view controller_name)
{
  std::size_t invalid_entries = 0;
  while (!text.empty())
  {
    const std::size_t separator = text.find(';');
    const std::string_view entry = Trim(text.substr(0, separator));
    if (!entry.empty())
    {
      const std::size_t colon = entry.find(':');
      const auto range =
          colon == std::string_view::npos ? std::nullopt : ParseFrameRange(entry.substr(0, colon));
      u16 mask = 0;
      bool valid_buttons = colon != std::string_view::npos;
      std::string_view buttons =
          colon == std::string_view::npos ? std::string_view{} : entry.substr(colon + 1);
      while (valid_buttons)
      {
        const std::size_t plus = buttons.find('+');
        const auto button = parse_button(buttons.substr(0, plus));
        if (!button)
        {
          valid_buttons = false;
          break;
        }
        mask |= *button;
        if (plus == std::string_view::npos)
          break;
        buttons.remove_prefix(plus + 1);
      }

      if (range && valid_buttons && mask != 0)
        spans->push_back(ButtonSpan{*range, mask});
      else
        ++invalid_entries;
    }

    if (separator == std::string_view::npos)
      break;
    text.remove_prefix(separator + 1);
  }

  if (invalid_entries != 0)
  {
    WARN_LOG_FMT(WIIMOTE, "Ignored {} malformed frame-scripted {} button span(s).", invalid_entries,
                 controller_name);
  }
}

void ParseStickScript(std::string_view text, ScriptedNunchukInput* script)
{
  std::size_t invalid_entries = 0;
  while (!text.empty())
  {
    const std::size_t separator = text.find(';');
    const std::string_view entry = Trim(text.substr(0, separator));
    if (!entry.empty())
    {
      const std::size_t colon = entry.find(':');
      const auto range =
          colon == std::string_view::npos ? std::nullopt : ParseFrameRange(entry.substr(0, colon));
      const std::string_view values =
          colon == std::string_view::npos ? std::string_view{} : entry.substr(colon + 1);
      const std::size_t comma = values.find(',');
      const auto x = comma == std::string_view::npos ?
                         std::nullopt :
                         ParseNumber<ControlState>(values.substr(0, comma));
      const auto y = comma == std::string_view::npos ?
                         std::nullopt :
                         ParseNumber<ControlState>(values.substr(comma + 1));

      if (range && x && y && std::isfinite(*x) && std::isfinite(*y) && *x >= -1.0 && *x <= 1.0 &&
          *y >= -1.0 && *y <= 1.0)
      {
        script->stick_spans.push_back(StickSpan{*range, *x, *y});
      }
      else
      {
        ++invalid_entries;
      }
    }

    if (separator == std::string_view::npos)
      break;
    text.remove_prefix(separator + 1);
  }

  if (invalid_entries != 0)
  {
    WARN_LOG_FMT(WIIMOTE, "Ignored {} malformed frame-scripted Nunchuk stick span(s).",
                 invalid_entries);
  }
}

void ParsePointerScript(std::string_view text, ScriptedInput* script)
{
  std::size_t invalid_entries = 0;
  while (!text.empty())
  {
    const std::size_t separator = text.find(';');
    const std::string_view entry = Trim(text.substr(0, separator));
    if (!entry.empty())
    {
      const std::size_t colon = entry.find(':');
      const auto range =
          colon == std::string_view::npos ? std::nullopt : ParseFrameRange(entry.substr(0, colon));
      std::string_view values =
          colon == std::string_view::npos ? std::string_view{} : entry.substr(colon + 1);
      const std::size_t first_comma = values.find(',');
      const auto x = first_comma == std::string_view::npos ?
                         std::nullopt :
                         ParseNumber<double>(values.substr(0, first_comma));
      if (first_comma != std::string_view::npos)
        values.remove_prefix(first_comma + 1);
      const std::size_t second_comma = values.find(',');
      const auto y = ParseNumber<double>(values.substr(0, second_comma));
      std::optional<bool> visible = true;
      if (second_comma != std::string_view::npos)
        visible = ParseBool(values.substr(second_comma + 1));

      if (range && x && y && visible)
        script->pointer_spans.push_back(PointerSpan{*range, *x, *y, *visible});
      else
        ++invalid_entries;
    }

    if (separator == std::string_view::npos)
      break;
    text.remove_prefix(separator + 1);
  }

  if (invalid_entries != 0)
  {
    WARN_LOG_FMT(WIIMOTE, "Ignored {} malformed frame-scripted Wii Remote pointer span(s).",
                 invalid_entries);
  }
}

double ReadPositiveDimension(std::string_view name, double fallback)
{
  const auto text = ReadEnvironment(name);
  if (!text)
    return fallback;
  const auto value = ParseNumber<double>(*text);
  if (!value || !std::isfinite(*value) || *value <= 0.0)
  {
    WARN_LOG_FMT(WIIMOTE, "Ignoring invalid {} value '{}'.", name, *text);
    return fallback;
  }
  return *value;
}

std::optional<u16> ButtonMaskForControl(std::string_view group_name, std::string_view control_name)
{
  if (group_name == Wiimote::BUTTONS_GROUP)
  {
    if (control_name == Wiimote::A_BUTTON)
      return Wiimote::BUTTON_A;
    if (control_name == Wiimote::B_BUTTON)
      return Wiimote::BUTTON_B;
    if (control_name == Wiimote::ONE_BUTTON)
      return Wiimote::BUTTON_ONE;
    if (control_name == Wiimote::TWO_BUTTON)
      return Wiimote::BUTTON_TWO;
    if (control_name == Wiimote::MINUS_BUTTON)
      return Wiimote::BUTTON_MINUS;
    if (control_name == Wiimote::PLUS_BUTTON)
      return Wiimote::BUTTON_PLUS;
    if (control_name == Wiimote::HOME_BUTTON)
      return Wiimote::BUTTON_HOME;
  }
  else if (group_name == Wiimote::DPAD_GROUP)
  {
    if (control_name == DIRECTION_UP)
      return Wiimote::PAD_UP;
    if (control_name == DIRECTION_DOWN)
      return Wiimote::PAD_DOWN;
    if (control_name == DIRECTION_LEFT)
      return Wiimote::PAD_LEFT;
    if (control_name == DIRECTION_RIGHT)
      return Wiimote::PAD_RIGHT;
  }
  return std::nullopt;
}

std::optional<u16> NunchukButtonMaskForControl(std::string_view group_name,
                                               std::string_view control_name)
{
  if (group_name != Nunchuk::BUTTONS_GROUP)
    return std::nullopt;
  if (control_name == Nunchuk::C_BUTTON)
    return Nunchuk::BUTTON_C;
  if (control_name == Nunchuk::Z_BUTTON)
    return Nunchuk::BUTTON_Z;
  return std::nullopt;
}
}  // namespace

ControllerEmu::InputOverrideFunction CreateScriptedInputOverride(unsigned int wiimote_index)
{
  // The parity contract currently describes Wii Remote 1. Leave every other emulated controller
  // completely untouched.
  if (wiimote_index != 0)
    return {};

  const auto button_text = ReadEnvironment(BUTTON_SCRIPT_ENV);
  const auto pointer_text = ReadEnvironment(POINTER_SCRIPT_ENV);
  if (!button_text && !pointer_text)
    return {};

  auto script = std::make_shared<ScriptedInput>();
  if (button_text)
  {
    ParseButtonScript(*button_text, &script->button_spans, ParseButton, "Wii Remote");
  }
  if (pointer_text)
    ParsePointerScript(*pointer_text, script.get());
  script->pointer_width = ReadPositiveDimension(POINTER_WIDTH_ENV, DEFAULT_POINTER_WIDTH);
  script->pointer_height = ReadPositiveDimension(POINTER_HEIGHT_ENV, DEFAULT_POINTER_HEIGHT);

  if (script->button_spans.empty() && script->pointer_spans.empty())
    return {};

  s_current_frame.store(0, std::memory_order_relaxed);

  INFO_LOG_FMT(WIIMOTE,
               "Enabled frame-scripted input for Wii Remote 1 ({} button span(s), {} pointer "
               "span(s), pointer viewport {}x{}).",
               script->button_spans.size(), script->pointer_spans.size(), script->pointer_width,
               script->pointer_height);

  return [script = std::move(script)](std::string_view group_name, std::string_view control_name,
                                      ControlState state) -> std::optional<ControlState> {
    const u64 frame = s_current_frame.load(std::memory_order_relaxed);

    if (const auto control_mask = ButtonMaskForControl(group_name, control_name))
    {
      u16 active_mask = 0;
      for (const ButtonSpan& span : script->button_spans)
      {
        if (IsActive(span.range, frame))
          active_mask |= span.mask;
      }
      return (active_mask & *control_mask) != 0 ? 1.0 : state;
    }

    if (group_name != Wiimote::IR_GROUP ||
        (control_name != ControllerEmu::ReshapableInput::X_INPUT_OVERRIDE &&
         control_name != ControllerEmu::ReshapableInput::Y_INPUT_OVERRIDE))
    {
      return std::nullopt;
    }

    const PointerSpan* active_pointer = nullptr;
    for (const PointerSpan& span : script->pointer_spans)
    {
      if (IsActive(span.range, frame))
        active_pointer = &span;
    }
    if (active_pointer == nullptr)
      return state;
    if (!active_pointer->visible)
    {
      return control_name == ControllerEmu::ReshapableInput::X_INPUT_OVERRIDE ?
                 std::numeric_limits<ControlState>::quiet_NaN() :
                 0.0;
    }

    if (control_name == ControllerEmu::ReshapableInput::X_INPUT_OVERRIDE)
      return std::clamp(2.0 * active_pointer->x / script->pointer_width - 1.0, -1.0, 1.0);
    return std::clamp(1.0 - 2.0 * active_pointer->y / script->pointer_height, -1.0, 1.0);
  };
}

ControllerEmu::InputOverrideFunction CreateScriptedNunchukInputOverride(unsigned int wiimote_index)
{
  if (wiimote_index != 0)
    return {};

  const auto stick_text = ReadEnvironment(NUNCHUK_STICK_SCRIPT_ENV);
  const auto button_text = ReadEnvironment(NUNCHUK_BUTTON_SCRIPT_ENV);
  if (!stick_text && !button_text)
    return {};

  auto script = std::make_shared<ScriptedNunchukInput>();
  if (stick_text)
    ParseStickScript(*stick_text, script.get());
  if (button_text)
  {
    ParseButtonScript(*button_text, &script->button_spans, ParseNunchukButton, "Nunchuk");
  }

  if (script->stick_spans.empty() && script->button_spans.empty())
    return {};

  s_current_frame.store(0, std::memory_order_relaxed);

  INFO_LOG_FMT(WIIMOTE,
               "Enabled frame-scripted input for Wii Remote 1 Nunchuk ({} stick span(s), {} "
               "button span(s)).",
               script->stick_spans.size(), script->button_spans.size());

  return [script = std::move(script)](std::string_view group_name, std::string_view control_name,
                                      ControlState state) -> std::optional<ControlState> {
    const u64 frame = s_current_frame.load(std::memory_order_relaxed);

    if (const auto control_mask = NunchukButtonMaskForControl(group_name, control_name))
    {
      u16 active_mask = 0;
      for (const ButtonSpan& span : script->button_spans)
      {
        if (IsActive(span.range, frame))
          active_mask |= span.mask;
      }
      return (active_mask & *control_mask) != 0 ? 1.0 : state;
    }

    if (group_name != Nunchuk::STICK_GROUP ||
        (control_name != ControllerEmu::ReshapableInput::X_INPUT_OVERRIDE &&
         control_name != ControllerEmu::ReshapableInput::Y_INPUT_OVERRIDE))
    {
      return std::nullopt;
    }

    const StickSpan* active_stick = nullptr;
    for (const StickSpan& span : script->stick_spans)
    {
      if (IsActive(span.range, frame))
        active_stick = &span;
    }
    if (active_stick == nullptr)
      return state;

    return control_name == ControllerEmu::ReshapableInput::X_INPUT_OVERRIDE ? active_stick->x :
                                                                              active_stick->y;
  };
}

void SetScriptedInputFrame(u64 frame)
{
  s_current_frame.store(frame, std::memory_order_relaxed);
}
}  // namespace WiimoteEmu
