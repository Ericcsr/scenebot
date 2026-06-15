// Keyboard control hints shown in on-screen control panels (full-browser mode).

/** Min width for control panels so hint text does not clip. */
export const SCENEBOT_PANEL_MIN_WIDTH = "420px";

export const KEYBOARD_CONTROLS_HINT_ROWS = [
  { keys: "WASD", action: "walk / turn" },
  { keys: "Q / E", action: "spin left / right" },
  { keys: "N / Z", action: "step on box / down" },
  { keys: "G / P", action: "pick up / put down" },
];

export const CONTROLS_USAGE_PREAMBLE_LINES = [
  "Press the next key when",
  "previous motion is finished.",
];
export const CONTROLS_EXAMPLE_SEQUENCE = "Q-L-L-E-W-N-G-W-Z-P";

function formatKeyBindingRows() {
  return KEYBOARD_CONTROLS_HINT_ROWS.map(({ keys, action }) => `${keys} — ${action}`).join("\n");
}

/** Multi-line block for on-screen control panels (speed bar, etc.). */
export function formatControlsHintPanel() {
  return [
    ...CONTROLS_USAGE_PREAMBLE_LINES,
    "Example sequence:",
    CONTROLS_EXAMPLE_SEQUENCE,
    "",
    formatKeyBindingRows(),
  ].join("\n");
}
