import { describe, it, expect } from "vitest";
import {
  COMMAND_PICK_UP_BOX,
  COMMAND_STOP,
  isPickupForwardCompleted,
  atSegmentEnd,
} from "../src/scenebot/motion_graph_runtime.js";
import { KeyboardCommandState } from "../src/scenebot/keyboard_state.js";

describe("pickup auto-freeze", () => {
  it("detects the final forward PickUpBox publish tick", () => {
    const done = {
      command: COMMAND_PICK_UP_BOX,
      playback_direction: 1,
      frame_idx: 120,
      segment_end_frame: 120,
    };
    expect(atSegmentEnd(done)).toBe(true);
    expect(isPickupForwardCompleted(done)).toBe(true);

    const mid = { ...done, frame_idx: 50 };
    expect(isPickupForwardCompleted(mid)).toBe(false);

    const reverse = { ...done, playback_direction: -1, frame_idx: 0, segment_end_frame: 0 };
    expect(isPickupForwardCompleted(reverse)).toBe(false);

    const stop = { ...done, command: COMMAND_STOP };
    expect(isPickupForwardCompleted(stop)).toBe(false);
  });

  it("activateUpperBodyFreeze enables freeze without F", () => {
    const kb = new KeyboardCommandState();
    expect(kb.upperBodyFreezeEnabled()).toBe(false);
    kb.activateUpperBodyFreeze();
    expect(kb.upperBodyFreezeEnabled()).toBe(true);
    expect(kb.pollUpperBodyFreezeToggle()).toBe(null);
  });
});
