import { describe, it, expect } from "vitest";
import {
  expand5wayContactTo10,
  padContactMaskToStreamDim,
  normalizeContactForPolicy,
} from "../src/scenebot/contact_utils.js";

const META_10WAY = {
  contact_dim: 10,
  use_10way_contact: true,
  use_5dim_contact_from_4dim: false,
  use_8way_contact: false,
  default_contact_label: [0, 0, 0, 0, 0],
};

describe("contact_utils", () => {
  it("pads 4-way clip labels to 5-way stream width", () => {
    expect(Array.from(padContactMaskToStreamDim([1, 0, 1, 0], 5))).toEqual([1, 0, 1, 0, 0]);
  });

  it("expands 5-way stream labels to 10-way policy obs", () => {
    // 4-way -> 8-way: feet -> env [0,2], wrists -> obj [5,7] per rl_policy.py.
    expect(Array.from(expand5wayContactTo10([1, 1, 1, 1, 0]))).toEqual([
      1, 0, 1, 0, 0, 1, 0, 1, 0, 0,
    ]);
    expect(Array.from(expand5wayContactTo10([1, 1, 1, 1, 1]))).toEqual([
      1, 0, 1, 0, 0, 1, 0, 1, 1, 0,
    ]);
  });

  it("normalizes sit/pickup stream masks for use_10way_contact policy", () => {
    const sit = normalizeContactForPolicy([1, 1, 1, 1, 0], META_10WAY);
    expect(Array.from(sit)).toEqual([1, 0, 1, 0, 0, 1, 0, 1, 0, 0]);

    const stair4 = normalizeContactForPolicy(
      padContactMaskToStreamDim([1, 0, 1, 0], 5),
      META_10WAY,
    );
    expect(Array.from(stair4)).toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
  });
});
