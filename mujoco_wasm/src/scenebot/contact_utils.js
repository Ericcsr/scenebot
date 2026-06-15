// Mirrors tml_humanoid_deploy/rl_policy.py contact-mask expansion helpers.

/** 4-way [Lfoot, Rfoot, Lwrist, Rwrist] -> 8-way env/obj split. */
export function expand4wayContactTo8(m4) {
  const src = m4 instanceof Float32Array ? m4 : Float32Array.from(m4);
  if (src.length !== 4) {
    throw new Error(`expand4wayContactTo8: expected 4 channels, got ${src.length}`);
  }
  const out = new Float32Array(8);
  out[0] = src[0];
  out[2] = src[1];
  out[5] = src[2];
  out[7] = src[3];
  return out;
}

/** 5-way stored labels -> 10-way policy obs. */
export function expand5wayContactTo10(m5) {
  const src = m5 instanceof Float32Array ? m5 : Float32Array.from(m5);
  if (src.length !== 5) {
    throw new Error(`expand5wayContactTo10: expected 5 channels, got ${src.length}`);
  }
  const m8 = expand4wayContactTo8(src.subarray(0, 4));
  const out = new Float32Array(10);
  out.set(m8, 0);
  out[8] = src[4];
  out[9] = 0.0;
  return out;
}

/** 4-way limbs -> 10-way with zero pelvis/seat channels. */
export function expand4wayLimbTo10wayWithZeroSeat(m4) {
  const m8 = expand4wayContactTo8(m4);
  const out = new Float32Array(10);
  out.set(m8, 0);
  return out;
}

/** 8-way -> 4-way by max(env, obj) per limb. */
export function reduce8wayContactTo4(m8) {
  const src = m8 instanceof Float32Array ? m8 : Float32Array.from(m8);
  if (src.length !== 8) {
    throw new Error(`reduce8wayContactTo4: expected 8 channels, got ${src.length}`);
  }
  return Float32Array.from([
    Math.max(src[0], src[1]),
    Math.max(src[2], src[3]),
    Math.max(src[4], src[5]),
    Math.max(src[6], src[7]),
  ]);
}

/** Append zero pelvis/seat column: 4 -> 5. */
export function pad4wayContactTo5(m4) {
  const src = m4 instanceof Float32Array ? m4 : Float32Array.from(m4);
  if (src.length !== 4) {
    throw new Error(`pad4wayContactTo5: expected 4 channels, got ${src.length}`);
  }
  const out = new Float32Array(5);
  out.set(src, 0);
  return out;
}

/**
 * Pad/truncate a motion-graph stream mask to the resolver's stream width (max label dim, usually 5).
 * Mirrors ContactLabelResolver padding 4-dim clip labels up to contact_dim.
 */
export function padContactMaskToStreamDim(raw, streamDim) {
  const src = raw instanceof Float32Array ? raw : Float32Array.from(raw);
  const out = new Float32Array(streamDim);
  const n = Math.min(src.length, streamDim);
  for (let i = 0; i < n; i++) out[i] = src[i];
  return out;
}

/**
 * Expand a motion-graph contact_mask payload to policy.contact_dim.
 * Mirrors RLStreamingContactPolicy stream decode (rl_policy.py:1352-1396) and default-label init.
 */
export function normalizeContactForPolicy(raw, meta, { limbContactFileIs8way = false } = {}) {
  const contactDim = meta.contact_dim;
  let arr = raw instanceof Float32Array ? Float32Array.from(raw) : Float32Array.from(raw);
  const use10 = !!meta.use_10way_contact;
  const use5from4 = !!meta.use_5dim_contact_from_4dim;
  const use8 = !!meta.use_8way_contact;

  if (use10 && use5from4) {
    const n = arr.length;
    if (n === 4) {
      arr = expand4wayLimbTo10wayWithZeroSeat(arr);
    } else if (n === 5) {
      arr = expand4wayLimbTo10wayWithZeroSeat(arr.subarray(0, 4));
    } else if (n === 8) {
      arr = limbContactFileIs8way
        ? (() => { const o = new Float32Array(10); o.set(arr, 0); return o; })()
        : expand4wayLimbTo10wayWithZeroSeat(reduce8wayContactTo4(arr));
    } else if (n === 10) {
      arr = (() => { const o = new Float32Array(10); o.set(arr.subarray(0, 8), 0); return o; })();
    } else {
      throw new Error(
        `contact stream: use_10way + use_5dim_from_4dim expected 4,5,8,10 values, got ${n}`,
      );
    }
  } else if (use10) {
    if (arr.length === 5) arr = expand5wayContactTo10(arr);
    if (arr.length !== contactDim) {
      throw new Error(`contact stream: use_10way expected 5 or ${contactDim} values, got ${arr.length}`);
    }
  } else if (use5from4) {
    if (arr.length === 4) arr = pad4wayContactTo5(arr);
    else if (arr.length === 5) {
      arr = Float32Array.from(arr);
      arr[4] = 0.0;
    }
    if (arr.length !== contactDim) {
      throw new Error(`contact stream: use_5dim_from_4dim expected 4 or ${contactDim} values, got ${arr.length}`);
    }
  } else if (use8) {
    if (arr.length === 4) arr = expand4wayContactTo8(arr);
    if (arr.length !== contactDim) {
      throw new Error(`contact stream: use_8way expected 4 or ${contactDim} values, got ${arr.length}`);
    }
  } else if (arr.length !== contactDim) {
    throw new Error(`contact stream: expected ${contactDim} values, got ${arr.length}`);
  }

  if (arr.length !== contactDim) {
    throw new Error(`contact stream: expected ${contactDim} values after normalize, got ${arr.length}`);
  }
  return arr;
}
