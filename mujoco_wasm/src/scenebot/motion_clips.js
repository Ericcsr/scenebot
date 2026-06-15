// Loader + zero-copy view layer for motion clips and contact labels packaged by
// tools/build_browser_assets.py.
//
// Wire format
// -----------
// `clips.bin` is a single Float32 blob. `clips_index.json` maps clip_idx → field offsets:
//   { joint_pos_off, joint_vel_off, body_pos_w_off, body_quat_w_off, n_frames, joint_dim, body_count }
// All offsets are in *floats* (i.e. byte offset = off * 4).
//
// `contact_labels.bin` is the same shape: index → { offset, n_frames, dim, present }.
// Missing labels => present:false; consumer falls back to default_contact_label.

export async function fetchBundle(binUrl, indexUrl) {
  const [bin, idxResp] = await Promise.all([
    fetch(binUrl).then((r) => {
      if (!r.ok) throw new Error(`fetch ${binUrl} ${r.status}`);
      return r.arrayBuffer();
    }),
    fetch(indexUrl).then((r) => {
      if (!r.ok) throw new Error(`fetch ${indexUrl} ${r.status}`);
      return r.json();
    }),
  ]);
  return { bin, index: idxResp };
}

export class Clip {
  /**
   * @param {Float32Array} floats   view over the whole clips.bin
   * @param {object} entry          one entry from clips_index.json
   */
  constructor(floats, entry) {
    this.name = entry.name;
    this.nFrames = entry.n_frames;
    this.jointDim = entry.joint_dim;
    this.bodyCount = entry.body_count;
    // Subarray views into the shared buffer; no copy.
    this.jointPos = floats.subarray(entry.joint_pos_off, entry.joint_pos_off + this.nFrames * this.jointDim);
    this.jointVel = floats.subarray(entry.joint_vel_off, entry.joint_vel_off + this.nFrames * this.jointDim);
    this.bodyPosW = floats.subarray(entry.body_pos_w_off, entry.body_pos_w_off + this.nFrames * this.bodyCount * 3);
    this.bodyQuatW = floats.subarray(entry.body_quat_w_off, entry.body_quat_w_off + this.nFrames * this.bodyCount * 4);
  }

  /** Return joint_pos[frame_idx] as a (jointDim,) view. Frame idx is clamped. */
  jointPosAt(frameIdx) {
    const i = clampFrame(frameIdx, this.nFrames);
    return this.jointPos.subarray(i * this.jointDim, (i + 1) * this.jointDim);
  }

  jointVelAt(frameIdx) {
    const i = clampFrame(frameIdx, this.nFrames);
    return this.jointVel.subarray(i * this.jointDim, (i + 1) * this.jointDim);
  }

  /** Return body_pos_w[frame_idx, body] as a (bodyCount, 3) flat view. */
  bodyPosAt(frameIdx) {
    const i = clampFrame(frameIdx, this.nFrames);
    const stride = this.bodyCount * 3;
    return this.bodyPosW.subarray(i * stride, (i + 1) * stride);
  }

  bodyQuatAt(frameIdx) {
    const i = clampFrame(frameIdx, this.nFrames);
    const stride = this.bodyCount * 4;
    return this.bodyQuatW.subarray(i * stride, (i + 1) * stride);
  }
}

export class ClipBundle {
  constructor(bin, index) {
    this._floats = new Float32Array(bin);
    this._index = index;
    this._cache = new Map();
    this.numClips = Object.keys(index).length;
  }

  static async load(binUrl, indexUrl) {
    const { bin, index } = await fetchBundle(binUrl, indexUrl);
    return new ClipBundle(bin, index);
  }

  clip(clipIdx) {
    const k = String(clipIdx);
    let c = this._cache.get(k);
    if (!c) {
      const entry = this._index[k];
      if (!entry) throw new Error(`clip_idx ${clipIdx} not in bundle (have ${this.numClips} clips)`);
      c = new Clip(this._floats, entry);
      this._cache.set(k, c);
    }
    return c;
  }
}

export class ContactLabels {
  constructor(bin, index) {
    this._floats = new Float32Array(bin);
    this._index = index;
  }

  static async load(binUrl, indexUrl) {
    const { bin, index } = await fetchBundle(binUrl, indexUrl);
    return new ContactLabels(bin, index);
  }

  /** True iff this clip has labels packaged. */
  hasClip(clipIdx) {
    const e = this._index[String(clipIdx)];
    return !!(e && e.present);
  }

  /** Return contact_mask[frame_idx] for clip as a (dim,) view, or null if no labels. */
  atFrame(clipIdx, frameIdx) {
    const e = this._index[String(clipIdx)];
    if (!e || !e.present) return null;
    const i = clampFrame(frameIdx, e.n_frames);
    return this._floats.subarray(e.offset + i * e.dim, e.offset + (i + 1) * e.dim);
  }

  dim(clipIdx) {
    const e = this._index[String(clipIdx)];
    return e?.dim ?? 0;
  }

  /** Max channel width across packaged clip labels (mirrors ContactLabelResolver.contact_dim). */
  maxStreamDim() {
    let maxDim = 4;
    for (const e of Object.values(this._index)) {
      if (e?.present && Number(e.dim) > maxDim) maxDim = Number(e.dim);
    }
    return maxDim;
  }
}

function clampFrame(frameIdx, nFrames) {
  if (!Number.isFinite(frameIdx)) return 0;
  const i = Math.floor(frameIdx);
  if (i < 0) return 0;
  if (i >= nFrames) return Math.max(0, nFrames - 1);
  return i;
}
