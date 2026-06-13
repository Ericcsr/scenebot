// Quaternion + small SE(2) helpers for the browser port of the scenebot stack.
//
// Convention: scipy.spatial.transform.Rotation uses (x,y,z,w); MuJoCo and most of the
// motion-graph code uses (w,x,y,z). The Python originals juggle both via index permutations.
// This module isolates the conversions and provides Rotation-like operations directly on
// xyzw-ordered Float32Array(4) / Float64Array(4) buffers.
//
// All functions accept length-4 arrays for quaternions (xyzw unless explicitly *Wxyz) and
// length-3 arrays for vectors. Output arrays are allocated unless an `out` is supplied.

const _v3 = () => new Float64Array(3);
const _q4 = () => new Float64Array(4);


export function wxyzToXyzw(qWxyz, out = _q4()) {
  out[0] = qWxyz[1];
  out[1] = qWxyz[2];
  out[2] = qWxyz[3];
  out[3] = qWxyz[0];
  return out;
}

export function xyzwToWxyz(qXyzw, out = _q4()) {
  out[0] = qXyzw[3];
  out[1] = qXyzw[0];
  out[2] = qXyzw[1];
  out[3] = qXyzw[2];
  return out;
}

/** Hamilton product: out = a * b, both xyzw. Matches scipy Rotation.__mul__,
 * which silently normalizes inputs before composing. */
export function multQuat(a, b, out = _q4()) {
  let ax = a[0], ay = a[1], az = a[2], aw = a[3];
  let bx = b[0], by = b[1], bz = b[2], bw = b[3];
  // Normalize each input so we match scipy.spatial.transform.Rotation behaviour
  // (Rotation.from_quat re-normalizes; downstream math assumes unit quats).
  let na = Math.sqrt(ax * ax + ay * ay + az * az + aw * aw);
  if (na > 0) { ax /= na; ay /= na; az /= na; aw /= na; }
  let nb = Math.sqrt(bx * bx + by * by + bz * bz + bw * bw);
  if (nb > 0) { bx /= nb; by /= nb; bz /= nb; bw /= nb; }
  out[0] = aw * bx + ax * bw + ay * bz - az * by;
  out[1] = aw * by - ax * bz + ay * bw + az * bx;
  out[2] = aw * bz + ax * by - ay * bx + az * bw;
  out[3] = aw * bw - ax * bx - ay * by - az * bz;
  return out;
}

/** q^-1 for unit quaternions (just conjugate; xyzw in/out). */
export function invQuat(q, out = _q4()) {
  out[0] = -q[0];
  out[1] = -q[1];
  out[2] = -q[2];
  out[3] = q[3];
  return out;
}

/** Rotate vector v (length 3) by quaternion q (xyzw). Matches scipy Rotation.from_quat(q).apply(v). */
export function applyQuat(qXyzw, v, out = _v3()) {
  // Standard formula: v' = v + 2 q.xyz × (q.xyz × v + q.w v)
  const qx = qXyzw[0], qy = qXyzw[1], qz = qXyzw[2], qw = qXyzw[3];
  const vx = v[0], vy = v[1], vz = v[2];
  // t = 2 * (q.xyz × v)
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  // v' = v + qw * t + q.xyz × t
  out[0] = vx + qw * tx + (qy * tz - qz * ty);
  out[1] = vy + qw * ty + (qz * tx - qx * tz);
  out[2] = vz + qw * tz + (qx * ty - qy * tx);
  return out;
}

/** Quat (xyzw) representing a rotation of `angle` rad about world Z. Matches Rotation.from_euler("z", a). */
export function quatFromYaw(angle, out = _q4()) {
  const half = angle * 0.5;
  out[0] = 0;
  out[1] = 0;
  out[2] = Math.sin(half);
  out[3] = Math.cos(half);
  return out;
}

/** yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)). q is wxyz. Mirrors stitch_core.yaw_from_quat_wxyz. */
export function headingFromQuatWxyz(qWxyz) {
  const qw = qWxyz[0], qx = qWxyz[1], qy = qWxyz[2], qz = qWxyz[3];
  return Math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
}

/** Same as above but xyzw input. */
export function headingFromQuatXyzw(qXyzw) {
  const qx = qXyzw[0], qy = qXyzw[1], qz = qXyzw[2], qw = qXyzw[3];
  return Math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
}

/**
 * Build the 3x3 rotation matrix (row-major) for q (xyzw). Matches Rotation.as_matrix().
 * `out` should be a length-9 array.
 */
export function quatAsMatrix(qXyzw, out = new Float64Array(9)) {
  const x = qXyzw[0], y = qXyzw[1], z = qXyzw[2], w = qXyzw[3];
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;
  out[0] = 1 - 2 * (yy + zz);  out[1] = 2 * (xy - wz);     out[2] = 2 * (xz + wy);
  out[3] = 2 * (xy + wz);      out[4] = 1 - 2 * (xx + zz); out[5] = 2 * (yz - wx);
  out[6] = 2 * (xz - wy);      out[7] = 2 * (yz + wx);     out[8] = 1 - 2 * (xx + yy);
  return out;
}

/**
 * Mirror of stitch_core.compute_se2_align_root: computes (yaw rotation, xyz translation)
 * that maps anchor_from -> anchor_to in world. r_delta is a quat (xyzw) representing yaw only.
 */
export function computeSe2AlignRoot(anchorPosFrom, anchorQuatFromWxyz, anchorPosTo, anchorQuatToWxyz) {
  const yawFrom = headingFromQuatWxyz(anchorQuatFromWxyz);
  const yawTo = headingFromQuatWxyz(anchorQuatToWxyz);
  const rDelta = quatFromYaw(yawTo - yawFrom);
  // t = p_to - r_delta.apply(p_from)
  const rotated = applyQuat(rDelta, anchorPosFrom);
  const t = new Float64Array(3);
  t[0] = anchorPosTo[0] - rotated[0];
  t[1] = anchorPosTo[1] - rotated[1];
  t[2] = anchorPosTo[2] - rotated[2];
  return { rDelta, t };
}

/**
 * Apply (rDelta, t) to a body position (length 3) and orientation (wxyz, length 4).
 * Mirrors stitch_core.apply_se2_rigid_to_bodies for a single body.
 * `posOut` and `quatOut` (length 3 / 4) are mutated in place; same arrays may be passed as input.
 */
export function applySe2ToBody(rDelta, t, posIn, quatInWxyz, posOut, quatOut) {
  // pos_out = r_delta @ pos_in + t
  const rp = applyQuat(rDelta, posIn);
  posOut[0] = rp[0] + t[0];
  posOut[1] = rp[1] + t[1];
  posOut[2] = rp[2] + t[2];
  // quat_out_wxyz = (r_delta * Rotation.from_quat(wxyz_to_xyzw(quat_in))).as_quat() then to wxyz
  const tmpXyzw = wxyzToXyzw(quatInWxyz);
  const rotatedXyzw = multQuat(rDelta, tmpXyzw);
  xyzwToWxyz(rotatedXyzw, quatOut);
}
