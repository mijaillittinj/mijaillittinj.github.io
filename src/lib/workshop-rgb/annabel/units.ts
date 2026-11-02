// Units of the stage-1 emission. ANNAbel works with lengths in mm (r, z, dr, the operator A),
// so with b in W m⁻² sr⁻¹ its J comes out in W m⁻² sr⁻¹ mm⁻¹ (core/units.py j_label). The
// tool reports j_j(r, z) in W m⁻³ sr⁻¹ (§5 of the brief), 1 mm⁻¹ = 1000 m⁻¹.

export const PER_MM_TO_PER_M = 1e3;

/** J per mm (W m⁻² sr⁻¹ mm⁻¹) → j in W m⁻³ sr⁻¹. Returns a new array. */
export function emissionPerMmToSI(jPerMm: ArrayLike<number>): Float32Array {
  const out = new Float32Array(jPerMm.length);
  for (let q = 0; q < jPerMm.length; q++) out[q] = jPerMm[q] * PER_MM_TO_PER_M;
  return out;
}

/** j in W m⁻³ sr⁻¹ → J per mm, the inverse. */
export function emissionSIToPerMm(jSI: ArrayLike<number>): Float32Array {
  const out = new Float32Array(jSI.length);
  for (let q = 0; q < jSI.length; q++) out[q] = jSI[q] / PER_MM_TO_PER_M;
  return out;
}
