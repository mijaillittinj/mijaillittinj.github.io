/* View factor from a small element to a coaxial facing disk, and the fire-break
 * estimate of the lecture. Same equations and constants as tools/prepare_data.py
 * and scenes/s60_disk_fire.py. Works as a classic browser script (window.FFDisk)
 * and as a CommonJS module for tests/test_widget.js.
 */
(function (root) {
  var SIGMA = 5.67e-8; // W/m^2 K^4, value used in the lecture
  function radius(areaM2) { return Math.sqrt(areaM2 / Math.PI); }
  // exact, from integrating cos1 cos2 dA2 / (pi r^2) over rings of the disk
  function fExact(R, L) { return (R * R) / (R * R + L * L); }
  // point-source approximation A/(pi L^2) = R^2/L^2, valid for L >> R
  function fPoint(R, L) { return (R * R) / (L * L); }
  function emissive(TK) { return SIGMA * Math.pow(TK, 4); }
  // incident flux on the target element, q = E_b F_{dA -> flame} (reciprocity)
  function fluxes(areaM2, TK, L) {
    var R = radius(areaM2), Eb = emissive(TK);
    var fe = fExact(R, L), fp = fPoint(R, L);
    return { R: R, Eb: Eb, fExact: fe, fPoint: fp,
             qExact: Eb * fe, qPoint: Eb * fp, overPct: 100 * (fp - fe) / fe };
  }
  var api = { SIGMA: SIGMA, radius: radius, fExact: fExact, fPoint: fPoint,
              emissive: emissive, fluxes: fluxes };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.FFDisk = api;
})(this);
