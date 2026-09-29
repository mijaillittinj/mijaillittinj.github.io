/* Instantaneous efficiency of three solar thermal collector technologies,
 * standard curve of EN 12975 / ISO 9806:
 *   eta = eta0 - a1 (Tm - Ta)/G - a2 (Tm - Ta)^2/G,  clipped at 0 (stagnation)
 *   Pu  = eta G Ac
 * Same parameters as tools/prepare_data.py and scenes/s51_technologies.py.
 * Classic browser script (window.EtaCollector) and CommonJS module for tests.
 */
(function (root) {
  var TECH = [
    { key: 'unglazed', name: 'Sin cubierta', eta0: 0.90, a1: 17.0, a2: 0.000 },
    { key: 'flat', name: 'Plano selectivo', eta0: 0.80, a1: 3.50, a2: 0.015 },
    { key: 'tubes', name: 'Tubos de vacío', eta0: 0.75, a1: 1.50, a2: 0.008 }
  ];
  var AC = 2.0; // m2
  function eta(t, dT, G) {
    var e = t.eta0 - t.a1 * dT / G - t.a2 * dT * dT / G;
    return e > 0 ? e : 0;
  }
  function evaluate(G, Ta, Tm) {
    var dT = Tm - Ta;
    return TECH.map(function (t) {
      var e = eta(t, dT, G);
      return { key: t.key, name: t.name, eta: e, Pu: e * G * AC };
    });
  }
  // temperature difference where two technologies have equal efficiency (bisection)
  function crossover(a, b, G) {
    var lo = 0.1, hi = 200, f = function (d) { return eta(a, d, G) - eta(b, d, G); };
    for (var i = 0; i < 200; i++) {
      var mid = 0.5 * (lo + hi);
      if (f(lo) * f(mid) <= 0) hi = mid; else lo = mid;
    }
    return 0.5 * (lo + hi);
  }
  var api = { TECH: TECH, AC: AC, eta: eta, evaluate: evaluate, crossover: crossover };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.EtaCollector = api;
})(this);
