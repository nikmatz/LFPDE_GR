/* LFPDE_GR — κοινά σενάρια: ρυθμίσεις MathJax (ίδιες μακροεντολές με το βιβλίο) και κουμπί αντιγραφής. */
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['\\[', '\\]'], ['$$', '$$']],
    macros: {
      R: '\\mathbb{R}', C: '\\mathbb{C}', N: '\\mathbb{N}', Z: '\\mathbb{Z}', Q: '\\mathbb{Q}',
      Re: '\\operatorname{Re}', Im: '\\operatorname{Im}',
      Lap: '\\mathcal{L}', Lapinv: '\\mathcal{L}^{-1}', Fou: '\\mathcal{F}', Fouinv: '\\mathcal{F}^{-1}',
      LT: ['\\mathcal{L}\\{#1\\}', 1], ILT: ['\\mathcal{L}^{-1}\\{#1\\}', 1],
      sinc: '\\operatorname{sinc}', sgn: '\\operatorname{sgn}', erf: '\\operatorname{erf}',
      dd: '\\mathrm{d}', abs: ['\\left|#1\\right|', 1], conj: ['\\overline{#1}', 1],
      Si: '\\operatorname{Si}', erfc: '\\operatorname{erfc}', rect: '\\operatorname{rect}', tri: '\\operatorname{tri}',
      Fc: '\\mathcal{F}_c', Fs: '\\mathcal{F}_s', FT: ['\\mathcal{F}\\{#1\\}', 1], IFT: ['\\mathcal{F}^{-1}\\{#1\\}', 1],
      norm: ['\\left\\|#1\\right\\|', 1], conv: '\\ast', eqdef: '\\mathrel{\\mathop:}=',
      pd: ['\\frac{\\partial #1}{\\partial #2}', 2], pdd: ['\\frac{\\partial^2 #1}{\\partial #2^2}', 2],
      pdm: ['\\frac{\\partial^2 #1}{\\partial #2\\,\\partial #3}', 3], ud: 'u', eval: ['\\Big[#1\\Big]_{#2}^{#3}', 3]
    }
  },
  options: { ignoreHtmlClass: 'tex2jax_ignore' },
  chtml: { scale: 1.0 }
};
function mxCopy(id, btn) {
  var el = document.getElementById(id);
  var txt = el ? el.textContent : '';
  var done = function () { if (btn) { var o = btn.textContent; btn.textContent = '✓ Αντιγράφηκε'; setTimeout(function () { btn.textContent = o; }, 1800); } };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(txt).then(done, function () { fallback(); });
  } else { fallback(); }
  function fallback() {
    var ta = document.createElement('textarea'); ta.value = txt; document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); done(); } catch (e) {} document.body.removeChild(ta);
  }
}
