window.MathJax = {
  loader: {
    // Preload the boldsymbol extension explicitly. Relying on MathJax's
    // autoload races with mkdocs-material instant navigation (typesetClear
    // runs before the dynamic fetch resolves), leaving \boldsymbol
    // rendered as a red "undefined control sequence" error.
    load: ["[tex]/boldsymbol"],
  },
  tex: {
    packages: { "[+]": ["boldsymbol"] },
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
