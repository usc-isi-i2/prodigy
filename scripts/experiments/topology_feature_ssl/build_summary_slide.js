// topology_feature_ssl — single SYNTHESIS slide.
// Q1: what did we learn as a whole?  Q2: how does it move us toward the best
// pretraining strategy for a retweet-graph foundation GNN?
// Same pptxgenjs palette/fonts as build_slides.js.
// Build:  NODE_PATH=<scratchpad>/node_modules node build_summary_slide.js
const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
p.author = "Philipp Eibl";
p.title = "topology_feature_ssl — what we learned";

const C = {
  bgDark: "0B3D2E", ink: "17251F", primary: "0F6E56",
  gLite: "A7DEC8", gMid: "1D9E75", gDark: "0B5A44",
  char: "33322F", muted: "6C7B74", panel: "EEF5F1",
  warnBg: "F7ECE4", warnInk: "A8481F", white: "FFFFFF", line: "D6E3DC",
  win: "E7F4EC", winInk: "0B5A44", axis: "8A9A93", grid: "E4ECE7",
};
const HFONT = "Cambria", BFONT = "Calibri";

const s = p.addSlide(); s.background = { color: C.white };

// ---------- header ----------
s.addText("PRETRAINING STRATEGY · topology_feature_ssl · what we learned".toUpperCase(),
  { x: 0.7, y: 0.36, w: 12, h: 0.32, fontFace: BFONT, fontSize: 12.5, bold: true, color: C.gMid, charSpacing: 2, margin: 0 });
s.addText([
  { text: "Structure and features ", options: { color: C.ink } },
  { text: "trade off", options: { color: C.primary } },
  { text: " — no single pretext learns both", options: { color: C.ink } },
], { x: 0.7, y: 0.66, w: 12.3, h: 0.62, fontFace: HFONT, fontSize: 29, bold: true, margin: 0 });
s.addText("Across all three levers (architecture · augmentation · objective), no frozen encoder is strong on BOTH the feature tasks (node classification + regression) AND the topological task (link prediction).",
  { x: 0.7, y: 1.28, w: 12.3, h: 0.42, fontFace: BFONT, fontSize: 13.5, italic: true, color: C.muted, margin: 0 });

// ============================================================
// LEFT — the capability plane (the proof of Q1)
// ============================================================
const plotL = 1.55, plotR = 7.30, plotT = 2.15, plotB = 6.05;
const XD0 = -0.16, XD1 = 0.16;     // regression rho
const YD0 = 0.20, YD1 = 0.80;      // link-prediction AUC
const mapX = v => plotL + (v - XD0) / (XD1 - XD0) * (plotR - plotL);
const mapY = v => plotB - (v - YD0) / (YD1 - YD0) * (plotB - plotT);

// plot panel
s.addShape(p.shapes.RECTANGLE, { x: plotL, y: plotT, w: plotR - plotL, h: plotB - plotT, fill: { color: C.white }, line: { color: C.line, width: 1 } });
// GOAL zone (reg > 0 AND auc > 0.7) — the empty corner
s.addShape(p.shapes.RECTANGLE, { x: mapX(0), y: plotT, w: plotR - mapX(0), h: mapY(0.70) - plotT, fill: { color: C.win, transparency: 25 }, line: { color: C.gMid, width: 1, dashType: "dash" } });
s.addText([{ text: "GOAL — strong on both\n", options: { bold: true, color: C.winInk } }, { text: "(empty)", options: { italic: true, color: C.winInk } }],
  { x: mapX(0) + 0.06, y: plotT + 0.10, w: plotR - mapX(0) - 0.12, h: 0.62, align: "center", fontFace: BFONT, fontSize: 11.5, lineSpacingMultiple: 0.95, margin: 0 });
// reference lines
s.addShape(p.shapes.LINE, { x: mapX(0), y: plotT, w: 0, h: plotB - plotT, line: { color: C.axis, width: 1, dashType: "dash" } });
s.addShape(p.shapes.LINE, { x: plotL, y: mapY(0.5), w: plotR - plotL, h: 0, line: { color: C.axis, width: 1, dashType: "dash" } });
s.addText("chance", { x: plotL + 0.14, y: mapY(0.5) - 0.24, w: 1.2, h: 0.2, fontFace: BFONT, fontSize: 9.5, italic: true, color: C.axis, margin: 0 });
s.addText("≤ raw features", { x: mapX(0) - 1.25, y: plotB - 0.26, w: 1.2, h: 0.2, align: "right", fontFace: BFONT, fontSize: 9.5, italic: true, color: C.axis, margin: 0 });

// axis labels (short + parallel; metric definitions in the caption below)
s.addText("Feature capability  →",
  { x: plotL, y: plotB + 0.10, w: plotR - plotL, h: 0.3, align: "center", fontFace: BFONT, fontSize: 13, bold: true, color: C.char, margin: 0 });
s.addText("Topological\ncapability  ↑",
  { x: 0.16, y: (plotT + plotB) / 2 - 0.36, w: 1.34, h: 0.72, align: "left", valign: "middle", fontFace: BFONT, fontSize: 13, bold: true, color: C.char, lineSpacingMultiple: 0.95, margin: 0 });
s.addText("feature = node-attribute regression ρ    ·    topological = link-prediction AUC    ·    chance = 0.5",
  { x: plotL - 0.35, y: plotB + 0.42, w: plotR - plotL + 0.7, h: 0.24, align: "center", fontFace: BFONT, fontSize: 10, italic: true, color: C.muted, margin: 0 });

// data points  [label, reg, auc, role]  role: feat|topo|obj|base
const PTS = [
  ["B0", -0.003, 0.675, "base"],
  ["B1", -0.122, 0.341, "base"],
  ["E1", 0.135, 0.657, "feat"],
  ["E2", -0.077, 0.761, "topo"],
  ["E2b", -0.001, 0.401, "base"],
  ["E4", -0.133, 0.662, "obj"],
  ["E4r", -0.124, 0.234, "obj"],
];
const COL = { feat: C.primary, topo: C.gMid, obj: C.warnInk, base: C.muted };
const R = { feat: 0.12, topo: 0.12, obj: 0.11, base: 0.085 };
// each point shows its regression ρ inline (the x-value, hard to read off position near 0)
const REGVAL = { B0: "0.00", B1: "−0.12", E1: "+0.14", E2: "−0.08", E2b: "0.00", E4: "−0.13", E4r: "−0.12" };
const NUDGE = { B0: [0.13, -0.11], B1: [0.13, -0.05], E2: [0.17, -0.11], E2b: [0.13, -0.06], E4: [0.15, -0.11], E4r: [0.13, -0.05] };
PTS.forEach(([lab, xr, yr, role]) => {
  const cx = mapX(xr), cy = mapY(yr), r = R[role];
  s.addShape(p.shapes.OVAL, { x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, fill: { color: COL[role] }, line: { color: C.white, width: 1.5 } });
  const strong = role === "feat" || role === "topo";
  const runs = [
    { text: lab, options: { bold: strong, color: COL[role] } },
    { text: "  ρ " + REGVAL[lab], options: { color: C.muted, fontSize: 10 } },
  ];
  if (lab === "E1") {  // near the right edge → label to the LEFT of the dot
    s.addText(runs, { x: cx - 1.74, y: cy - 0.11, w: 1.5, h: 0.22, align: "right", fontFace: BFONT, fontSize: 12, margin: 0 });
  } else {
    const [nx, ny] = NUDGE[lab];
    s.addText(runs, { x: cx + nx, y: cy + ny, w: 1.5, h: 0.22, align: "left", fontFace: BFONT, fontSize: 12, margin: 0 });
  }
});
// champion callouts (below the dots)
s.addText("features champ", { x: mapX(0.135) - 0.75, y: mapY(0.657) + 0.14, w: 1.5, h: 0.2, align: "center", fontFace: BFONT, fontSize: 9.5, italic: true, color: C.primary, margin: 0 });
s.addText("topology champ", { x: mapX(-0.077) - 0.5, y: mapY(0.761) + 0.14, w: 1.5, h: 0.2, align: "center", fontFace: BFONT, fontSize: 9.5, italic: true, color: C.gMid, margin: 0 });
// legend
s.addText([
  { text: "E1", options: { bold: true, color: C.primary } }, { text: " degree inputs    ", options: { color: C.muted } },
  { text: "E2", options: { bold: true, color: C.gMid } }, { text: " count-aware aggregation    ", options: { color: C.muted } },
  { text: "E4", options: { bold: true, color: C.warnInk } }, { text: " multi-task objective", options: { color: C.muted } },
], { x: plotL - 0.2, y: 1.80, w: plotR - plotL + 0.4, h: 0.24, align: "center", fontFace: BFONT, fontSize: 10.5, margin: 0 });

// ============================================================
// RIGHT — how this moves us toward the best strategy (Q2)
// ============================================================
const RX = 7.95, RW = 4.75;
s.addText("HOW THIS MOVES US TOWARD THE BEST STRATEGY",
  { x: RX, y: 1.72, w: RW, h: 0.3, fontFace: BFONT, fontSize: 12.5, bold: true, color: C.ink, charSpacing: 0.5, margin: 0 });

function card(y, h, barColor, head, headColor, bullets) {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: RX, y, w: RW, h, rectRadius: 0.06, fill: { color: C.panel }, line: { color: C.line, width: 1 } });
  s.addShape(p.shapes.RECTANGLE, { x: RX, y, w: 0.09, h, fill: { color: barColor }, line: { type: "none" } });
  s.addText(head, { x: RX + 0.22, y: y + 0.08, w: RW - 0.34, h: 0.28, fontFace: BFONT, fontSize: 12.5, bold: true, color: headColor, margin: 0 });
  s.addText(bullets, { x: RX + 0.22, y: y + 0.40, w: RW - 0.40, h: h - 0.46, fontFace: BFONT, fontSize: 12, color: C.ink, lineSpacingMultiple: 1.02, margin: 0,
    bullet: { indent: 12 } });
}
card(2.10, 1.42, C.primary, "✓  INGREDIENTS THAT WORK", C.winInk, [
  { text: "Count-aware aggregation (E2) → topology: link-pred 0.76, the best result", options: { bullet: { indent: 12 } } },
  { text: "Directed-degree inputs (E1) → features: the only arm with real regression", options: { bullet: { indent: 12 }, paraSpaceBefore: 4 } },
]);
card(3.64, 1.24, C.warnInk, "✕  LEVERS RULED OUT", C.warnInk, [
  { text: "Cheap augmentation (B1) — backfired, below chance", options: { bullet: { indent: 12 } } },
  { text: "Multi-task objective (E4/E4r) — degrades BOTH tasks", options: { bullet: { indent: 12 }, paraSpaceBefore: 4 } },
]);
card(5.00, 1.48, C.gMid, "→  SEARCH IS NARROWED", C.primary, [
  { text: "Topology comes from aggregation (E2) — not from inputs (E1) or objectives (E4)", options: { bullet: { indent: 12 } } },
  { text: "Open problem: keep that topology while holding features — needs a stronger anchor than masked-reconstruction, or a different backbone", options: { bullet: { indent: 12 }, paraSpaceBefore: 4 } },
]);

// ---------- bottom line ----------
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 6.72, w: 11.99, h: 0.52, rectRadius: 0.06, fill: { color: C.bgDark } });
s.addText([
  { text: "NET  ", options: { bold: true, color: C.gLite, charSpacing: 1 } },
  { text: "we mapped the design space for a retweet-graph foundation GNN — two winning ingredients found, two levers eliminated, and the open problem sharpened: get topology from aggregation while holding features with more than masked reconstruction.", options: { color: C.white } },
], { x: 0.95, y: 6.72, w: 11.5, h: 0.52, valign: "middle", fontFace: BFONT, fontSize: 12.5, margin: 0 });

p.writeFile({ fileName: __dirname + "/topology_feature_ssl_summary.pptx" }).then(f => console.log("wrote", f));
