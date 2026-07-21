// topology_feature_ssl — 2-slide deck (design + results).
// Same pptxgenjs style/palette as slides/build_slides.js.
// Build:  NODE_PATH=<scratchpad>/node_modules node build_slides.js
// Writes topology_feature_ssl_slides.pptx next to this file.
const path = require("path");
const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
p.author = "Philipp Eibl";
p.title = "SSL for topology + features — which lever wins";

// palette (shared with the nm_merged_vs_single deck)
const C = {
  bgDark: "0B3D2E", ink: "17251F", primary: "0F6E56",
  gLite: "A7DEC8", gMid: "1D9E75", gDark: "0B5A44",
  char: "33322F", muted: "6C7B74", panel: "EEF5F1",
  warnBg: "F7ECE4", warnInk: "A8481F", white: "FFFFFF", line: "D6E3DC",
  win: "E7F4EC", srcA: "1C7293", srcB: "C77D33",
};
const HFONT = "Cambria", BFONT = "Calibri";
const W = 13.33, H = 7.5;
const shadow = () => ({ type: "outer", color: "000000", blur: 7, offset: 3, angle: 90, opacity: 0.10 });

function header(s, kicker, title) {
  s.addText(kicker.toUpperCase(), { x: 0.7, y: 0.42, w: 12, h: 0.35, fontFace: BFONT, fontSize: 13, bold: true, color: C.gMid, charSpacing: 2, margin: 0 });
  s.addText(title, { x: 0.7, y: 0.74, w: 12.2, h: 0.8, fontFace: HFONT, fontSize: 30, bold: true, color: C.ink, margin: 0 });
}

// ============================================================
// SLIDE 1 — the experiment: goal, why it's hard, the three levers
// ============================================================
let s = p.addSlide(); s.background = { color: C.white };
header(s, "Pretraining strategy · SSL for topology + features", "Which lever makes SSL learn topology, not just features?");

// --- GOAL bar ---
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 1.60, w: 11.93, h: 0.66, rectRadius: 0.08, fill: { color: C.panel }, line: { color: C.line, width: 1 } });
s.addText([
  { text: "GOAL   ", options: { bold: true, color: C.primary, charSpacing: 1 } },
  { text: "one frozen encoder that is strong on ", options: { color: C.ink } },
  { text: "both", options: { bold: true, italic: true, color: C.ink } },
  { text: " — feature tasks (node classification + regression) ", options: { color: C.ink } },
  { text: "and", options: { bold: true, italic: true, color: C.ink } },
  { text: " a topological task (static link prediction). Learn topology ", options: { color: C.ink } },
  { text: "and", options: { italic: true, color: C.ink } },
  { text: " features — not features only.", options: { color: C.ink } },
], { x: 0.95, y: 1.60, w: 11.5, h: 0.66, valign: "middle", fontFace: BFONT, fontSize: 14.5, lineSpacingMultiple: 1.0, margin: 0 });

// --- WHY IT'S HARD callout ---
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 2.40, w: 11.93, h: 0.92, rectRadius: 0.08, fill: { color: C.warnBg }, line: { color: "E4C4AD", width: 1 } });
s.addText([
  { text: "WHY IT'S HARD   ", options: { bold: true, color: C.warnInk, charSpacing: 1 } },
  { text: "our default SSL — neighbor-matching (instance discrimination) — is ", options: { color: C.ink } },
  { text: "features-only", options: { bold: true, color: C.warnInk } },
  { text: ". It has no topological solution: rewiring the edges doesn't hurt it and capability probes sit at chance. Tuning the encoder can't change what the objective rewards — so the fix must come from one of three levers.", options: { color: C.ink } },
], { x: 0.95, y: 2.40, w: 11.5, h: 0.92, valign: "middle", fontFace: BFONT, fontSize: 14, lineSpacingMultiple: 1.03, margin: 0 });

// --- the three-lever map ---
function leverLabel(s, x, y, w, h, n, name, sub) {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: C.gDark }, line: { color: C.gDark, width: 1 } });
  s.addText([
    { text: `LEVER ${n}`, options: { bold: true, fontSize: 10.5, color: C.gLite, charSpacing: 1, breakLine: true, paraSpaceAfter: 3 } },
    { text: name, options: { bold: true, fontSize: 16, color: C.white, breakLine: true, paraSpaceAfter: 2 } },
    { text: sub, options: { fontSize: 10.5, color: "CFE6DB", italic: true } },
  ], { x: x + 0.15, y, w: w - 0.28, h, valign: "middle", fontFace: BFONT, lineSpacingMultiple: 1.0, margin: 0 });
}
function armChip(s, x, y, w, h, code, desc, st) {
  const pend = st === "pending", anchor = st === "anchor";
  const fill = anchor ? "F3F8F5" : (pend ? C.white : C.panel);
  const line = pend ? C.muted : (anchor ? C.line : C.gMid);
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.09, fill: { color: fill }, line: { color: line, width: pend ? 1.25 : 1.5, dashType: pend ? "dash" : "solid" }, shadow: st === "run" ? shadow() : undefined });
  s.addText([
    { text: code + (anchor ? "  ·  baseline" : ""), options: { bold: true, fontSize: 14, color: pend ? C.muted : (anchor ? C.gDark : C.primary), breakLine: true, paraSpaceAfter: 2 } },
    { text: desc, options: { fontSize: 10, color: pend ? C.muted : C.ink } },
  ], { x: x + 0.14, y: y + 0.05, w: w - 0.28, h: h - 0.1, valign: "middle", fontFace: BFONT, lineSpacingMultiple: 1.0, margin: 0 });
}
function arrow(s, xEnd0, xStart1, y, label) {
  const ax = xEnd0 + 0.06, aw = xStart1 - xEnd0 - 0.12;
  s.addShape(p.shapes.LINE, { x: ax, y: y + 0.41, w: aw, h: 0, line: { color: C.gMid, width: 2, endArrowType: "triangle" } });
  s.addText(label, { x: xEnd0 - 0.1, y: y - 0.14, w: (xStart1 - xEnd0) + 0.2, h: 0.28, align: "center", fontFace: BFONT, fontSize: 9.5, italic: true, color: C.gMid, margin: 0 });
}

const laneY = [3.44, 4.55, 5.66], laneH = 1.02;
const chipY = i => laneY[i] + 0.10, chipH = 0.82;
const cx = [3.30, 6.55, 9.80], chipW = 2.70; // chip left-edges; ends at cx+chipW

// lane 1 — augmentation
leverLabel(s, 0.7, laneY[0], 2.42, laneH, 1, "Augmentation", "corrupt the feature shortcut");
armChip(s, cx[0], chipY(0), chipW, chipH, "B0", "control — NM, mean-agg, bio feats", "anchor");
armChip(s, cx[1], chipY(0), chipW, chipH, "B1", "+ NR0.3 feature-shortcut corruption", "run");
arrow(s, cx[0] + chipW, cx[1], chipY(0), "does aug fix it?");

// lane 2 — encoder
leverLabel(s, 0.7, laneY[1], 2.42, laneH, 2, "Encoder", "make topology representable");
armChip(s, cx[0], chipY(1), chipW, chipH, "B0", "control — mean-agg SAGE", "anchor");
armChip(s, cx[1], chipY(1), chipW, chipH, "E1", "+ directed structural inputs (in/out/log-deg)", "run");
armChip(s, cx[2], chipY(1), chipW, chipH, "E2", "+ count-aware aggregator (mean⊕sum⊕max)", "run");
arrow(s, cx[0] + chipW, cx[1], chipY(1), "inject structure");
arrow(s, cx[1] + chipW, cx[2], chipY(1), "count it");

// lane 3 — objective
leverLabel(s, 0.7, laneY[2], 2.42, laneH, 3, "Objective", "make topology used");
armChip(s, cx[0], chipY(2), chipW, chipH, "E2", "capable encoder", "anchor");
armChip(s, cx[1], chipY(2), chipW, chipH, "E3", "swap NM → masked-feature reconstruction", "pending");
armChip(s, cx[2], chipY(2), chipW, chipH, "E4", "+ multi-task: dir-LP ⊕ structural head", "pending");
arrow(s, cx[0] + chipW, cx[1], chipY(2), "generative");
arrow(s, cx[1] + chipW, cx[2], chipY(2), "+ topo heads");

// legend chips (run vs next)
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 9.95, y: 6.86, w: 0.34, h: 0.2, rectRadius: 0.04, fill: { color: C.panel }, line: { color: C.gMid, width: 1.5 } });
s.addText("run (4/6)", { x: 10.33, y: 6.80, w: 1.2, h: 0.3, fontFace: BFONT, fontSize: 10, color: C.muted, margin: 0 });
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 11.45, y: 6.86, w: 0.34, h: 0.2, rectRadius: 0.04, fill: { color: C.white }, line: { color: C.muted, width: 1.25, dashType: "dash" } });
s.addText("next", { x: 11.83, y: 6.80, w: 0.9, h: 0.3, fontFace: BFONT, fontSize: 10, color: C.muted, margin: 0 });

// footer — the deciding diagnostics
s.addText([
  { text: "Decided by free diagnostics (primary, seed-robust): ", options: { bold: true, color: C.primary } },
  { text: "2×2 ablation {real / random feats} × {real / rewired edges}, planted-rule capability probes, and a trivial-floor baseline (probe raw features / degree, no encoder). Benchmark is confirmatory.", options: { color: C.muted } },
], { x: 0.7, y: 6.82, w: 9.1, h: 0.5, fontFace: BFONT, fontSize: 10.5, italic: true, lineSpacingMultiple: 1.0, margin: 0 });

// ============================================================
// SLIDE 2 — what won, what's next
// ============================================================
s = p.addSlide(); s.background = { color: C.white };
header(s, "Results · 4 of 6 arms + diagnostics · matched budget, 1 seed", "Structural inputs win — the encoder and augmentation don't");

// --- the bar to clear ---
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 1.58, w: 11.93, h: 0.62, rectRadius: 0.08, fill: { color: C.panel }, line: { color: C.line, width: 1 } });
s.addText([
  { text: "THE BAR TO CLEAR   ", options: { bold: true, color: C.primary, charSpacing: 1 } },
  { text: "an arm improves transfer only if it beats the ", options: { color: C.ink } },
  { text: "trivial floor", options: { bold: true, color: C.ink } },
  { text: " — a 10-shot linear probe on raw bio features / raw degree, no encoder. On regression, B0 · B1 · E2 all lose to doing nothing; only E1 clears it.", options: { color: C.ink } },
], { x: 0.95, y: 1.58, w: 11.5, h: 0.62, valign: "middle", fontFace: BFONT, fontSize: 13.5, lineSpacingMultiple: 1.0, margin: 0 });

// --- results table ---
const th = t => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.gDark }, fontSize: 11, align: "center", valign: "middle" } });
const td = (t, o = {}) => ({ text: t, options: Object.assign({ color: C.ink, fontSize: 12, align: "center", valign: "middle" }, o) });
const winF = { color: C.win };
const rows = [
  [th("Arm"), th("Lever it tests"), th("Regression¹\n(account_age)"), th("Static-LP\n(ROC-AUC)"), th("Probes²\ncount / in-deg"), th("Verdict")],
  [td("B0", { bold: true }), td("NM control (features-only)", { align: "left" }), td("−0.08  ✗", { color: C.warnInk }), td("0.72"), td(".51 / .50"), td("Features-only · structurally blind", { align: "left", fill: { color: "F2F2F0" }, color: C.muted })],
  [td("B1", { bold: true }), td("+ NR0.3 augmentation", { align: "left" }), td("−0.13  ✗", { color: C.warnInk }), td("0.36  ↓", { color: C.warnInk }), td(".52 / .51"), td("Backfires · LP below chance", { align: "left", fill: { color: C.warnBg }, color: C.warnInk })],
  [td("E1", { bold: true, fill: winF }), td("+ directed structural inputs", { align: "left", fill: winF }), td("+0.13  ✓", { bold: true, color: C.primary, fill: winF }), td("0.76", { bold: true, fill: winF }), td(".64 / .59", { fill: winF }), td("Only arm to beat the floor", { align: "left", bold: true, fill: { color: C.gMid }, color: C.white })],
  [td("E2", { bold: true }), td("+ count-aware encoder", { align: "left" }), td("−0.05  ✗", { color: C.warnInk }), td("~0.39", { color: C.warnInk }), td(".59 / .53"), td("No help · count washed out (BatchNorm)", { align: "left", fill: { color: C.warnBg }, color: C.warnInk })],
];
s.addTable(rows, { x: 0.7, y: 2.34, w: 11.93, colW: [0.82, 2.86, 1.9, 1.4, 1.75, 3.2], rowH: [0.52, 0.44, 0.44, 0.46, 0.44], border: { type: "solid", color: C.line, pt: 1 }, fontFace: BFONT, valign: "middle", autoPage: false });

s.addText([
  { text: "¹ account_age (content×structure) Spearman, 10-shot — cleanest leakage baseline; E1 also beats the floor on friends, classification & static-LP.   ", options: {} },
  { text: "² linear-probe AUC on planted single-rule graphs, chance 0.50.   Matched 30–40k budget · 1 seed · 3-way merged retweet graph (34M nodes).", options: {} },
], { x: 0.7, y: 4.66, w: 11.93, h: 0.42, fontFace: BFONT, fontSize: 9.5, italic: true, color: C.muted, lineSpacingMultiple: 1.0, margin: 0 });

// --- WHAT'S NEXT card (dark, left) ---
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 5.16, w: 7.35, h: 2.02, rectRadius: 0.1, fill: { color: C.bgDark } });
s.addText("WHAT'S NEXT — the objective axis (E3 / E4)", { x: 0.95, y: 5.30, w: 6.9, h: 0.34, fontFace: BFONT, fontSize: 13, bold: true, color: C.gLite, charSpacing: 0.5, margin: 0 });
s.addText([
  { text: "The one untested lever: ", options: { bold: true, color: C.white } },
  { text: "swap NM → a generative / multi-task objective (masked-feature reconstruction ⊕ directed-LP ⊕ structural head). Make topology ", options: { color: "CFE6DB" } },
  { text: "used", options: { italic: true, bold: true, color: C.white } },
  { text: ", not just representable.", options: { color: "CFE6DB", breakLine: true, paraSpaceAfter: 6 } },
  { text: "Not a freebie — ", options: { bold: true, color: C.white } },
  { text: "the free-preview masked-feature checkpoint (≈ E3) does not beat NM on regression (mean −0.016).", options: { color: "CFE6DB", breakLine: true, paraSpaceAfter: 6 } },
  { text: "E2's miss is fixable — ", options: { bold: true, color: C.white } },
  { text: "BatchNorm erases the count magnitude; one targeted encoder retry, else a data-ceiling conclusion.", options: { color: "CFE6DB" } },
], { x: 0.95, y: 5.66, w: 6.9, h: 1.44, fontFace: BFONT, fontSize: 11.5, valign: "top", lineSpacingMultiple: 1.05, margin: 0 });

// --- STRATEGIC TAKEAWAY card (accent, right) ---
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 8.2, y: 5.16, w: 4.43, h: 2.02, rectRadius: 0.1, fill: { color: C.primary } });
s.addText("FOR THE BEST PRETRAINING STRATEGY", { x: 8.42, y: 5.30, w: 4.0, h: 0.3, fontFace: BFONT, fontSize: 11, bold: true, color: C.gLite, charSpacing: 0.5, margin: 0 });
s.addText([
  { text: "Cheap structural inputs are a genuine, low-cost win. ", options: { bold: true, color: C.white } },
  { text: "A fancier encoder and feature-corruption augmentation are not — capacity and data-side corruption weren't the binding constraint.", options: { color: "E6F2EC" } },
], { x: 8.42, y: 5.66, w: 4.0, h: 1.5, fontFace: BFONT, fontSize: 13, valign: "top", lineSpacingMultiple: 1.08, margin: 0 });

const out = path.join(__dirname, "topology_feature_ssl_slides.pptx");
p.writeFile({ fileName: out }).then(f => console.log("wrote", f));
