const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
p.author = "Philipp Eibl";
p.title = "Merging retweet graphs for neighbor-matching";

// palette
const C = {
  bgDark: "0B3D2E", ink: "17251F", primary: "0F6E56",
  gLite: "A7DEC8", gMid: "1D9E75", gDark: "0B5A44",
  char: "33322F", muted: "6C7B74", panel: "EEF5F1",
  warnBg: "F7ECE4", warnInk: "A8481F", white: "FFFFFF", line: "D6E3DC",
  srcA: "1C7293", srcB: "C77D33",
};
const HFONT = "Cambria", BFONT = "Calibri";
const W = 13.33, H = 7.5;
const shadow = () => ({ type: "outer", color: "000000", blur: 7, offset: 3, angle: 90, opacity: 0.10 });

function node(s, cx, cy, color, d = 0.17) {
  s.addShape(p.shapes.OVAL, { x: cx - d / 2, y: cy - d / 2, w: d, h: d, fill: { color }, line: { color: "FFFFFF", width: 1 } });
}
function ring(s, cx, cy, color, d = 0.30) {
  s.addShape(p.shapes.OVAL, { x: cx - d / 2, y: cy - d / 2, w: d, h: d, fill: { color: "FFFFFF", transparency: 100 }, line: { color, width: 2 } });
}
function header(s, kicker, title) {
  s.addText(kicker.toUpperCase(), { x: 0.7, y: 0.42, w: 12, h: 0.35, fontFace: BFONT, fontSize: 13, bold: true, color: C.gMid, charSpacing: 2, margin: 0 });
  s.addText(title, { x: 0.7, y: 0.74, w: 12, h: 0.8, fontFace: HFONT, fontSize: 32, bold: true, color: C.ink, margin: 0 });
}

// ---------- Slide 1: title ----------
let s = p.addSlide(); s.background = { color: C.bgDark };
// motif nodes
[[11.7,1.2,C.gMid],[12.3,1.7,C.gLite],[11.4,2.0,C.srcB],[12.6,2.5,C.gMid],[11.9,2.7,C.gLite]].forEach(([x,y,c])=>node(s,x,y,c,0.22));
s.addShape(p.shapes.LINE,{x:11.55,y:1.31,w:0.68,h:0.44,line:{color:"3C6E5C",width:1.5}});
s.addShape(p.shapes.LINE,{x:11.5,y:1.31,w:0.42,h:0.72,line:{color:"3C6E5C",width:1.5}});
s.addShape(p.shapes.LINE,{x:11.98,y:1.78,w:0.35,h:0.72,line:{color:"3C6E5C",width:1.5}});
s.addText("NEIGHBOR-MATCHING ON RETWEET GRAPHS", { x: 0.9, y: 1.7, w: 10, h: 0.4, fontFace: BFONT, fontSize: 15, bold: true, color: C.gLite, charSpacing: 2, margin: 0 });
s.addText("Does training on more graphs help?", { x: 0.9, y: 2.25, w: 11.2, h: 1.6, fontFace: HFONT, fontSize: 52, bold: true, color: C.white, margin: 0 });
s.addText([
  { text: "On the graphs you train on — ", options: { color: "CFE6DB" } },
  { text: "yes, merging wins.", options: { color: C.gLite, bold: true } },
  { text: "   On a new, unseen graph — ", options: { color: "CFE6DB" } },
  { text: "no boost.", options: { color: C.srcB, bold: true } },
], { x: 0.9, y: 4.05, w: 11.4, h: 0.6, fontFace: BFONT, fontSize: 20, margin: 0 });
s.addText("Four training strategies · neighbor-matching · 3-shot / 30-way · single seed", { x: 0.9, y: 6.5, w: 11, h: 0.4, fontFace: BFONT, fontSize: 13, color: "9DBBAD", margin: 0 });

// ---------- helper: graph cloud ----------
function graphCloud(s, x, y, w, h, label, color, seed) {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: shadow() });
  s.addText(label, { x: x, y: y + h - 0.42, w: w, h: 0.32, align: "center", fontFace: BFONT, fontSize: 12, italic: true, color: C.muted, margin: 0 });
  const pts = [[.28,.34],[.5,.24],[.72,.34],[.35,.55],[.6,.52],[.8,.6],[.46,.7],[.24,.68]];
  const cs = [[0,1],[1,2],[0,3],[3,4],[1,4],[4,5],[3,6],[4,6],[6,7],[3,7]];
  const P = pts.map(([px,py])=>[x+px*w, y+0.12+py*(h-0.6)]);
  cs.forEach(([a,b])=> s.addShape(p.shapes.LINE,{x:P[a][0],y:P[a][1],w:P[b][0]-P[a][0],h:P[b][1]-P[a][1],line:{color:C.line,width:1.25}}));
  P.forEach(([px,py])=>node(s,px,py,color,0.16));
}
// helper: episode box
function episodeBox(s, x, y, w, h, title, nodes, opts={}) {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: opts.fill||C.panel }, line: { color: opts.line||C.gMid, width: opts.lw||1.5, dashType: opts.dash||"solid" }, shadow: opts.shadow?shadow():undefined });
  s.addText(title, { x, y: y+0.08, w, h: 0.3, align:"center", fontFace: BFONT, fontSize: 12, bold:true, color: opts.titleColor||C.primary, margin:0 });
  nodes.forEach(([px,py,c,isCenter])=>{ if(isCenter) ring(s,x+px*w,y+py*h,c,0.34); node(s,x+px*w,y+py*h,c,0.18); });
}

// ---------- Slide 2: single-source ----------
s = p.addSlide(); s.background = { color: C.white }; header(s, "Strategy 1 of 4 · baseline", "Single-source");
s.addText([
  { text: "Train on one graph only.", options:{ bold:true, color:C.ink, breakLine:true, paraSpaceAfter:10 } },
  { text: "Every neighbor-matching episode — a center node, its true neighbor, and negatives — is drawn from that single graph.", options:{ color:C.ink, breakLine:true, paraSpaceAfter:10 } },
  { text: "Strong on its own domain, but nothing forces it to work anywhere else.", options:{ color:C.muted } },
], { x: 0.7, y: 2.0, w: 5.3, h: 3, fontFace: BFONT, fontSize: 17, lineSpacingMultiple: 1.15, margin: 0 });
graphCloud(s, 7.0, 1.9, 3.0, 2.9, "one source graph (e.g. covid)", C.srcA);
s.addShape(p.shapes.LINE, { x: 10.15, y: 3.3, w: 0.7, h: 0, line: { color: C.gMid, width: 2, endArrowType:"triangle" } });
episodeBox(s, 10.95, 2.55, 1.75, 1.5, "episode", [[.3,.62,C.srcA,true],[.55,.42,C.srcA],[.72,.66,C.srcA],[.45,.78,C.srcA]]);

// ---------- Slide 3: merged naive ----------
s = p.addSlide(); s.background = { color: C.white }; header(s, "Strategy 2 of 4", "Merged (naive)");
s.addText([
  { text: "Concatenate the graphs as disjoint components", options:{ bold:true, color:C.ink } },
  { text: " (no edges between them).", options:{ color:C.ink, breakLine:true, paraSpaceAfter:10 } },
  { text: "Sample episodes uniformly over all merged nodes — so a single episode can contain nodes from ", options:{ color:C.ink } },
  { text: "both sources.", options:{ color:C.warnInk, bold:true, breakLine:true, paraSpaceAfter:10 } },
  { text: "This mixing is what the next two strategies fix.", options:{ color:C.muted, italic:true } },
], { x: 0.7, y: 2.0, w: 5.3, h: 3, fontFace: BFONT, fontSize: 17, lineSpacingMultiple: 1.15, margin: 0 });
graphCloud(s, 6.7, 1.9, 2.5, 2.9, "source A (covid)", C.srcA);
graphCloud(s, 6.7, 4.9, 2.5, 2.1, "source B (ukr)", C.srcB);
s.addShape(p.shapes.LINE, { x: 9.35, y: 3.6, w: 0.7, h: 0, line: { color: C.gMid, width: 2, endArrowType:"triangle" } });
episodeBox(s, 10.15, 2.75, 2.4, 1.9, "merged episode — mixes sources", [[.28,.6,C.srcB,true],[.5,.4,C.srcA],[.68,.62,C.srcA],[.44,.78,C.srcB],[.8,.44,C.srcA]], {line:C.warnInk, dash:"dash"});

// ---------- Slide 4: merged-within (shortcut) ----------
s = p.addSlide(); s.background = { color: C.white }; header(s, "Strategy 3 of 4 · improvement", "Merged-within-source");
// problem callout
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 1.78, w: 11.9, h: 1.02, rectRadius: 0.08, fill: { color: C.warnBg }, line: { color: "E4C4AD", width: 1 } });
s.addText([
  { text: "Problem — cross-source shortcut.  ", options:{ bold:true, color:C.warnInk } },
  { text: "In a mixed episode the true neighbor and the negatives come from different graphs, so the model can separate them by ", options:{ color:C.ink } },
  { text: "which source they belong to", options:{ italic:true, color:C.warnInk } },
  { text: " — not by neighborhood structure. It learns a shortcut that doesn't transfer.", options:{ color:C.ink } },
], { x: 0.95, y: 1.9, w: 11.4, h: 0.8, fontFace: BFONT, fontSize: 15.5, valign:"middle", lineSpacingMultiple:1.05, margin: 0 });
// before / after
episodeBox(s, 1.4, 3.35, 3.6, 2.5, "naive episode (mixed)", [[.3,.6,C.srcB,true],[.52,.4,C.srcA],[.7,.64,C.srcA],[.46,.8,C.srcA],[.82,.46,C.srcA]], {line:C.warnInk, dash:"dash", fill:"FBF4EE"});
s.addText("negatives are all the OTHER color → guess by source", { x: 1.4, y: 5.9, w: 3.6, h: 0.5, align:"center", fontFace: BFONT, fontSize: 12.5, italic:true, color: C.warnInk, margin:0 });
s.addShape(p.shapes.LINE, { x: 5.35, y: 4.6, w: 1.0, h: 0, line: { color: C.gMid, width: 2.5, endArrowType:"triangle" } });
s.addText("confine to\none source", { x: 5.05, y: 3.95, w: 1.6, h: 0.6, align:"center", fontFace: BFONT, fontSize: 11.5, bold:true, color: C.primary, margin:0 });
episodeBox(s, 6.7, 3.35, 3.6, 2.5, "within-source episode", [[.3,.6,C.srcB,true],[.52,.4,C.srcB],[.7,.64,C.srcB],[.46,.8,C.srcB],[.82,.46,C.srcB]], {line:C.gMid, fill:C.panel, shadow:true});
s.addText("all one source → must use neighborhood structure", { x: 6.7, y: 5.9, w: 3.6, h: 0.5, align:"center", fontFace: BFONT, fontSize: 12.5, italic:true, color: C.primary, margin:0 });
// fix note card
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 10.75, y: 3.35, w: 1.85, h: 2.5, rectRadius:0.08, fill:{color:C.bgDark} });
s.addText([{text:"THE FIX",options:{bold:true,color:C.gLite,charSpacing:1,breakLine:true,paraSpaceAfter:8,fontSize:12}},{text:"Each episode drawn from one source (chosen ∝ its size).",options:{color:C.white,fontSize:14}}], { x: 10.95, y: 3.6, w: 1.5, h: 2, fontFace: BFONT, valign:"top", lineSpacingMultiple:1.1, margin:0 });

// ---------- Slide 5: merged-within-balanced (starvation) ----------
s = p.addSlide(); s.background = { color: C.white }; header(s, "Strategy 4 of 4 · improvement", "Merged-within-balanced");
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 1.78, w: 11.9, h: 1.02, rectRadius: 0.08, fill: { color: C.warnBg }, line: { color: "E4C4AD", width: 1 } });
s.addText([
  { text: "Problem — the small source is starved.  ", options:{ bold:true, color:C.warnInk } },
  { text: "When sources differ wildly in size (covid ≈ 23M nodes, midterm ≈ 0.34M), proportional sampling trains on midterm only ", options:{ color:C.ink } },
  { text: "~1.5% of episodes", options:{ bold:true, color:C.warnInk } },
  { text: " — so the merged model effectively ignores it.", options:{ color:C.ink } },
], { x: 0.95, y: 1.9, w: 11.4, h: 0.8, fontFace: BFONT, fontSize: 15.5, valign:"middle", lineSpacingMultiple:1.05, margin: 0 });
// composition bars
function compBar(s, x, y, w, label, covidFrac, note, noteColor) {
  s.addText(label, { x: x, y: y-0.34, w: w, h: 0.3, fontFace: BFONT, fontSize: 13, bold:true, color: C.ink, margin:0 });
  const cW = w*covidFrac, mW = w*(1-covidFrac);
  s.addShape(p.shapes.RECTANGLE, { x, y, w: cW, h: 0.55, fill:{color:C.srcA} });
  s.addShape(p.shapes.RECTANGLE, { x: x+cW+0.02, y, w: Math.max(mW,0.05), h: 0.55, fill:{color:C.srcB} });
  s.addText(`covid ${Math.round(covidFrac*100)}%`, { x, y, w: cW, h:0.55, align:"center", valign:"middle", fontFace:BFONT, fontSize:13, bold:true, color:"FFFFFF", margin:0 });
  s.addText(note, { x: x, y: y+0.62, w: w, h: 0.3, align:"right", fontFace: BFONT, fontSize: 12, italic:true, color: noteColor, margin:0 });
}
compBar(s, 1.2, 3.55, 7.6, "Proportional (naive / within-source)", 0.985, "midterm 1.5% — a sliver of episodes", C.warnInk);
// callout for the sliver
s.addText("midterm →", { x: 8.85, y: 3.5, w: 1.4, h: 0.3, fontFace:BFONT, fontSize:11.5, color:C.srcB, bold:true, margin:0 });
compBar(s, 1.2, 5.35, 7.6, "Balanced (pick the source 50 / 50)", 0.5, "midterm 50% — equal exposure", C.primary);
s.addText("midterm", { x: 5.05, y: 5.35, w: 3.9, h:0.55, align:"center", valign:"middle", fontFace:BFONT, fontSize:13, bold:true, color:"FFFFFF", margin:0 });
// fix card
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 9.55, y: 3.35, w: 3.05, h: 2.9, rectRadius:0.08, fill:{color:C.bgDark} });
s.addText([{text:"THE FIX",options:{bold:true,color:C.gLite,charSpacing:1,breakLine:true,paraSpaceAfter:8,fontSize:12}},
  {text:"Choose the episode's source uniformly, not by size.",options:{color:C.white,fontSize:15,breakLine:true,paraSpaceAfter:10}},
  {text:"Rescues the small domain: midterm 0.33 → 0.43 accuracy.",options:{color:C.gLite,fontSize:14,italic:true}}], { x: 9.78, y: 3.62, w: 2.6, h: 2.4, fontFace: BFONT, valign:"top", lineSpacingMultiple:1.1, margin:0 });

// ---------- Slide 6: in-distribution ----------
s = p.addSlide(); s.background = { color: C.white }; header(s, "Results · trained-on graphs", "Merging buys a generalist");
function scatter(s, px, py, pw, ph, title, xlab, ylab, xr, yr, pts) {
  s.addText(title, { x: px, y: py-0.42, w: pw, h: 0.32, align:"center", fontFace:HFONT, fontSize:16, bold:true, color:C.ink, margin:0 });
  s.addShape(p.shapes.RECTANGLE, { x: px, y: py, w: pw, h: ph, fill:{color:"FBFDFC"}, line:{color:C.line,width:1} });
  // top-right hint
  s.addText("good on both →", { x: px+pw-2.1, y: py+0.08, w: 2.0, h:0.25, align:"right", fontFace:BFONT, fontSize:10.5, italic:true, color:C.muted, margin:0 });
  const sx=v=>px+ (v-xr[0])/(xr[1]-xr[0])*pw;
  const sy=v=>py+ph-(v-yr[0])/(yr[1]-yr[0])*ph;
  s.addText(xlab, { x: px, y: py+ph+0.06, w: pw, h:0.3, align:"center", fontFace:BFONT, fontSize:12, color:C.muted, margin:0 });
  s.addText(ylab, { x: px-1.15, y: py+ph/2-0.6, w: 1.2, h:0.3, align:"center", fontFace:BFONT, fontSize:12, color:C.muted, rotate:270, margin:0 });
  pts.forEach(pt=>{ const cx=sx(pt.x), cy=sy(pt.y); node(s,cx,cy,pt.c,0.19);
    s.addText(pt.label, { x: cx+pt.lx, y: cy+pt.ly, w: pt.lw||2.0, h:0.25, align:pt.la||"left", fontFace:BFONT, fontSize:10.5, bold:pt.bold, color:pt.c==C.gLite?"5F9C84":pt.c, margin:0 }); });
}
scatter(s, 1.9, 2.55, 4.3, 3.4, "ukr + covid  (balanced sizes)", "ROC-AUC on covid →", "ROC-AUC on ukr →",
  [0.972,0.985],[0.918,0.955],
  [{x:0.974,y:0.950,c:C.char,label:"ukr",lx:-0.62,ly:-0.14,la:"right",lw:0.5,bold:1},
   {x:0.982,y:0.924,c:C.char,label:"covid",lx:0.14,ly:-0.02,bold:1},
   {x:0.978,y:0.937,c:C.gLite,label:"merged",lx:-1.4,ly:-0.02,la:"right",lw:1.3},
   {x:0.981,y:0.945,c:C.gMid,label:"merged-within",lx:0.14,ly:-0.05,bold:1}]);
scatter(s, 8.3, 2.55, 4.3, 3.4, "covid + midterm  (imbalanced)", "ROC-AUC on covid →", "ROC-AUC on midterm →",
  [0.865,0.99],[0.878,0.932],
  [{x:0.879,y:0.926,c:C.char,label:"midterm",lx:0.14,ly:-0.05,bold:1},
   {x:0.981,y:0.886,c:C.char,label:"covid",lx:-0.66,ly:0.02,la:"right",lw:0.55,bold:1},
   {x:0.981,y:0.885,c:C.gLite,label:"merged",lx:0.14,ly:0.02},
   {x:0.981,y:0.894,c:C.gMid,label:"merged-within",lx:0.14,ly:-0.14,bold:1},
   {x:0.978,y:0.923,c:C.gDark,label:"merged-within-balanced",lx:-3.0,ly:-0.02,la:"right",lw:2.9,bold:1}]);
s.addText([
  { text:"Single models are lopsided (great on one axis, weak on the other). Merged models pull to the top-right. ", options:{color:C.ink} },
  { text:"Under imbalance, only ", options:{color:C.ink} },
  { text:"balanced", options:{bold:true,color:C.primary} },
  { text:" reaches it — the rest collapse onto covid and ignore midterm.", options:{color:C.ink} },
], { x: 1.9, y: 6.55, w: 10.7, h: 0.7, fontFace:BFONT, fontSize:14.5, margin:0 });

// ---------- Slide 7: OOD ----------
s = p.addSlide(); s.background = { color: C.white }; header(s, "Results · held-out (unseen) graph", "No generalization boost");
function oodChart(s, x, y, w, h, title, ceil, ceilLbl, cats, vals, cols, minv) {
  s.addText(title, { x, y: y-0.42, w, h:0.32, align:"center", fontFace:HFONT, fontSize:16, bold:true, color:C.ink, margin:0 });
  s.addChart(p.charts.BAR, [{ name:"AUC", labels:cats, values:vals }], {
    x, y, w, h, barDir:"bar", chartColors: cols, valAxisMinVal:minv, valAxisMaxVal:0.96,
    showValue:true, dataLabelPosition:"outEnd", dataLabelColor:C.ink, dataLabelFontSize:11, dataLabelFontFace:BFONT,
    catAxisLabelColor:C.ink, catAxisLabelFontSize:11.5, catAxisLabelFontFace:BFONT,
    valAxisHidden:true, valGridLine:{style:"none"}, catGridLine:{style:"none"},
    chartArea:{fill:{color:"FFFFFF"}}, showLegend:false, barGapWidthPct:45,
  });
}
oodChart(s, 3.0, 2.5, 4.0, 3.2, "Train ukr+cov → test midterm", null, null,
  ["single ukr","single covid","merged","merged-within"], [0.884,0.884,0.876,0.885],
  [C.char,C.char,C.gLite,C.gMid], 0.83);
oodChart(s, 9.0, 2.5, 4.0, 3.2, "Train cov+mid → test ukr", null, null,
  ["single covid","merged","merged-within","merged-bal"], [0.926,0.925,0.924,0.921],
  [C.char,C.gLite,C.gMid,C.gDark], 0.90);
s.addText([
  { text:"On a graph no model trained on, merged ≈ the best single source — no gain from more graphs. ", options:{color:C.ink} },
  { text:"Transfer rides on the most-similar source (covid); ", options:{color:C.ink} },
  { text:"balanced is actually weakest OOD.", options:{bold:true,color:C.warnInk} },
  { text:"  (In-domain ceilings: midterm 0.93, ukr 0.95 — all bars fall short.)", options:{italic:true,color:C.muted} },
], { x: 1.4, y: 6.35, w: 11.5, h: 0.8, fontFace:BFONT, fontSize:14.5, margin:0 });

// ---------- Slide 8: summary ----------
s = p.addSlide(); s.background = { color: C.bgDark };
s.addText("Two takeaways", { x: 0.9, y: 0.8, w: 11, h: 0.9, fontFace: HFONT, fontSize: 34, bold: true, color: C.white, margin: 0 });
function takeaway(s, y, n, head, body, accent) {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 0.9, y, w: 11.5, h: 2.1, rectRadius:0.1, fill:{color:"124A38"} });
  s.addShape(p.shapes.OVAL, { x: 1.25, y: y+0.6, w: 0.9, h:0.9, fill:{color:accent} });
  s.addText(String(n), { x: 1.25, y: y+0.6, w:0.9, h:0.9, align:"center", valign:"middle", fontFace:HFONT, fontSize:30, bold:true, color:C.bgDark, margin:0 });
  s.addText(head, { x: 2.5, y: y+0.32, w: 9.5, h: 0.6, fontFace:HFONT, fontSize:22, bold:true, color:C.white, margin:0 });
  s.addText(body, { x: 2.5, y: y+0.92, w: 9.6, h: 1.0, fontFace:BFONT, fontSize:16, color:"CFE6DB", lineSpacingMultiple:1.1, margin:0 });
}
takeaway(s, 2.0, 1, "Merging wins on the graphs you train on.", "No inversion: a merged model matches or beats single-source on every trained domain. Under size imbalance, balanced within-source sampling is what keeps the small domain from being starved.", C.gLite);
takeaway(s, 4.5, 2, "But more graphs ≠ better generalization.", "On an unseen graph, merged gives no boost over the best single source — transfer rides on having a similar source, not more of them. Balanced, best in-distribution, is the weakest out-of-distribution.", C.srcB);

p.writeFile({ fileName: "/private/tmp/claude-502/-Users-philipp-projects-gfm-prodigy/69eeedce-4dbe-4115-8a16-45ddd7d93d61/scratchpad/deck/nm_merged_vs_single.pptx" }).then(f => console.log("wrote", f));
