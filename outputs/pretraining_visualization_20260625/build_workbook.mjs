import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const inputPath = "/Users/philipp/Downloads/experiments_jun_24.tsv";
const outputDir = "/Users/philipp/projects/gfm/prodigy/outputs/pretraining_visualization_20260625";
const outputPath = `${outputDir}/gnn_pretraining_strategy_analysis.xlsx`;

const lines = (await fs.readFile(inputPath, "utf8"))
  .trim()
  .split(/\r?\n/)
  .map((line) => line.split("\t"));
const headers = lines[0];
const rows = lines.slice(1);
const testHeaders = headers.slice(1, 13);

const lpIndexes = testHeaders.map((h, i) => (h.endsWith("+lp") ? i : -1)).filter((i) => i >= 0);
const nmIndexes = testHeaders.map((h, i) => (h.endsWith("+nm") ? i : -1)).filter((i) => i >= 0);
const plIndexes = testHeaders.map((h, i) => (h.endsWith("+pl") ? i : -1)).filter((i) => i >= 0);

const computed = rows.map((row, sourceIndex) => {
  const scores = row.slice(1, 13).map(Number);
  const mean = (indexes) => indexes.reduce((sum, i) => sum + scores[i], 0) / indexes.length;
  const lp = mean(lpIndexes);
  const nm = mean(nmIndexes);
  const pl = mean(plIndexes);
  return {
    sourceIndex,
    trainId: row[0],
    experiment: row[13],
    strategy: row[14],
    scores,
    lp,
    nm,
    pl,
    balanced: (lp + nm + pl) / 3,
    overall: scores.reduce((a, b) => a + b, 0) / scores.length,
  };
});

for (let col = 0; col < 12; col++) {
  const ranked = [...computed].sort((a, b) => b.scores[col] - a.scores[col]);
  ranked.forEach((item, index) => {
    item.rankSum = (item.rankSum ?? 0) + index + 1;
    item.wins = (item.wins ?? 0) + (index === 0 ? 1 : 0);
  });
}
for (const item of computed) item.meanRank = item.rankSum / 12;

const byBalanced = [...computed].sort((a, b) => b.balanced - a.balanced);
const byMeanRank = [...computed].sort((a, b) => a.meanRank - b.meanRank);
const maxWins = [...computed].sort((a, b) => b.wins - a.wins || a.meanRank - b.meanRank);
const rowByTrainId = new Map(computed.map((item) => [item.trainId, item.sourceIndex + 2]));
const shortLabel = (trainId) => {
  const labels = {
    "covid19_twitter+nm": "Cov NM",
    "midterm+nm": "Mid NM",
    "ukr_rus_twitter+nm": "Ukr NM",
    "covid19_twitter+pl": "Cov PL",
    "ukr_rus_twitter+pl": "Ukr PL",
    "midterm+pl": "Mid PL",
    "covid19_twitter+lp": "Cov LP",
    "midterm+lp": "Mid LP",
    "ukr_rus_twitter+lp": "Ukr LP",
    "covid19_twitter+lp>ukr_rus_twitter+lp": "Cov LP → Ukr LP",
    "midterm+lp>covid19_twitter+lp": "Mid LP → Cov LP",
    "midterm+lp>ukr_rus_twitter+lp": "Mid LP → Ukr LP",
    "covid19_twitter+nm>ukr_rus_twitter+nm": "Cov NM → Ukr NM",
    "midterm+nm>covid19_twitter+nm": "Mid NM → Cov NM",
    "midterm+nm>ukr_rus_twitter+nm": "Mid NM → Ukr NM",
    "midterm+nm>covid19_twitter+lp": "Mid NM → Cov LP",
    "covid19_twitter+nm>ukr_rus_twitter+lp": "Cov NM → Ukr LP",
    "midterm+nm>ukr_rus_twitter+lp": "Mid NM → Ukr LP",
    merged_nm: "2G merged NM",
    merged_lp: "2G merged LP",
    merged_ukr_covid_midterm_aug_10k: "3G merged + joint",
  };
  return labels[trainId] ?? trainId;
};

const workbook = Workbook.create();
const dashboard = workbook.worksheets.add("Dashboard");
const metrics = workbook.worksheets.add("Metrics");
const heatmap = workbook.worksheets.add("Heatmap");
const raw = workbook.worksheets.add("Raw Data");
const readme = workbook.worksheets.add("README");

const navy = "#17324D";
const blue = "#2F6B9A";
const teal = "#2A9D8F";
const gold = "#E9C46A";
const orange = "#F4A261";
const red = "#E76F51";
const paleBlue = "#EAF2F8";
const paleTeal = "#E8F5F2";
const paleGold = "#FFF6D8";
const paleGray = "#F4F6F8";
const midGray = "#D7DEE5";
const darkGray = "#455A64";
const white = "#FFFFFF";

for (const sheet of [dashboard, metrics, heatmap, raw, readme]) {
  sheet.showGridLines = false;
}

// Raw data: untouched source table.
raw.getRangeByIndexes(0, 0, rows.length + 1, headers.length).values = [
  headers,
  ...rows.map((row) => row.map((value, i) => (i >= 1 && i <= 12 ? Number(value) : value))),
];
raw.getRange("A1:O1").format = {
  fill: navy,
  font: { bold: true, color: white },
  wrapText: true,
  verticalAlignment: "center",
};
raw.getRange("A2:A22").format.font = { color: navy };
raw.getRange("B2:M22").format.numberFormat = "0.000";
raw.getRange("B2:M22").conditionalFormats.add("colorScale", {
  colors: ["#FDECEC", "#FFF4CC", "#DFF3EA"],
  thresholds: ["min", { type: "percentile", value: 50 }, "max"],
});
raw.getRange("A1:O22").format.borders = {
  insideHorizontal: { style: "thin", color: "#E8ECEF" },
  bottom: { style: "thin", color: midGray },
};
raw.freezePanes.freezeRows(1);
raw.freezePanes.freezeColumns(1);
raw.getRange("A:O").format.autofitColumns();
raw.getRange("A:A").format.columnWidth = 39;
raw.getRange("B:M").format.columnWidth = 19;
raw.getRange("N:N").format.columnWidth = 27;
raw.getRange("O:O").format.columnWidth = 38;
raw.getRange("1:1").format.rowHeight = 42;
raw.tables.add("A1:O22", true, "RawResultsTable").style = "TableStyleMedium2";

// Formula-driven metric table.
const metricHeaders = [
  "train_id",
  "Experiment",
  "Pretrain Strategy",
  "LP mean",
  "NM mean",
  "PL mean",
  "Balanced score",
  "All-test mean",
  "Worst test",
  "Mean test rank",
  "Test wins",
  "Mean regret vs best",
];
metrics.getRange("A1:L1").values = [metricHeaders];
metrics.getRange("A1:L1").format = {
  fill: navy,
  font: { bold: true, color: white },
  wrapText: true,
};

for (let i = 0; i < rows.length; i++) {
  const r = i + 2;
  const rawR = i + 2;
  metrics.getRange(`A${r}:C${r}`).formulas = [[
    `='Raw Data'!A${rawR}`,
    `='Raw Data'!N${rawR}`,
    `='Raw Data'!O${rawR}`,
  ]];
  metrics.getRange(`D${r}:I${r}`).formulas = [[
    `=AVERAGE('Raw Data'!B${rawR},'Raw Data'!H${rawR},'Raw Data'!L${rawR})`,
    `=AVERAGE('Raw Data'!C${rawR},'Raw Data'!D${rawR},'Raw Data'!F${rawR},'Raw Data'!I${rawR},'Raw Data'!J${rawR},'Raw Data'!M${rawR})`,
    `=AVERAGE('Raw Data'!E${rawR},'Raw Data'!G${rawR},'Raw Data'!K${rawR})`,
    `=AVERAGE(D${r}:F${r})`,
    `=AVERAGE('Raw Data'!B${rawR}:M${rawR})`,
    `=MIN('Raw Data'!B${rawR}:M${rawR})`,
  ]];
  const rankTerms = Array.from({ length: 12 }, (_, j) => {
    const col = String.fromCharCode("B".charCodeAt(0) + j);
    return `RANK.EQ('Raw Data'!${col}${rawR},'Raw Data'!${col}$2:${col}$22,0)`;
  });
  const winTerms = Array.from({ length: 12 }, (_, j) => {
    const col = String.fromCharCode("B".charCodeAt(0) + j);
    return `--('Raw Data'!${col}${rawR}=MAX('Raw Data'!${col}$2:${col}$22))`;
  });
  const regretTerms = Array.from({ length: 12 }, (_, j) => {
    const col = String.fromCharCode("B".charCodeAt(0) + j);
    return `(MAX('Raw Data'!${col}$2:${col}$22)-'Raw Data'!${col}${rawR})`;
  });
  metrics.getRange(`J${r}:L${r}`).formulas = [[
    `=AVERAGE(${rankTerms.join(",")})`,
    `=SUM(${winTerms.join(",")})`,
    `=AVERAGE(${regretTerms.join(",")})`,
  ]];
}
metrics.getRange("D2:I22").format.numberFormat = "0.000";
metrics.getRange("J2:J22").format.numberFormat = "0.00";
metrics.getRange("K2:K22").format.numberFormat = "0";
metrics.getRange("L2:L22").format.numberFormat = "0.000";
metrics.getRange("G2:G22").conditionalFormats.add("dataBar", { color: blue, gradient: true });
metrics.getRange("J2:J22").conditionalFormats.add("colorScale", {
  colors: ["#DFF3EA", "#FFF4CC", "#FDECEC"],
  thresholds: ["min", { type: "percentile", value: 50 }, "max"],
});
metrics.getRange("K2:K22").conditionalFormats.add("dataBar", { color: teal, gradient: true });
metrics.getRange("A1:L22").format.borders = {
  insideHorizontal: { style: "thin", color: "#E8ECEF" },
};
metrics.freezePanes.freezeRows(1);
metrics.freezePanes.freezeColumns(1);
metrics.getRange("A:L").format.autofitColumns();
metrics.getRange("A:A").format.columnWidth = 39;
metrics.getRange("B:B").format.columnWidth = 27;
metrics.getRange("C:C").format.columnWidth = 38;
metrics.getRange("D:L").format.columnWidth = 15;
metrics.getRange("1:1").format.rowHeight = 48;
metrics.tables.add("A1:L22", true, "MetricsTable").style = "TableStyleMedium2";

// Dashboard.
dashboard.getRange("A1:N2").merge();
dashboard.getRange("A1").values = [["GNN Pretraining Strategy — Decision Dashboard"]];
dashboard.getRange("A1:N2").format = {
  fill: navy,
  font: { bold: true, color: white, size: 20 },
  verticalAlignment: "center",
};
dashboard.getRange("A3:N3").merge();
dashboard.getRange("A3").values = [[
  "Scores are compared four ways because one aggregate can hide task transfer and brittleness.",
]];
dashboard.getRange("A3:N3").format = {
  fill: paleBlue,
  font: { color: darkGray, italic: true },
};

const cards = [
  {
    range: "A5:C8",
    label: "Best balanced score",
    value: byBalanced[0].trainId,
    metric: byBalanced[0].balanced,
    note: "Equal weight to LP, NM, and PL families",
    fill: paleBlue,
  },
  {
    range: "E5:G8",
    label: "Best mean test rank",
    value: byMeanRank[0].trainId,
    metric: byMeanRank[0].meanRank,
    note: "Lower is better across all 12 tests",
    fill: paleTeal,
  },
  {
    range: "I5:K8",
    label: "Most test-suite wins",
    value: maxWins[0].trainId,
    metric: maxWins[0].wins,
    note: "Number of columns where approach is best",
    fill: paleGold,
  },
  {
    range: "M5:N8",
    label: "Top-score gap",
    value: "Near tie",
    metric: byBalanced[0].balanced - byBalanced[1].balanced,
    note: "#1 minus #2 balanced score",
    fill: "#FCEFEA",
  },
];

for (const card of cards) {
  const [start, end] = card.range.split(":");
  const startCol = start.match(/[A-Z]+/)[0];
  const endCol = end.match(/[A-Z]+/)[0];
  dashboard.getRange(`${startCol}5:${endCol}5`).merge();
  dashboard.getRange(`${startCol}6:${endCol}6`).merge();
  dashboard.getRange(`${startCol}7:${endCol}7`).merge();
  dashboard.getRange(`${startCol}8:${endCol}8`).merge();
  dashboard.getRange(`${startCol}5`).values = [[card.label]];
  dashboard.getRange(`${startCol}6`).values = [[card.value]];
  dashboard.getRange(`${startCol}7`).values = [[card.metric]];
  dashboard.getRange(`${startCol}8`).values = [[card.note]];
  dashboard.getRange(card.range).format = {
    fill: card.fill,
    borders: { preset: "outside", style: "thin", color: midGray },
  };
  dashboard.getRange(`${startCol}5`).format.font = { bold: true, color: navy };
  dashboard.getRange(`${startCol}6`).format = {
    font: { bold: true, color: darkGray, size: 11 },
    wrapText: true,
    verticalAlignment: "center",
  };
  dashboard.getRange(`${startCol}7`).format = {
    font: { bold: true, color: navy, size: 16 },
    horizontalAlignment: "center",
  };
  dashboard.getRange(`${startCol}8`).format = {
    font: { color: darkGray, size: 9 },
    wrapText: true,
  };
}
dashboard.getRange("A7:C7").format.numberFormat = "0.000";
dashboard.getRange("E7:G7").format.numberFormat = "0.00";
dashboard.getRange("I7:K7").format.numberFormat = "0";
dashboard.getRange("M7:N7").format.numberFormat = "0.0000";

dashboard.getRange("A10:N10").merge();
dashboard.getRange("A10").values = [["What the current evidence says"]];
dashboard.getRange("A10:N10").format = {
  fill: navy,
  font: { bold: true, color: white, size: 13 },
};
dashboard.getRange("A11:N15").merge();
dashboard.getRange("A11").values = [[
  "1. NM pretraining is the reliable core: all leading broad-transfer approaches use NM.\n" +
  "2. midterm+nm → covid19_twitter+nm and the 3-graph merged / 2-task joint model are effectively tied on the balanced score (gap ≈ 0.0002).\n" +
  "3. The 3-graph joint model is the strongest breadth option: it wins 5/12 tests and is much better on PL transfer, but gives up LP performance.\n" +
  "4. covid19_twitter+nm has the best average test rank and remains the strongest simple baseline.\n" +
  "5. PL-only pretraining is uniformly weak. LP-only pretraining specializes in LP and transfers poorly to PL/NM.\n" +
  "6. The final 3-graph run changes graph count, task count, merging/joint training, and augmentation together; its gain cannot be assigned to one design choice.\n" +
  "Important: there are no repeated seeds or uncertainty estimates here, so tiny differences should not be interpreted as statistically meaningful."
]];
dashboard.getRange("A11:N15").format = {
  fill: "#FAFBFC",
  font: { color: darkGray, size: 11 },
  wrapText: true,
  verticalAlignment: "top",
  borders: { preset: "outside", style: "thin", color: midGray },
};

// Formula-backed chart helper ranges, placed below the dashboard.
dashboard.getRange("A40:C40").values = [["Approach", "Balanced score", "All-test mean"]];
for (let i = 0; i < 10; i++) {
  const dashR = 41 + i;
  const metricR = byBalanced[i].sourceIndex + 2;
  dashboard.getRange(`A${dashR}`).values = [[shortLabel(byBalanced[i].trainId)]];
  dashboard.getRange(`B${dashR}:C${dashR}`).formulas = [[
    `='Metrics'!G${metricR}`,
    `='Metrics'!H${metricR}`,
  ]];
}
dashboard.getRange("E40:H40").values = [["Approach", "LP", "NM", "PL"]];
for (let i = 0; i < 6; i++) {
  const dashR = 41 + i;
  const metricR = byBalanced[i].sourceIndex + 2;
  dashboard.getRange(`E${dashR}`).values = [[shortLabel(byBalanced[i].trainId)]];
  dashboard.getRange(`F${dashR}:H${dashR}`).formulas = [[
    `='Metrics'!D${metricR}`,
    `='Metrics'!E${metricR}`,
    `='Metrics'!F${metricR}`,
  ]];
}
dashboard.getRange("J40:L40").values = [["Approach", "Mean test rank", "Wins"]];
for (let i = 0; i < 8; i++) {
  const dashR = 41 + i;
  const metricR = byMeanRank[i].sourceIndex + 2;
  dashboard.getRange(`J${dashR}:L${dashR}`).formulas = [[
    `='Metrics'!A${metricR}`,
    `='Metrics'!J${metricR}`,
    `='Metrics'!K${metricR}`,
  ]];
}
dashboard.getRange("B41:C61").format.numberFormat = "0.000";
dashboard.getRange("F41:H46").format.numberFormat = "0.000";
dashboard.getRange("K41:K48").format.numberFormat = "0.00";
dashboard.getRange("L41:L48").format.numberFormat = "0";

const rankingChart = dashboard.charts.add("bar", dashboard.getRange("A40:B50"));
rankingChart.title = "Top 10 balanced scores; #1 and #2 differ by only 0.0002";
rankingChart.hasLegend = false;
rankingChart.yAxis = { numberFormatCode: "0.000", min: 0.5, max: 0.85 };
rankingChart.setPosition("A17", "G38");

const profileChart = dashboard.charts.add("bar", dashboard.getRange("E40:H46"));
profileChart.title = "Top approaches have different task-family profiles";
profileChart.hasLegend = true;
profileChart.yAxis = { numberFormatCode: "0.000", min: 0.5, max: 1.0 };
profileChart.setPosition("H17", "N38");

dashboard.getRange("A40:L61").format.font = { size: 9, color: darkGray };
dashboard.getRange("A40:L40").format = {
  fill: paleGray,
  font: { bold: true, color: navy },
};
dashboard.getRange("A:A").format.columnWidth = 15;
dashboard.getRange("B:N").format.columnWidth = 13;
dashboard.getRange("5:8").format.rowHeight = 24;
dashboard.getRange("11:15").format.rowHeight = 27;
dashboard.freezePanes.freezeRows(3);

// Sorted full-suite heatmap.
heatmap.getRange("A1:O2").merge();
heatmap.getRange("A1").values = [["Full Test-Suite Heatmap (sorted by balanced score)"]];
heatmap.getRange("A1:O2").format = {
  fill: navy,
  font: { bold: true, color: white, size: 18 },
  verticalAlignment: "center",
};
heatmap.getRange("A3:O3").values = [[
  "train_id",
  ...testHeaders,
  "Balanced",
  "Mean rank",
]];
heatmap.getRange("A3:O3").format = {
  fill: blue,
  font: { bold: true, color: white },
  wrapText: true,
  verticalAlignment: "center",
};
for (let i = 0; i < byBalanced.length; i++) {
  const heatR = i + 4;
  const rawR = rowByTrainId.get(byBalanced[i].trainId);
  const metricR = byBalanced[i].sourceIndex + 2;
  heatmap.getRange(`A${heatR}:M${heatR}`).formulas = [[
    `='Raw Data'!A${rawR}`,
    ...Array.from({ length: 12 }, (_, j) => {
      const col = String.fromCharCode("B".charCodeAt(0) + j);
      return `='Raw Data'!${col}${rawR}`;
    }),
  ]];
  heatmap.getRange(`N${heatR}:O${heatR}`).formulas = [[
    `='Metrics'!G${metricR}`,
    `='Metrics'!J${metricR}`,
  ]];
}
heatmap.getRange("B4:N24").format.numberFormat = "0.000";
heatmap.getRange("O4:O24").format.numberFormat = "0.00";
heatmap.getRange("B4:M24").conditionalFormats.add("colorScale", {
  colors: ["#F8D7DA", "#FFF3CD", "#D1E7DD"],
  thresholds: [
    { type: "num", value: 0.48 },
    { type: "num", value: 0.75 },
    { type: "num", value: 1.0 },
  ],
});
heatmap.getRange("N4:N24").conditionalFormats.add("dataBar", { color: blue, gradient: true });
heatmap.getRange("O4:O24").conditionalFormats.add("colorScale", {
  colors: ["#D1E7DD", "#FFF3CD", "#F8D7DA"],
  thresholds: ["min", { type: "percentile", value: 50 }, "max"],
});
heatmap.getRange("A3:O24").format.borders = {
  insideHorizontal: { style: "thin", color: "#E8ECEF" },
};
heatmap.freezePanes.freezeRows(3);
heatmap.freezePanes.freezeColumns(1);
heatmap.getRange("A:O").format.autofitColumns();
heatmap.getRange("A:A").format.columnWidth = 39;
heatmap.getRange("B:M").format.columnWidth = 17;
heatmap.getRange("N:O").format.columnWidth = 12;
heatmap.getRange("3:3").format.rowHeight = 45;

// Method notes.
readme.getRange("A1:H2").merge();
readme.getRange("A1").values = [["How to read this workbook"]];
readme.getRange("A1:H2").format = {
  fill: navy,
  font: { bold: true, color: white, size: 18 },
  verticalAlignment: "center",
};
readme.getRange("A4:B10").values = [
  ["View", "Purpose"],
  ["Dashboard", "Decision summary plus two charts: overall ranking and LP/NM/PL transfer profiles."],
  ["Metrics", "Formula-driven per-approach aggregates, rank consistency, wins, and regret."],
  ["Heatmap", "Exact per-test pattern. Use it to see specialization and catastrophic transfer failures."],
  ["Raw Data", "Original TSV values, unchanged."],
  ["Balanced score", "Average of LP mean, NM mean, and PL mean. Each task family gets equal weight."],
  ["Mean test rank", "Average rank over the 12 test columns. Lower is better; robust to column difficulty."],
];
readme.getRange("A4:B4").format = {
  fill: blue,
  font: { bold: true, color: white },
};
readme.getRange("A4:B10").format.borders = {
  insideHorizontal: { style: "thin", color: "#E8ECEF" },
  outside: { style: "thin", color: midGray },
};
readme.getRange("D4:H4").merge();
readme.getRange("D4").values = [["Recommended decision rule"]];
readme.getRange("D4:H4").format = {
  fill: teal,
  font: { bold: true, color: white },
};
readme.getRange("D5:H10").merge();
readme.getRange("D5").values = [[
  "Do not select a winner from one score alone.\n\n" +
  "Primary: balanced score (equal task-family weight).\n" +
  "Tie-breaker: mean test rank and mean regret.\n" +
  "Constraint check: inspect LP/NM/PL means for the deployment mix.\n" +
  "Statistical gate: rerun finalists across multiple seeds before declaring small gaps real.\n\n" +
  "Current finalists: midterm+nm → covid19_twitter+nm; merged_ukr_covid_midterm_aug_10k; covid19_twitter+nm."
]];
readme.getRange("D5:H10").format = {
  fill: paleTeal,
  font: { color: darkGray, size: 11 },
  wrapText: true,
  verticalAlignment: "top",
  borders: { preset: "outside", style: "thin", color: midGray },
};
readme.getRange("A12:H12").merge();
readme.getRange("A12").values = [["Interpretation caveats"]];
readme.getRange("A12:H12").format = {
  fill: orange,
  font: { bold: true, color: white },
};
readme.getRange("A13:H17").merge();
readme.getRange("A13").values = [[
  "• One result per approach/test means there is no estimate of run-to-run variance.\n" +
  "• Test columns differ in difficulty; raw averages favor families with more columns, which is why the dashboard includes balanced score and mean rank.\n" +
  "• Some tests overlap pretraining graph domains. For a strict out-of-domain claim, add a flag for graph overlap and report in-domain vs out-of-domain aggregates separately.\n" +
  "• The experiment matrix is not factorial: the 3-graph result simultaneously changes graph count, task count, merge/joint strategy, and augmentation. Add matched ablations before claiming which ingredient caused the gain.\n" +
  "• ukr_rus_suspended+pl is near chance for every approach and compresses worst-case statistics; inspect it separately rather than using minimum score as the main selector."
]];
readme.getRange("A13:H17").format = {
  fill: "#FFF8E7",
  font: { color: darkGray, size: 11 },
  wrapText: true,
  verticalAlignment: "top",
  borders: { preset: "outside", style: "thin", color: midGray },
};
readme.getRange("A:B").format.autofitColumns();
readme.getRange("A:A").format.columnWidth = 20;
readme.getRange("B:B").format.columnWidth = 62;
readme.getRange("C:C").format.columnWidth = 3;
readme.getRange("D:H").format.columnWidth = 15;
readme.getRange("5:10").format.rowHeight = 28;
readme.getRange("13:17").format.rowHeight = 28;

await fs.mkdir(outputDir, { recursive: true });

const inspect = await workbook.inspect({
  kind: "table",
  range: "Metrics!A1:L8",
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 12,
  maxChars: 10000,
});
console.log(inspect.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

for (const [sheetName, fileName, range] of [
  ["Dashboard", "dashboard_preview.png", "A1:N38"],
  ["Heatmap", "heatmap_preview.png", "A1:O24"],
  ["Metrics", "metrics_preview.png", "A1:L22"],
  ["Raw Data", "raw_preview.png", "A1:O22"],
  ["README", "readme_preview.png", "A1:H17"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1.2, format: "png" });
  await fs.writeFile(`${outputDir}/${fileName}`, new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(outputPath);
