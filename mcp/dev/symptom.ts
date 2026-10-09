/**
 * Dev-set check for long symptom questions. Not an eval and not held-out.
 * Run: npx vite-node dev/symptom.ts
 */
import { readFileSync } from "node:fs";
import { markPackagedIndex, primeIndex } from "../src/bm25.js";
import { listManuals } from "../src/catalog.js";
import { denseHits, warmHybrid } from "../src/hybrid.js";
import { fetchText } from "../src/http.js";
import { readPrebuilt } from "../src/index-store.js";
import { searchManuals, searchManualsHybrid } from "../src/search.js";
import { searchDocs } from "../src/service.js";

type DevCase = { id: string; query: string; hope: string[] };
type ParaCase = { id: string; query: string; manual?: string; expectNoGoodMatch?: boolean; board?: "x3" | "x5" | "s100" | "s600" };

const cases = (JSON.parse(readFileSync(new URL("./symptom-cases.json", import.meta.url), "utf8")) as { search: DevCase[] }).search;
const paraphrase = (
  JSON.parse(readFileSync(new URL("../eval/paraphrase-cases.json", import.meta.url), "utf8")) as { search: ParaCase[] }
).search;
const manuals = listManuals().filter((manual) => manual.searchable).map((manual) => manual.id);
const groups = manuals.map((id) => {
  const docs = readPrebuilt(id) ?? [];
  if (docs.length > 0) {
    markPackagedIndex(docs);
    primeIndex(docs);
  }
  return docs;
});

function summarize(label: string, rows: Array<{ id: string; at1: boolean; at3: boolean; top: string }>, samples: number[]): void {
  const top1 = rows.filter((row) => row.at1).length;
  const top3 = rows.filter((row) => row.at3).length;
  const misses = rows.filter((row) => !row.at1).map((row) => `${row.at3 ? "top3" : "miss"} ${row.id} -> ${row.top}`);
  samples.sort((a, b) => a - b);
  const p95 = samples[Math.min(samples.length - 1, Math.floor(samples.length * 0.95))] ?? 0;
  process.stdout.write(`${label} top1 ${top1}/${rows.length} top3 ${top3}/${rows.length} warm p95 ${p95.toFixed(2)}ms n=${samples.length} max ${samples.at(-1)?.toFixed(2)}ms\n`);
  if (misses.length > 0) process.stdout.write(`${misses.join("\n")}\n`);
}

function judge(query: string, urls: string[], hope: string[]): { at1: boolean; at3: boolean; top: string } {
  const at1 = hope.some((hopeUrl) => urls[0]?.includes(hopeUrl));
  const at3 = hope.some((hopeUrl) => urls.some((url) => url.includes(hopeUrl)));
  return { at1, at3, top: (urls[0] ?? "(none)").replace("https://developer.d-robotics.cc", "") };
}

const bmRows = cases.map((item) => {
  const hits = searchManuals(groups, item.query, 8);
  return { id: item.id, ...judge(item.query, hits.map((hit) => hit.url), item.hope) };
});
const bmSamples: number[] = [];
for (let round = 0; round < 3; round += 1) {
  for (const item of cases) {
    const started = performance.now();
    searchManuals(groups, item.query, 8);
    bmSamples.push(performance.now() - started);
  }
}
summarize("bm25", bmRows, bmSamples);

process.env.RDK_DOCS_HYBRID = "1";
process.env.RDK_DOCS_OFFLINE = "1";
const coldStarted = performance.now();
const warmed = await warmHybrid();
process.stdout.write(`hybrid cold ${warmed ? (performance.now() - coldStarted).toFixed(0) : "failed"}ms\n`);

const firstStarted = performance.now();
await searchManualsHybrid(groups, cases[0]?.query ?? "摄像头黑屏", 8);
process.stdout.write(`hybrid first search ${(performance.now() - firstStarted).toFixed(2)}ms\n`);

const hyRows = [];
for (const item of cases) {
  const hits = await searchManualsHybrid(groups, item.query, 8);
  hyRows.push({ id: item.id, ...judge(item.query, hits.map((hit) => hit.url), item.hope) });
}
const hySamples: number[] = [];
for (let round = 0; round < 3; round += 1) {
  for (const item of cases) {
    const started = performance.now();
    await searchManualsHybrid(groups, item.query, 8);
    hySamples.push(performance.now() - started);
  }
}
summarize("hybrid", hyRows, hySamples);

const ood = paraphrase.filter((item) => item.expectNoGoodMatch);
process.stdout.write("abstain id bm25 hybrid cosine\n");
const devCos: number[] = [];
for (const item of cases) {
  const dense = await denseHits(groups, item.query, 1);
  devCos.push(dense.topCosine);
}
devCos.sort((a, b) => a - b);
for (const item of ood) {
  delete process.env.RDK_DOCS_HYBRID;
  const bm = await searchDocs({ query: item.query, manual: item.manual, limit: 5 }, fetchText);
  process.env.RDK_DOCS_HYBRID = "1";
  const hy = await searchDocs({ query: item.query, manual: item.manual, limit: 5 }, fetchText);
  const dense = await denseHits(groups, item.query, 1);
  process.stdout.write(`${item.id} bm25=${bm.noGoodMatch} hybrid=${hy.noGoodMatch} cosine=${dense.topCosine.toFixed(3)} top=${(hy.hits[0]?.url ?? "(none)").replace("https://developer.d-robotics.cc", "")}\n`);
}
const pct = (p: number) => devCos[Math.min(devCos.length - 1, Math.floor(devCos.length * p))] ?? 0;
process.stdout.write(`dev cosine min ${devCos[0]?.toFixed(3)} p50 ${pct(0.5).toFixed(3)} p10 ${pct(0.1).toFixed(3)} max ${devCos.at(-1)?.toFixed(3)}\n`);

let flipped = 0;
for (const item of paraphrase) {
  if (item.expectNoGoodMatch) continue;
  delete process.env.RDK_DOCS_HYBRID;
  const bm = await searchDocs({ query: item.query, manual: item.manual, board: item.board, limit: 5 }, fetchText);
  process.env.RDK_DOCS_HYBRID = "1";
  const hy = await searchDocs({ query: item.query, manual: item.manual, board: item.board, limit: 5 }, fetchText);
  if (bm.noGoodMatch !== hy.noGoodMatch) {
    flipped += 1;
    process.stdout.write(`abstain-flip ${item.id} bm25=${bm.noGoodMatch} hybrid=${hy.noGoodMatch}\n`);
  }
}
process.stdout.write(`paraphrase in-domain abstain flips ${flipped}\n`);
