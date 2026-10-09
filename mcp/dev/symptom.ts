/**
 * Dev-set check for long symptom questions. Not an eval.
 * Run: npx vite-node dev/symptom.ts
 */
import { readFileSync } from "node:fs";
import { markPackagedIndex, primeIndex } from "../src/bm25.js";
import { listManuals } from "../src/catalog.js";
import { readPrebuilt } from "../src/index-store.js";
import { searchManuals } from "../src/search.js";

type DevCase = { id: string; query: string; hope: string[] };

const cases = (JSON.parse(readFileSync(new URL("./symptom-cases.json", import.meta.url), "utf8")) as { search: DevCase[] }).search;
const manuals = listManuals().filter((manual) => manual.searchable).map((manual) => manual.id);
const groups = manuals.map((id) => {
  const docs = readPrebuilt(id) ?? [];
  if (docs.length > 0) {
    markPackagedIndex(docs);
    primeIndex(docs);
  }
  return docs;
});

function hit(query: string) {
  return searchManuals(groups, query, 8);
}

let top1 = 0;
let top3 = 0;
const misses: string[] = [];
for (const item of cases) {
  const hits = hit(item.query);
  const urls = hits.slice(0, 3).map((row) => row.url);
  const at1 = item.hope.some((hope) => urls[0]?.includes(hope));
  const at3 = item.hope.some((hope) => urls.some((url) => url.includes(hope)));
  if (at1) top1 += 1;
  if (at3) top3 += 1;
  if (!at1) misses.push(`${at3 ? "top3" : "miss"} ${item.id} -> ${(urls[0] ?? "(none)").replace("https://developer.d-robotics.cc", "")}`);
}
process.stdout.write(`symptom top1 ${top1}/${cases.length} top3 ${top3}/${cases.length}\n`);
if (misses.length > 0) process.stdout.write(`${misses.join("\n")}\n`);

const samples: number[] = [];
for (let round = 0; round < 3; round += 1) {
  for (const item of cases) {
    const started = performance.now();
    hit(item.query);
    samples.push(performance.now() - started);
  }
}
samples.sort((a, b) => a - b);
const p95 = samples[Math.min(samples.length - 1, Math.floor(samples.length * 0.95))] ?? 0;
process.stdout.write(`warm p95 ${p95.toFixed(2)}ms n=${samples.length} max ${samples.at(-1)?.toFixed(2)}ms\n`);
