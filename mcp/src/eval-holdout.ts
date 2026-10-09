import { readFile, mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { execSync } from "node:child_process";
import { fetchText } from "./http.js";
import { searchDocs } from "./service.js";

/**
 * Blind holdout scorer for search_docs.
 *
 * Relevant cases (noGoodMatch false): hit@1, hit@3, and MRR against expectedUrls.
 * A hit is an exact path match after stripping origin, hash, .html, and a trailing slash.
 * Rank is computed on the docs search list (limit 8), the same call an agent makes
 * with no manual filter.
 *
 * noGoodMatch cases have no acceptable page. The system abstains when the hit list
 * is empty, or when the payload sets abstain / noMatch / no_good_match, or when
 * warnings/guidance explicitly say there is no matching document. Precision is
 * TP/(TP+FP) over abstentions. It is null when the system never abstains.
 */

export type HoldoutCase = {
  id: string;
  query: string;
  board: string;
  noGoodMatch: boolean;
  expectedUrls: string[];
  why: string;
};

type SearchPayload = {
  hits: Array<{ title: string; url: string; score: number; manual?: string }>;
  warnings?: string[];
  guidance?: string;
  abstain?: boolean;
  noMatch?: boolean;
  no_good_match?: boolean;
};

const root = dirname(fileURLToPath(import.meta.url));
const defaultCases = join(root, "..", "eval", "holdout-cases.json");

function arg(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  if (index === -1) return undefined;
  return process.argv[index + 1];
}

export function canonicalPath(url: string): string {
  const noHash = url.split("#")[0] ?? url;
  const path = noHash.replace(/^https?:\/\/[^/]+/i, "");
  return path.replace(/\.html$/i, "").replace(/\/+$/, "").toLowerCase();
}

export function urlsMatch(hitUrl: string, expectedUrl: string): boolean {
  const hit = canonicalPath(hitUrl);
  const expected = canonicalPath(expectedUrl);
  return hit.length > 1 && hit === expected;
}

export function firstRank(hitUrls: string[], expectedUrls: string[]): number | null {
  for (let i = 0; i < hitUrls.length; i += 1) {
    if (expectedUrls.some((expected) => urlsMatch(hitUrls[i] ?? "", expected))) return i + 1;
  }
  return null;
}

const ABSTAIN_TEXT = /没有(?:找到|检索到)?(?:合适|相关|匹配)的?(?:文档|页面|手册)|未找到相关|no good match|no relevant (?:document|page|hit)|nothing relevant/i;

export function systemAbstained(payload: SearchPayload): boolean {
  if (payload.abstain === true || payload.noMatch === true || payload.no_good_match === true) return true;
  if (payload.hits.length === 0) return true;
  const text = [...(payload.warnings ?? []), payload.guidance ?? ""].join("\n");
  return ABSTAIN_TEXT.test(text);
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[index] ?? 0;
}

function gitHead(): string {
  try {
    return execSync("git rev-parse HEAD", { encoding: "utf8" }).trim();
  } catch {
    return "unknown";
  }
}

async function main() {
  const casesPath = arg("--cases") ?? defaultCases;
  const label = arg("--label") ?? "unlabeled";
  const outPath = arg("--out") ?? join(root, "..", "eval", "results", `holdout-${label}.json`);
  const file = JSON.parse(await readFile(casesPath, "utf8")) as { cases: HoldoutCase[] };
  const cases = file.cases;
  if (cases.length !== 50) {
    throw new Error(`expected 50 holdout cases, found ${cases.length}`);
  }

  await searchDocs({ query: "rdk", limit: 1 }, fetchText);

  const rows = [];
  for (const item of cases) {
    const started = performance.now();
    const search = (await searchDocs({ query: item.query, limit: 8 }, fetchText)) as SearchPayload;
    const latencyMs = performance.now() - started;
    const hitUrls = search.hits.map((hit) => hit.url);
    const rank = item.noGoodMatch ? null : firstRank(hitUrls, item.expectedUrls);
    const abstained = systemAbstained(search);
    rows.push({
      id: item.id,
      board: item.board,
      noGoodMatch: item.noGoodMatch,
      query: item.query,
      expectedUrls: item.expectedUrls,
      why: item.why,
      rank,
      hitAt1: !item.noGoodMatch && rank === 1,
      hitAt3: !item.noGoodMatch && rank !== null && rank <= 3,
      reciprocalRank: !item.noGoodMatch ? (rank === null ? 0 : 1 / rank) : null,
      abstained,
      latencyMs: Math.round(latencyMs * 10) / 10,
      top: search.hits.slice(0, 3).map((hit) => ({
        title: hit.title,
        url: hit.url,
        score: Math.round(hit.score * 100) / 100,
        manual: hit.manual,
      })),
      warnings: search.warnings ?? [],
    });
    const mark = item.noGoodMatch
      ? abstained
        ? "ABSTAIN"
        : "ACCEPT"
      : rank === null
        ? "MISS"
        : `R${rank}`;
    console.log(`${mark}\t${item.id}\t${search.hits[0]?.url ?? "(empty)"}`);
  }

  const relevant = rows.filter((row) => !row.noGoodMatch);
  const negatives = rows.filter((row) => row.noGoodMatch);
  const abstentions = rows.filter((row) => row.abstained);
  const trueAbstain = negatives.filter((row) => row.abstained).length;
  const falseAbstain = relevant.filter((row) => row.abstained).length;
  const precisionDenom = trueAbstain + falseAbstain;
  const latencies = rows.map((row) => row.latencyMs).sort((a, b) => a - b);
  const mean = latencies.reduce((sum, value) => sum + value, 0) / latencies.length;

  const metrics = {
    relevant: relevant.length,
    noGoodMatch: negatives.length,
    hitAt1: relevant.filter((row) => row.hitAt1).length / relevant.length,
    hitAt3: relevant.filter((row) => row.hitAt3).length / relevant.length,
    mrr: relevant.reduce((sum, row) => sum + (row.reciprocalRank ?? 0), 0) / relevant.length,
    noGoodMatchPrecision: precisionDenom === 0 ? null : trueAbstain / precisionDenom,
    noGoodMatchRecall: trueAbstain / negatives.length,
    abstentions: abstentions.length,
    falseAccepts: negatives.filter((row) => !row.abstained).map((row) => row.id),
    latencyMs: {
      mean: Math.round(mean * 10) / 10,
      p50: Math.round(percentile(latencies, 50) * 10) / 10,
      p95: Math.round(percentile(latencies, 95) * 10) / 10,
    },
  };

  const report = {
    label,
    commit: gitHead(),
    casesPath,
    searchedAt: new Date().toISOString(),
    limit: 8,
    metrics,
    misses: relevant
      .filter((row) => !row.hitAt1)
      .map((row) => ({
        id: row.id,
        rank: row.rank,
        query: row.query,
        expectedUrls: row.expectedUrls,
        top: row.top,
      })),
    noGoodMatchAccepted: negatives
      .filter((row) => !row.abstained)
      .map((row) => ({ id: row.id, query: row.query, why: row.why, top: row.top })),
    cases: rows,
  };

  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(`\n${label} ${JSON.stringify(metrics)}`);
  console.log(`wrote ${outPath}`);
}

main().catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
