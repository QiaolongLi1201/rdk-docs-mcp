import { RETRIEVAL_ALIASES } from "./aliases.js";
import { absentCommandToken, contextBoards, rankCorpora, type RankOptions } from "./bm25.js";
import type { IndexedDoc, ResultBoard, SearchHit } from "./types.js";

export type { RankOptions };

function aliasNote(query: string, hit: SearchHit): SearchHit {
  const raw = query.trim().toLowerCase();
  const aliases = RETRIEVAL_ALIASES[raw];
  if (!aliases) return hit;
  const blob = `${hit.title} ${hit.url} ${hit.snippet}`.toLowerCase();
  if (blob.includes(raw)) return hit;
  const via = aliases.find((alias) => blob.includes(alias.toLowerCase()));
  if (!via) return hit;
  return {
    ...hit,
    matchedVia: "alias",
    snippet: `No indexed page contains \`${raw}\`. Closest documented page (${via}). ${hit.snippet}`.slice(0, 280),
  };
}

export function groupHits(hits: SearchHit[]): Array<{ board: ResultBoard; hits: SearchHit[] }> {
  const order: ResultBoard[] = [];
  const map = new Map<ResultBoard, SearchHit[]>();
  for (const hit of hits) {
    const board: ResultBoard = hit.board ?? "agnostic";
    const list = map.get(board);
    if (list) list.push(hit);
    else {
      map.set(board, [hit]);
      order.push(board);
    }
  }
  return order.map((board) => ({ board, hits: map.get(board) ?? [] }));
}

function diversityKey(hit: SearchHit): string {
  return hit.board ?? "agnostic";
}

/**
 * When the query names no board, surface the best hit of each board first,
 * then the rest in score order. The global best stays first.
 */
function diversifyByBoard(hits: SearchHit[]): SearchHit[] {
  if (hits.length < 2) return hits;
  const groups = new Map<string, SearchHit[]>();
  for (const hit of hits) {
    const key = diversityKey(hit);
    const list = groups.get(key);
    if (list) list.push(hit);
    else groups.set(key, [hit]);
  }
  if (groups.size <= 1) return hits;
  const keys = [...groups.keys()].sort((a, b) => (groups.get(b)?.[0]?.score ?? 0) - (groups.get(a)?.[0]?.score ?? 0));
  const seen = new Set<SearchHit>();
  const first: SearchHit[] = [];
  for (const key of keys) {
    const hit = groups.get(key)?.[0];
    if (!hit) continue;
    first.push(hit);
    seen.add(hit);
  }
  return [...first, ...hits.filter((hit) => !seen.has(hit))];
}

/** In the top window, defer extras that share a key so one page cannot fill it. */
function capRepeats(hits: SearchHit[], limit: number, keyOf: (hit: SearchHit) => string, maxPer: number): SearchHit[] {
  const counts = new Map<string, number>();
  const head: SearchHit[] = [];
  const rest: SearchHit[] = [];
  for (const hit of hits) {
    const key = keyOf(hit);
    const seen = counts.get(key) ?? 0;
    if (head.length < limit && seen >= maxPer) {
      rest.push(hit);
      continue;
    }
    counts.set(key, seen + 1);
    head.push(hit);
  }
  return [...head, ...rest];
}

function cjkChars(query: string): number {
  let count = 0;
  for (let i = 0; i < query.length; i += 1) {
    const code = query.charCodeAt(i);
    if (code >= 0x4e00 && code <= 0x9fff) count += 1;
  }
  return count;
}

function orderHits(docsGroups: IndexedDoc[][], query: string, limit: number, options: RankOptions): SearchHit[] {
  const ranked = rankCorpora(docsGroups, query, options);
  const unscoped = contextBoards(query, options).length === 0;
  const longQuery = cjkChars(query) >= 24;
  // Short queries keep one hit per board in front. A long question stays in
  // score order and only defers a third section from the same page.
  const diverse = unscoped && !longQuery ? diversifyByBoard(ranked) : ranked;
  const ordered =
    unscoped && longQuery ? capRepeats(diverse, limit, (hit) => hit.url.split("#")[0] ?? hit.url, 2) : diverse;
  return ordered.slice(0, limit).map((hit) => aliasNote(query, hit));
}

export function rankHits(docs: IndexedDoc[], query: string, limit: number, options: RankOptions = {}): SearchHit[] {
  return orderHits([docs], query, limit, options);
}

/** Score each manual's stable doc array on its own cached index, then merge. */
export function searchManuals(
  groups: IndexedDoc[][],
  query: string,
  limit: number,
  options: RankOptions = {},
): SearchHit[] {
  return orderHits(groups, query, limit, options);
}

export function matchQuality(
  hits: SearchHit[],
  groups: IndexedDoc[][] = [],
  query = "",
): {
  noGoodMatch: boolean;
  matchQuality: "good" | "weak" | "none";
  confidence: number;
} {
  const top = hits[0];
  const confidence = top?.confidence ?? 0;
  if (!top || top.score <= 0) {
    const missing = query ? absentCommandToken(groups, query) : false;
    return missing
      ? { noGoodMatch: true, matchQuality: "weak", confidence: 0 }
      : { noGoodMatch: false, matchQuality: "none", confidence: 0 };
  }
  if (top.quality === "weak") return { noGoodMatch: true, matchQuality: "weak", confidence };
  return { noGoodMatch: false, matchQuality: "good", confidence };
}
