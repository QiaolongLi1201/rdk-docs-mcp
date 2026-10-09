import { CONCEPT_SYNONYMS } from "./aliases.js";
import { GLOSSARY_ALIASES } from "./glossary-aliases.js";
import type { IndexedDoc } from "./types.js";

/**
 * Title scorer fused with BM25 by reciprocal rank. It scores query tokens
 * against the title and body. It does not look at URL path classes, page
 * names, or which manual section a link points at.
 */

const CJK_STOPWORDS = ["怎么样", "怎么", "怎样", "如何", "什么", "哪些", "哪里", "是否", "多少", "请问", "帮我", "一下", "可不可以", "能不能", "有没有"];
const CJK_STOP_CHARS = new Set(["的", "了", "吗", "呢", "啊", "吧", "把", "是", "有", "个", "和", "或", "在", "给", "去", "到", "太", "很"]);

/** English function words. They are not retrieval terms. */
const LEX_STOP = new Set([
  "a",
  "an",
  "the",
  "and",
  "or",
  "not",
  "to",
  "of",
  "in",
  "on",
  "for",
  "with",
  "from",
  "into",
  "onto",
  "by",
  "at",
  "as",
  "is",
  "are",
  "was",
  "were",
  "be",
  "been",
  "do",
  "does",
  "did",
  "can",
  "will",
  "how",
  "what",
  "when",
  "where",
  "which",
  "who",
  "why",
  "this",
  "that",
  "these",
  "those",
  "i",
  "me",
  "my",
  "we",
  "you",
  "your",
  "it",
  "if",
  "check",
  "please",
  "just",
  "about",
  "than",
  "then",
  "there",
  "here",
]);

/** Generic words that collide across manuals. A title made of these is not an answer. */
const WEAK_TITLE = new Set([
  "docker",
  "error",
  "errors",
  "static",
  "install",
  "network",
  "system",
  "default",
  "config",
  "board",
  "linux",
  "image",
  "device",
  "type",
  "file",
  "user",
  "root",
  "info",
  "test",
  "data",
]);

const BASE_SYNONYMS: Record<string, string[]> = {
  flash: ["烧录", "burn"],
  burn: ["烧录"],
  wifi: ["wi-fi", "无线"],
  install: ["安装"],
};

function mergeSynonyms(...maps: Array<Record<string, string[]>>): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  for (const map of maps) {
    for (const [key, extras] of Object.entries(map)) {
      out[key] = [...new Set([...(out[key] ?? []), ...extras])];
    }
  }
  return out;
}

function activeSynonyms(): Record<string, string[]> {
  const concept = (process.env.RDK_ABLATE ?? "").split(",").includes("synonym") ? {} : CONCEPT_SYNONYMS;
  return mergeSynonyms(BASE_SYNONYMS, GLOSSARY_ALIASES, concept);
}

const CJK_RUN = /[\u4e00-\u9fff]+/g;

/** Grammatical fillers. Removed only when cutting phrases out of a long clause. */
const PHRASE_PARTICLES = ["之后", "之前", "以后", "然后", "已经", "可是", "但是", "因为", "所以", "这个", "那个", "一直", "非常", "特别", "还是"];

function stripParticles(segment: string): string {
  let text = segment;
  for (const particle of PHRASE_PARTICLES) text = text.replaceAll(particle, "");
  return text;
}

function cjkCount(query: string): number {
  let count = 0;
  for (let i = 0; i < query.length; i += 1) {
    const code = query.charCodeAt(i);
    if (code >= 0x4e00 && code <= 0x9fff) count += 1;
  }
  return count;
}

export type LexicalPlan = {
  tokens: string[];
  matchers: Array<{ token: string; test: (text: string) => boolean }>;
  /** Longer CJK windows from a long clause. They add points and do not dilute coverage. */
  phrases: Array<{ token: string; test: (text: string) => boolean; title: number }>;
};

function cjkRuns(query: string): string[] {
  const segments: string[] = [];
  for (const match of query.matchAll(CJK_RUN)) {
    let run = match[0];
    for (const stop of CJK_STOPWORDS) run = run.replaceAll(stop, "\u0000");
    run = [...run].map((ch) => (CJK_STOP_CHARS.has(ch) ? "\u0000" : ch)).join("");
    for (const segment of run.split("\u0000")) {
      if (segment.length >= 2) segments.push(segment);
    }
  }
  return segments;
}

function cjkBigrams(query: string): string[] {
  const grams: string[] = [];
  for (const segment of cjkRuns(query)) {
    for (let i = 0; i + 2 <= segment.length; i += 1) grams.push(segment.slice(i, i + 2));
  }
  return grams;
}

/**
 * 3- and 4-character windows from a long CJK clause. Stop-characters split
 * the clause for matching, but the gate is the raw run, so "左上角" still
 * counts after "在". Short queries never take this path.
 */
function cjkPhrases(query: string): string[] {
  const buckets: string[][] = [];
  for (const match of query.matchAll(CJK_RUN)) {
    const raw = match[0];
    if (!raw || raw.length < 10) continue;
    let run = raw;
    for (const stop of CJK_STOPWORDS) run = run.replaceAll(stop, "\u0000");
    run = [...run].map((ch) => (CJK_STOP_CHARS.has(ch) ? "\u0000" : ch)).join("");
    for (const segment of run.split("\u0000")) {
      const glued = stripParticles(segment);
      const source = glued.length >= 3 ? glued : segment;
      if (source.length < 3) continue;
      const windows: string[] = [];
      const seenLocal = new Set<string>();
      const max = Math.min(4, source.length);
      for (let len = max; len >= 3; len -= 1) {
        for (let i = 0; i + len <= source.length; i += 1) {
          const phrase = source.slice(i, i + len);
          if (seenLocal.has(phrase)) continue;
          seenLocal.add(phrase);
          windows.push(phrase);
        }
      }
      windows.sort((a, b) => b.length - a.length);
      buckets.push(windows.slice(0, 6));
    }
  }
  if (cjkCount(query) >= 24) {
    for (const segment of cjkRuns(query)) {
      if (segment.length < 4 || segment.length >= 10) continue;
      const source = stripParticles(segment);
      const phrase = (source.length >= 4 ? source : segment).slice(-4);
      if (phrase.length < 4) continue;
      buckets.push([phrase]);
    }
  }
  const phrases: string[] = [];
  const seen = new Set<string>();
  for (let slot = 0; slot < 6 && phrases.length < 16; slot += 1) {
    for (const bucket of buckets) {
      const phrase = bucket[slot];
      if (!phrase || seen.has(phrase)) continue;
      seen.add(phrase);
      phrases.push(phrase);
      if (phrases.length >= 16) return phrases;
    }
  }
  return phrases;
}

export function lexicalTokens(query: string): string[] {
  const seen = new Set<string>();
  const lowered = query.trim().toLowerCase();
  const add = (token: string) => {
    if (!token || LEX_STOP.has(token)) return;
    seen.add(token);
  };
  for (const match of lowered.matchAll(/[a-z][a-z0-9_.-]*|\d+[a-z][a-z0-9_.-]*|\d+/g)) add(match[0]);
  for (const match of lowered.matchAll(/(\d+)[\s-]+([a-z][a-z0-9_.-]*)/g)) add(`${match[1]}${match[2]}`);
  for (const gram of cjkBigrams(lowered)) add(gram);
  for (const [key, extras] of Object.entries(activeSynonyms())) {
    if (!seen.has(key)) continue;
    for (const item of extras) add(item.toLowerCase());
  }
  if (seen.size === 0) {
    for (const part of lowered.split(/\s+/).filter(Boolean)) add(part);
  }
  return [...seen];
}

function buildMatcher(token: string): { token: string; test: (text: string) => boolean } {
  if (!/^[a-z0-9][a-z0-9_.-]*$/.test(token)) return { token, test: (text) => text.includes(token) };
  const escaped = token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern =
    token.length <= 3 ? new RegExp(`(?<![a-z])${escaped}(?![a-z])`) : new RegExp(`(?<![a-z0-9])${escaped}`);
  return { token, test: (text) => pattern.test(text) };
}

const planCache = new Map<string, LexicalPlan>();

export function prepareLexical(query: string): LexicalPlan {
  const cached = planCache.get(query);
  if (cached) return cached;
  if (planCache.size > 64) planCache.clear();
  const tokens = lexicalTokens(query);
  const tokenSet = new Set(tokens);
  const phrases = cjkPhrases(query.trim().toLowerCase())
    .filter((phrase) => !tokenSet.has(phrase))
    .map((token) => ({ ...buildMatcher(token), title: token.length >= 4 ? 14 : 12 }));
  const plan = { tokens, matchers: tokens.map(buildMatcher), phrases };
  planCache.set(query, plan);
  return plan;
}

/** A landing page is a short label (a board or product name) with no procedure text. */
export function isThinLanding(doc: IndexedDoc): boolean {
  if (structureScale(doc) > 1) return false;
  const text = `${doc.text ?? ""} ${doc.answer ?? ""}`.trim();
  if (text.length >= 80) return false;
  const tokens = doc.title
    .trim()
    .split(/[\s/|:_-]+/)
    .filter((token) => token.length > 0);
  if (tokens.length === 0 || tokens.length > 3) return false;
  // A content word in the title is a topic page, even when the stored snippet is short.
  if (tokens.some((token) => /[\u4e00-\u9fff]{2,}/.test(token) || /[a-z]{5,}/i.test(token))) return false;
  return true;
}

/**
 * Corpus-independent shape of a how-to section: a guide-style heading,
 * numbered steps, or code blocks. No URL or page name is consulted.
 */
export function structureScale(doc: IndexedDoc): number {
  const text = doc.text ?? "";
  const title = doc.title.trim();
  const steps = text.match(/(?:^|\n)\s*\d+[.、)]|第[0-9一二三四五六七八九十]{1,3}步/g)?.length ?? 0;
  const code = text.match(/```|`[^`\n]{2,}`/g)?.length ?? 0;
  const guideTitle = /^(?:\d+\.)+\d*\s+\S/.test(title) || /指南|教程|步骤|示例|howto|\btutorial\b|\bguide\b/i.test(title);
  if (steps >= 2 || code >= 2) return guideTitle ? 1.22 : 1.15;
  if (steps === 1 || code === 1 || guideTitle) return 1.08;
  return 1;
}

/**
 * Title fusion weight. High enough that a title match moves a near neighbor,
 * low enough that a rare BM25 hit stays ahead of a generic title.
 */
export const LEX_FUSION_WEIGHT = 0.22;

/** Reciprocal-rank fusion. Rank 0 is the best rank. */
export function fuseRanks(bmRank: number, lexRank: number, lexWeight = LEX_FUSION_WEIGHT, k = 10): number {
  return 1 / (k + bmRank) + lexWeight / (k + lexRank);
}

const LEX_SCAN = 280;

export function lexicalScore(doc: IndexedDoc, plan: LexicalPlan, common?: ReadonlySet<string>): number {
  const title = doc.title.toLowerCase();
  const extra = [doc.snippet, doc.text, ...(doc.breadcrumbs ?? [])]
    .filter(Boolean)
    .join(" ")
    .toLowerCase()
    .slice(0, LEX_SCAN);
  const answer = (doc.answer ?? "").toLowerCase().slice(0, LEX_SCAN);
  let score = 0;
  let matched = 0;
  let titleMatched = 0;
  for (const matcher of plan.matchers) {
    // A title made only of generic or very common words is not an answer.
    if (WEAK_TITLE.has(matcher.token) || common?.has(matcher.token)) {
      if (matcher.test(extra) || (answer && matcher.test(answer))) {
        score += 1;
        matched += 1;
      }
      continue;
    }
    let hit = false;
    if (title === matcher.token) {
      score += 14;
      hit = true;
      titleMatched += 1;
    } else if (matcher.test(title)) {
      score += 10;
      hit = true;
      titleMatched += 1;
    }
    if (matcher.test(extra)) {
      score += 3;
      hit = true;
    } else if (answer && matcher.test(answer)) {
      // Answer text is evidence, but a title or section body outranks it.
      score += 1;
      hit = true;
    }
    if (doc.kind === "page" && matcher.test(title)) score += 2;
    if (hit) matched += 1;
  }
  for (const phrase of plan.phrases) {
    if (WEAK_TITLE.has(phrase.token) || common?.has(phrase.token)) continue;
    if (phrase.test(title)) score += phrase.title;
    else if (phrase.test(extra) || (answer && phrase.test(answer))) score += 2;
  }
  if (plan.matchers.length > 1) {
    score += Math.round((matched / plan.matchers.length) * 12);
    if (matched === 1 && plan.matchers.length >= 4 && titleMatched === 0) score = Math.min(score, 4);
  }
  return score;
}
