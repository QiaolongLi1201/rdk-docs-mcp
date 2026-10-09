import { RETRIEVAL_ALIASES } from "./aliases.js";
import { GLOSSARY_ALIASES } from "./glossary-aliases.js";
import { mentionedBoards, soleBoard, urlLooksLikeBoard, type BoardId } from "./products.js";
import type { IndexedDoc, SearchHit } from "./types.js";

/** Question filler that carries no retrieval signal in Chinese queries. */
const CJK_STOPWORDS = [
  "怎么样",
  "怎么",
  "怎样",
  "如何",
  "什么",
  "哪些",
  "哪里",
  "是否",
  "多少",
  "请问",
  "帮我",
  "一下",
  "可不可以",
  "能不能",
  "有没有",
];

const CJK_STOP_CHARS = new Set(["的", "了", "吗", "呢", "啊", "吧", "把", "是", "有", "个", "和", "或", "在", "给", "去", "到", "太", "很"]);

const SYNONYMS: Record<string, string[]> = {
  flash: ["烧录", "burn"],
  burn: ["烧录"],
  wifi: ["wi-fi", "无线"],
  install: ["安装"],
};

/** 并集合并多份同义词表:key 冲突时数组取并集,而非后者覆盖前者。 */
function mergeSynonyms(...maps: Array<Record<string, string[]>>): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  for (const map of maps) {
    for (const [key, extras] of Object.entries(map)) {
      out[key] = [...new Set([...(out[key] ?? []), ...extras])];
    }
  }
  return out;
}

const ALL_SYNONYMS = mergeSynonyms(SYNONYMS, GLOSSARY_ALIASES, RETRIEVAL_ALIASES);

const CJK_RUN = /[\u4e00-\u9fff]+/g;
const NEWER: Record<BoardId, number> = { s600: 4, s100: 3, x5: 2, x3: 1 };
const BOARDS: BoardId[] = ["s600", "s100", "x5", "x3"];

function cjkPieces(query: string): string[] {
  const grams: string[] = [];
  for (const match of query.matchAll(CJK_RUN)) {
    let run = match[0];
    for (const stop of CJK_STOPWORDS) {
      run = run.replaceAll(stop, "\u0000");
    }
    run = [...run].map((ch) => (CJK_STOP_CHARS.has(ch) ? "\u0000" : ch)).join("");
    for (const segment of run.split("\u0000")) {
      if (segment.length < 2) continue;
      if (segment.length <= 12) grams.push(segment);
      for (let i = 0; i + 2 <= segment.length; i += 1) {
        grams.push(segment.slice(i, i + 2));
      }
    }
  }
  return grams;
}

export function tokens(query: string): string[] {
  const seen = new Set<string>();
  const lowered = query.trim().toLowerCase();
  for (const match of lowered.matchAll(/[a-z][a-z0-9_.-]*|\d+[a-z][a-z0-9_.-]*|\d+/g)) {
    seen.add(match[0]);
  }
  for (const match of lowered.matchAll(/(\d+)[\s-]+([a-z][a-z0-9_.-]*)/g)) {
    seen.add(`${match[1]}${match[2]}`);
  }
  for (const gram of cjkPieces(lowered)) {
    seen.add(gram);
  }
  for (const [key, extras] of Object.entries(ALL_SYNONYMS)) {
    if (seen.has(key)) extras.forEach((item) => seen.add(item.toLowerCase()));
  }
  if (RETRIEVAL_ALIASES[lowered]) {
    for (const extra of RETRIEVAL_ALIASES[lowered]) seen.add(extra.toLowerCase());
  }
  if (seen.size === 0) {
    for (const part of lowered.split(/\s+/).filter(Boolean)) seen.add(part);
  }
  return [...seen];
}

type Matcher = { token: string; weight: number; test: (text: string) => boolean };

function isIdentifier(token: string): boolean {
  return /^[a-z][a-z0-9_.-]{3,}$/.test(token) && (token.includes("_") || token.includes(".") || token.length >= 6);
}

function buildMatcher(token: string): Matcher {
  const weight = isIdentifier(token) ? 1 : token.length <= 1 ? 0.4 : 1;
  if (!/^[a-z0-9][a-z0-9_.-]*$/.test(token)) {
    return { token, weight, test: (text) => text.includes(token) };
  }
  const escaped = token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern =
    token.length <= 3
      ? new RegExp(`(?<![a-z])${escaped}(?![a-z])`)
      : new RegExp(`(?<![a-z0-9])${escaped}`);
  const hyphenated = token.replaceAll("_", "-");
  const underscored = token.replaceAll("-", "_");
  const flat = token.replace(/[-_]/g, "");
  return {
    token,
    weight,
    test: (text) =>
      pattern.test(text) ||
      (hyphenated !== token && text.includes(hyphenated)) ||
      (underscored !== token && text.includes(underscored)) ||
      (flat.length >= 6 && flat !== token && text.includes(flat)),
  };
}

function haystack(doc: IndexedDoc): { title: string; extra: string; url: string } {
  return {
    title: doc.title.toLowerCase(),
    extra: [doc.snippet, doc.text, ...(doc.breadcrumbs ?? [])].filter(Boolean).join(" ").toLowerCase(),
    url: doc.url.toLowerCase(),
  };
}

export type RankOptions = { board?: BoardId };

function contextBoard(query: string, options: RankOptions): BoardId | undefined {
  return soleBoard(query) ?? options.board;
}

function newerRank(doc: IndexedDoc): number {
  let best = 0;
  for (const board of BOARDS) {
    if (urlLooksLikeBoard(doc.url, board) || urlLooksLikeBoard(doc.title, board)) {
      best = Math.max(best, NEWER[board]);
    }
  }
  return best;
}

function boardsOn(doc: IndexedDoc): BoardId[] {
  return BOARDS.filter((board) => urlLooksLikeBoard(doc.url, board) || urlLooksLikeBoard(doc.title, board));
}

function scoreDoc(doc: IndexedDoc, matchers: Matcher[], query: string, options: RankOptions): number {
  const { title, extra, url } = haystack(doc);
  const queryTokens = matchers.map((item) => item.token);
  let score = 0;
  let matched = 0;
  let titleMatched = 0;
  for (const { token, weight, test } of matchers) {
    let hit = false;
    const ident = isIdentifier(token);
    if (title === token) {
      score += 14 * weight;
      hit = true;
      titleMatched += 1;
    } else if (test(title)) {
      score += 10 * weight;
      hit = true;
      titleMatched += 1;
    }
    if (test(extra)) {
      score += 3 * weight;
      hit = true;
    }
    if (test(url)) {
      score += 4 * weight;
      hit = true;
      if (ident) score += 10;
    }
    if (doc.kind === "page" && test(title)) score += 2;
    if (ident && hit) score += 8;
    if (hit) matched += 1;
  }
  if (matchers.length > 1) {
    score += Math.round((matched / matchers.length) * 12);
    if (matched === 1 && matchers.length >= 4 && titleMatched === 0) {
      score = Math.min(score, 4);
    }
  }

  const wantsBurn = queryTokens.some((token) => ["烧录", "flash", "burn", "镜像"].includes(token));
  const wantsInstall = queryTokens.some((token) => ["安装", "install"].includes(token));
  const wantsWifi = queryTokens.some((token) => ["wifi", "wi-fi", "无线"].includes(token));
  const wantsGpio = queryTokens.some((token) => token === "gpio");
  const wantsPin = queryTokens.some((token) => token === "pin" || token === "40pin");
  const wantsCases = queryTokens.some((token) => token === "案例");
  const wantsInfer = queryTokens.some((token) => ["inference", "推理", "bpu"].includes(token));
  const wantsApt = queryTokens.some((token) => ["apt", "软件源", "sources.list"].includes(token));
  const wantsMipi = queryTokens.some((token) => token === "mipi" || token === "摄像头" || token === "camera");
  const wantsV4l2 = queryTokens.some((token) => token === "v4l2");
  const bareIdent = /^[a-z][a-z0-9_.-]+$/i.test(query.trim());
  const comparison = /对比|区别|不同|相比|\bvs\b|versus|比较/i.test(query);
  const phrase = query.trim().toLowerCase();
  if (phrase.length >= 2 && extra.includes(phrase) && !title.includes(phrase)) score += 9;

  if (wantsBurn && /burn|xburn|flash/.test(url)) score += 10;
  if (wantsInstall && /install/.test(url) && !/cross_compile/.test(url)) score += 8;
  if (wantsWifi && /wifi|remote_login|wlan/.test(url)) score += 10;
  if (wantsGpio && /40pin|user_sample/.test(url) && /gpio/.test(url)) score += 8;
  if (wantsPin && /40pin|user_sample/.test(url)) score += 8;
  if (wantsCases && (/\/case\/?$/.test(url) || title.includes("应用案例"))) score += 10;
  if (wantsMipi && /mipi_camera|mipi-camera/.test(url)) score += 12;
  if (wantsV4l2 && /v4l2/.test(url)) score += 24;
  if (wantsV4l2 && /mipi_camera|usb_camera|web_display_camera/.test(url) && !/mipi|usb/i.test(query)) score -= 60;
  if (wantsInfer && /bpu_api|pyeasy_dnn|ai-python-api|python-api/.test(url)) score += 12;
  if (wantsInfer && /bpu_mem|stress|sysfs/.test(url) && !/内存|占用|sysfs/.test(query)) score -= 10;
  if (wantsApt && /hardware_and_system/.test(url)) score += 8;
  if (wantsApt && /sources\.list/.test(query) && /hardware_and_system/.test(url)) score += 12;
  if (wantsApt && /isp_|light_source|awb/.test(url)) score -= 24;
  if (wantsApt && /tros_ros/.test(url) && !/ros|tros/i.test(query)) score -= 12;
  if (titleMatched > 0 && (/\/overview(?:\.html)?$/.test(url) || title.includes("概述"))) score += 4;
  if (/\/faq\/|accessory|release_note|changelog|config_txt/.test(url)) score -= 6;
  if (bareIdent && /\/faq\//.test(url)) score -= 14;

  if (comparison) {
    const overview =
      /\/hardware_introduction\/rdk_x[35](?:[#/?]|$)/.test(url) ||
      /\/01_rdk_s100_kit(?:[#/?]|$)/.test(url) ||
      /\/01_rdk_s600_kit(?:[#/?]|$)/.test(url);
    if (overview) score += 28;
    const sideTopic = /network|blueteeth|bluetooth|remote_login|cdev_demo|camera_expansion|driver_/.test(url);
    const asksSide = /网络|wifi|蓝牙|ip|摄像头|相机|驱动|demo|示例/i.test(query);
    if (sideTopic && !asksSide) score -= 14;
  }

  const troubleQuery = /no image|无图|黑屏|不出|打不开|失败|报错|error|troubleshoot/i.test(query);
  if (doc.manualId === "forum") {
    const promo = /新品|首发|重磅|正式发布|发布会|发布包/.test(doc.title);
    if (promo && troubleQuery) score -= 18;
    if (troubleQuery && /求助|报错|失败|无图|黑屏|打不开|无法|不出/.test(doc.title)) score += 8;
  }

  const mentioned = mentionedBoards(query);
  const sole = contextBoard(query, options);
  if (mentioned.length > 1) {
    const matchesMentioned = mentioned.some((board) => urlLooksLikeBoard(doc.url, board) || urlLooksLikeBoard(doc.title, board));
    const other = BOARDS.filter((board) => !mentioned.includes(board)).some(
      (board) => urlLooksLikeBoard(doc.url, board) || urlLooksLikeBoard(doc.title, board),
    );
    const unrelatedFamily =
      (doc.manualId === "rdk-s" && mentioned.every((board) => board === "x3" || board === "x5")) ||
      (doc.manualId === "rdk-x" && mentioned.every((board) => board === "s100" || board === "s600"));
    if ((other && !matchesMentioned) || unrelatedFamily) score = -1;
  } else if (sole) {
    const mine = urlLooksLikeBoard(doc.url, sole) || urlLooksLikeBoard(doc.title, sole);
    const other = BOARDS.filter((board) => board !== sole).some(
      (board) => urlLooksLikeBoard(doc.url, board) && !urlLooksLikeBoard(doc.url, sole),
    );
    if (mine) score += 8;
    if (other) score -= 12;
  }

  return score;
}

function canonicalUrl(url: string): string {
  return url.split("#")[0] ?? url;
}

function lastUrlSegment(url: string): string {
  const path = canonicalUrl(url).split("/").filter(Boolean);
  return path.at(-1) || url;
}

function fillSnippet(doc: IndexedDoc): string {
  const crumbs = doc.breadcrumbs?.filter(Boolean).join(" / ");
  const filled = doc.snippet || doc.text?.slice(0, 180) || crumbs || "";
  return filled.trim().slice(0, 240) || lastUrlSegment(doc.url);
}

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

export function rankHits(docs: IndexedDoc[], query: string, limit: number, options: RankOptions = {}): SearchHit[] {
  const queryTokens = tokens(query);
  if (queryTokens.length === 0) return [];
  const matchers = queryTokens.map(buildMatcher);
  const preferNewer = !soleBoard(query) && !options.board && mentionedBoards(query).length === 0;

  const best = new Map<string, SearchHit & { newer: number }>();
  const titleFromPage = new Map<string, boolean>();
  for (const doc of docs) {
    const score = scoreDoc(doc, matchers, query, options);
    if (score <= 0) continue;
    const base = canonicalUrl(doc.url);
    const boards = boardsOn(doc);
    const hit: SearchHit & { newer: number } = {
      title: doc.title,
      url: doc.url,
      manual: doc.manualId,
      snippet: fillSnippet(doc),
      score,
      source: doc.manualId === "forum" ? "forum" : "docs",
      board: boards.length === 1 ? boards[0] : boards.length > 1 ? "multiple" : undefined,
      quality: score >= 12 ? "good" : "weak",
      newer: newerRank(doc),
    };
    const prev = best.get(base);
    if (!prev) {
      best.set(base, hit);
      titleFromPage.set(base, doc.kind === "page");
      continue;
    }

    const prevWasPage = titleFromPage.get(base) === true;
    const nextIsPage = doc.kind === "page";
    let title = prev.title;
    if (nextIsPage && (!prevWasPage || hit.score >= prev.score)) title = hit.title;
    else if (!prevWasPage && hit.score > prev.score) title = hit.title;

    const winner = hit.score > prev.score ? hit : prev;
    const snippet = (hit.score > prev.score ? hit.snippet : prev.snippet) || hit.snippet || prev.snippet;
    const url = hit.score > prev.score ? hit.url : prev.url;
    best.set(base, { ...winner, title, snippet, url, score: Math.max(hit.score, prev.score) });
    titleFromPage.set(base, prevWasPage || nextIsPage);
  }

  return [...best.values()]
    .sort((a, b) => b.score - a.score || (preferNewer ? b.newer - a.newer : 0))
    .slice(0, limit)
    .map(({ newer: _newer, ...hit }) => aliasNote(query, hit));
}

export const GOOD_MATCH_SCORE = 12;

export function matchQuality(hits: SearchHit[]): { noGoodMatch: boolean; matchQuality: "good" | "weak" | "none" } {
  const top = hits[0]?.score ?? 0;
  if (hits.length === 0 || top <= 0) return { noGoodMatch: true, matchQuality: "none" };
  if (top < GOOD_MATCH_SCORE) return { noGoodMatch: true, matchQuality: "weak" };
  return { noGoodMatch: false, matchQuality: "good" };
}
