import { CONCEPT_SYNONYMS, RETRIEVAL_ALIASES } from "./aliases.js";
import { GLOSSARY_ALIASES } from "./glossary-aliases.js";
import type { IndexedDoc } from "./types.js";

/**
 * Title-heavy lexical score from the pre-BM25 ranker, kept as a second
 * signal. BM25 is better at rare identifiers; this score is better when the
 * page title or a stable URL class (install, burn, remote login) is the
 * actual answer. The two are fused with reciprocal rank.
 */

const CJK_STOPWORDS = ["怎么样", "怎么", "怎样", "如何", "什么", "哪些", "哪里", "是否", "多少", "请问", "帮我", "一下", "可不可以", "能不能", "有没有"];
const CJK_STOP_CHARS = new Set(["的", "了", "吗", "呢", "啊", "吧", "把", "是", "有", "个", "和", "或", "在", "给", "去", "到", "太", "很"]);

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

const ALL_SYNONYMS = mergeSynonyms(BASE_SYNONYMS, GLOSSARY_ALIASES, RETRIEVAL_ALIASES, CONCEPT_SYNONYMS);
const CJK_RUN = /[\u4e00-\u9fff]+/g;

export type LexicalPlan = {
  tokens: string[];
  matchers: Array<{ token: string; test: (text: string) => boolean }>;
};

function cjkBigrams(query: string): string[] {
  const grams: string[] = [];
  for (const match of query.matchAll(CJK_RUN)) {
    let run = match[0];
    for (const stop of CJK_STOPWORDS) run = run.replaceAll(stop, "\u0000");
    run = [...run].map((ch) => (CJK_STOP_CHARS.has(ch) ? "\u0000" : ch)).join("");
    for (const segment of run.split("\u0000")) {
      if (segment.length < 2) continue;
      for (let i = 0; i + 2 <= segment.length; i += 1) grams.push(segment.slice(i, i + 2));
    }
  }
  return grams;
}

export function lexicalTokens(query: string): string[] {
  const seen = new Set<string>();
  const lowered = query.trim().toLowerCase();
  for (const match of lowered.matchAll(/[a-z][a-z0-9_.-]*|\d+[a-z][a-z0-9_.-]*|\d+/g)) seen.add(match[0]);
  for (const match of lowered.matchAll(/(\d+)[\s-]+([a-z][a-z0-9_.-]*)/g)) seen.add(`${match[1]}${match[2]}`);
  for (const gram of cjkBigrams(lowered)) seen.add(gram);
  for (const [key, extras] of Object.entries(ALL_SYNONYMS)) {
    if (seen.has(key)) extras.forEach((item) => seen.add(item.toLowerCase()));
  }
  if (/网线供电|以太网供电|网线.*供电/.test(lowered)) seen.add("poe");
  if (/风扇|烫|过热/.test(query)) {
    seen.add("温度");
    seen.add("散热");
  }
  if (seen.size === 0) {
    for (const part of lowered.split(/\s+/).filter(Boolean)) seen.add(part);
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
  const plan = { tokens, matchers: tokens.map(buildMatcher) };
  planCache.set(query, plan);
  return plan;
}

export function isThinLanding(doc: IndexedDoc): boolean {
  const url = doc.url.toLowerCase();
  const title = doc.title.toLowerCase().replace(/[\s_-]+/g, "");
  if (/^(rdk)?(x3|x5|s100|s600|s100p)$/.test(title)) return true;
  const base = url.split("#")[0] ?? url;
  // The manual home (…/RDK) is a stack of one-line headings, not a procedure.
  if (/\/rdk\/?$/.test(base)) return true;
  return false;
}

/** Model-family markers. A VLA page should not answer an LLM question. */
const TOPIC_FAMILIES = ["llm", "vla", "vlm", "asr"];

export function topicClash(query: string, doc: IndexedDoc): boolean {
  const q = query.toLowerCase();
  const queried = new Set<string>();
  for (const family of TOPIC_FAMILIES) {
    if (new RegExp(`(?:^|[^a-z])${family}(?:[^a-z]|$)`).test(q)) queried.add(family);
  }
  if (/大模型|大语言模/.test(query) && !/vla|vlm|动作模型|视觉语言动作/.test(q)) queried.add("llm");
  if (queried.size === 0) return false;
  const blob = `${doc.url} ${doc.title}`.toLowerCase();
  const present = TOPIC_FAMILIES.filter((family) => new RegExp(`(?:^|[^a-z])${family}(?:[^a-z]|$)`).test(blob));
  if (present.length === 0) return false;
  for (const family of queried) if (present.includes(family)) return false;
  return true;
}

export function urlIntentScale(plan: LexicalPlan, url: string, query: string): number {
  const path = url.toLowerCase();
  const tokens = plan.tokens;
  let scale = 1;
  const wantsBurn = tokens.some((token) => ["烧录", "flash", "burn", "镜像", "刷机"].includes(token));
  const wantsInstall = tokens.some((token) => token === "安装" || token === "install");
  const wantsWifi = tokens.some((token) => ["wifi", "wi-fi", "无线", "wlan", "nmcli"].includes(token));
  const wantsGpio = tokens.some((token) => token === "gpio");
  const wantsPin = tokens.some((token) => token === "pin" || token === "40pin");
  const howTo = /怎么|如何|怎样|how to|how do/i.test(query);
  if (wantsBurn && /burn|xburn|flash/.test(path)) scale *= 1.42;
  if (wantsInstall && /install/.test(path) && !/cross_compile/.test(path)) scale *= 1.38;
  if (wantsWifi && /wifi|remote_login|wlan|network/.test(path)) scale *= 1.36;
  if (wantsGpio && howTo && /40pin|user_sample/.test(path) && /gpio/.test(path)) scale *= 1.28;
  if (wantsPin && /40pin/.test(path)) scale *= 1.18;
  return scale;
}

function commandIsSubject(query: string, url: string): boolean {
  const match = url.toLowerCase().match(/\/cmd_([a-z0-9_-]+)/);
  if (!match?.[1]) return false;
  const name = match[1].replace(/-/g, "");
  const q = query.toLowerCase();
  if (!q.includes(name) && !q.includes(match[1])) return false;
  if (/密码|账号|账户|口令/.test(query)) return false;
  if (/板子|开发板|地址/.test(query) && /怎么|哪里|如何|查|多少/.test(query)) return false;
  return true;
}

export function linuxCommandPenalty(query: string, url: string): number {
  if (!/linux-command-manual/.test(url.toLowerCase())) return 1;
  const setup = /怎么|如何|怎样|什么|多少|默认|密码|账号|账户|板子|开发板|波特|静态|wifi|无线|ip/i.test(query);
  if (!setup || commandIsSubject(query, url)) return 1;
  return 0.52;
}

export function credentialFaqBoost(query: string, doc: IndexedDoc): number {
  // Credential FAQs stay in the list, but they must not outrank the login
  // procedure. The login page is the how-to; the FAQ is a one-line reminder.
  if (!/\/faq\//i.test(doc.url)) return 1;
  if (!/密码|账号|账户|口令/.test(query) || !/密码|账户|账号|登录/.test(doc.title)) return 1;
  if (/怎么|如何|登/.test(query)) return 0.72;
  return 1;
}

/**
 * Promote the manual's primary how-to page over a secondary page that merely
 * mentions the same words: the quick-start procedure over a legacy duplicate,
 * the user sample over a driver chapter when the question is "how do I use
 * this", and the page that states a default over the page that edits it.
 * Scales are applied to the BM25 score, before rank fusion.
 */
export function primaryScale(query: string, doc: IndexedDoc): number {
  const q = query.toLowerCase();
  const path = doc.url.toLowerCase();
  const title = doc.title.toLowerCase();
  let scale = 1;

  const burn = /烧录|刷机|镜像|\bflash\b|\bburn\b/.test(q);
  if (burn) {
    if (/\/system-burn\/|\/system-flashing\//.test(path)) scale *= 1.8;
    else if (/rdk_x_doc\/.*\/install_os\//.test(path)) scale *= 0.52;
  }
  if (/支持哪些|哪些板/.test(query) && /\/overview(?:\/|$|#)/.test(path)) scale *= 2.6;

  if (/wifi|wi-fi|wlan|无线|nmcli/.test(q)) {
    if (/remote_login/.test(path)) scale *= 9;
    else if (/network_blueteeth|network_bluetooth/.test(path)) scale *= 1.55;
    else if (/driver_wifi|wifi_performance/.test(path)) scale *= 0.48;
    else if (/configuration_wizard/.test(path)) scale *= 0.58;
  }
  if (/口令|密码|账号|\bssh\b|登录/.test(q) && /remote_login/.test(path)) scale *= 1.75;

  if (/简介/.test(query) && /硬件|接口|hardware/.test(q)) {
    if (/hardware_introduction|_kit(?:\/|$|#)/.test(path)) scale *= 2.15;
    else if (/board_bringup|release_note/.test(path)) scale *= 0.45;
  }

  const gpio = /gpio|管脚|引脚/.test(q);
  const usage = /怎么|如何|应用|示例|教程|步骤|sample|how to|40pin/.test(q);
  const pinmap = /pin\s*map|引脚定义|管脚定义|40pin/.test(q);
  if (gpio && usage && /40pin_user_(sample|guide)/.test(path)) scale *= 1.7;
  if (gpio && usage && /driver_gpio/.test(path)) scale *= 0.58;
  if (gpio && !usage && !pinmap && /driver_gpio/.test(path)) scale *= 1.85;
  if (gpio && !usage && !pinmap && /40pin_user_(sample|guide)/.test(path)) scale *= 0.4;
  if (pinmap && /40pin_define|40pin_user_sample\/gpio|40pin_user_guide/.test(path)) scale *= 1.5;
  if (pinmap && /driver_gpio|driver_pinctrl/.test(path)) scale *= 0.7;

  if (/usb/.test(q) && /摄像头|camera/.test(q)) {
    if (/\/usb_camera(?:\/|$|#)/.test(path) && !/sample/.test(path)) scale *= 1.5;
    else if (/hardware_introduction/.test(path)) scale *= 0.68;
  }

  if (/mipi/.test(q) && /摄像头|camera/.test(q) && !/怎么|如何|失败|无法|没有/.test(query)) {
    if (/\/mipi_camera(?:\/|$|#)/.test(path)) scale *= 2.1;
    else if (/\/faq\//.test(path)) scale *= 0.55;
  }

  if (/tros|togetheros|togetherros/.test(q) && /安装|install/.test(q)) {
    if (/install_tros/.test(path)) scale *= 2.5;
    else if (/ros_pkg|cross_compile/.test(path)) scale *= 0.58;
  }

  if (/升级|upgrade/.test(q)) {
    if (/upgrade-/.test(path)) scale *= 1.9;
    else if (/\/faq\//.test(path)) scale *= 0.5;
  }

  if (/风扇|温度|散热|烫/.test(query)) {
    if (/frequency_management/.test(path)) scale *= 1.55;
    else if (/driver_thermal/.test(path)) scale *= 0.58;
  }

  if (/默认/.test(query) && /ip|地址|静态/.test(q)) {
    if (/修改/.test(title) || /修改/.test(path)) scale *= 0.58;
    if (/remote_login|hardware_introduction/.test(path)) scale *= 2;
    if (/\/faq\//.test(path) && /ip|网口|地址/.test(title)) scale *= 1.75;
  }

  const model = /\b(qwen|whisper|llama|deepseek|yolo(?:v\d+)?)\b/.exec(q);
  if (model) {
    if (path.includes(model[1])) scale *= 1.65;
    const base = path.split("#")[0] ?? path;
    if (/\/s100p?$|\/s600$/.test(base)) scale *= 0.5;
  }

  if (/\bapi\b/.test(q) && /\/faq\//.test(path)) scale *= 0.42;
  if (/\bapi\b/.test(q) && /python-api|pyeasy|ai-python-api|basic_sample/.test(path)) scale *= 1.45;
  if (/hbm_runtime/.test(q) && /python-api/.test(path)) scale *= 1.7;
  if (/hbm_runtime/.test(q) && /python_sample|yolov5/.test(path)) scale *= 0.6;

  return scale;
}

export function lexicalScore(doc: IndexedDoc, plan: LexicalPlan, query: string): number {
  const title = doc.title.toLowerCase();
  const extra = [doc.snippet, doc.text, ...(doc.breadcrumbs ?? [])].filter(Boolean).join(" ").toLowerCase();
  const url = doc.url.toLowerCase();
  let score = 0;
  let matched = 0;
  let titleMatched = 0;
  for (const matcher of plan.matchers) {
    const weak = WEAK_TITLE.has(matcher.token) ? 0.3 : 1;
    let hit = false;
    if (title === matcher.token) {
      score += 14 * weak;
      hit = true;
      titleMatched += weak >= 1 ? 1 : 0;
    } else if (matcher.test(title)) {
      score += 10 * weak;
      hit = true;
      titleMatched += weak >= 1 ? 1 : 0;
    }
    if (matcher.test(extra)) {
      score += 3 * weak;
      hit = true;
    }
    if (matcher.test(url)) {
      score += 4 * weak;
      hit = true;
    }
    if (doc.kind === "page" && matcher.test(title)) score += 2;
    if (hit) matched += 1;
  }
  if (plan.matchers.length > 1) {
    score += Math.round((matched / plan.matchers.length) * 12);
    if (matched === 1 && plan.matchers.length >= 4 && titleMatched === 0) score = Math.min(score, 4);
  }
  const tokens = plan.tokens;
  if (tokens.some((token) => ["烧录", "flash", "burn", "镜像"].includes(token)) && /burn|xburn|flash/.test(url)) score += 10;
  if (tokens.some((token) => token === "安装" || token === "install") && /install/.test(url) && !/cross_compile/.test(url)) score += 8;
  if (tokens.some((token) => ["wifi", "wi-fi", "无线"].includes(token)) && /wifi|remote_login|wlan/.test(url)) score += 10;
  if (tokens.some((token) => token === "gpio") && /40pin|user_sample/.test(url) && /gpio/.test(url)) score += 8;
  if (tokens.some((token) => token === "pin" || token === "40pin") && /40pin|user_sample/.test(url)) score += 8;
  if (titleMatched > 0 && (/\/overview(?:\.html)?(?:#|$)/.test(url) || title.includes("概述"))) score += 4;
  if (isThinLanding(doc)) score -= 12;
  if (topicClash(query, doc)) score -= 14;
  if (linuxCommandPenalty(query, doc.url) < 1) score = Math.round(score * 0.55);
  if (/\/faq\//i.test(url)) score -= 6;
  if (/登录|口令|密码|账号|ssh/.test(query) && /remote_login/.test(url)) score += 8;
  return score;
}
