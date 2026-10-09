import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

/** Published Xenova export of BAAI/bge-small-zh-v1.5. Loaded from disk only. */
export function bgeModelDir(): string {
  return join(dirname(fileURLToPath(import.meta.url)), "..", "models", "bge-small-zh-v1.5");
}

const CLS = 101;
const SEP = 102;
const UNK = 100;
const MAX_WORD_CHARS = 100;

let vocab: Map<string, number> | undefined;

function vocabMap(): Map<string, number> {
  if (vocab) return vocab;
  const text = readFileSync(join(bgeModelDir(), "vocab.txt"), "utf8");
  const map = new Map<string, number>();
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i += 1) {
    const token = lines[i]?.replace(/\r$/, "") ?? "";
    if (token.length === 0) continue;
    map.set(token, i);
  }
  vocab = map;
  return map;
}

function isControl(cp: number): boolean {
  if (cp === 0x09 || cp === 0x0a || cp === 0x0d) return false;
  return /\p{Cc}|\p{Cf}/u.test(String.fromCodePoint(cp));
}

function isWhitespace(cp: number): boolean {
  if (cp === 0x20 || cp === 0x09 || cp === 0x0a || cp === 0x0d) return true;
  return /\p{Zs}/u.test(String.fromCodePoint(cp));
}

/** BertNormalizer Chinese range, plus the compatibility ideograph blocks. */
function isChinese(cp: number): boolean {
  return (
    (cp >= 0x4e00 && cp <= 0x9fff) ||
    (cp >= 0x3400 && cp <= 0x4dbf) ||
    (cp >= 0xf900 && cp <= 0xfaff) ||
    (cp >= 0x20000 && cp <= 0x2a6df) ||
    (cp >= 0x2f800 && cp <= 0x2fa1f)
  );
}

function isPunct(cp: number): boolean {
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
  return /\p{P}/u.test(String.fromCodePoint(cp));
}

/** BertNormalizer with lowercase off, matching tokenizer_config of this model. */
function normalize(text: string): string {
  let out = "";
  for (const ch of text) {
    const cp = ch.codePointAt(0) ?? 0;
    if (cp === 0 || cp === 0xfffd || isControl(cp)) continue;
    if (isWhitespace(cp)) {
      out += " ";
      continue;
    }
    if (isChinese(cp)) {
      out += ` ${ch} `;
      continue;
    }
    out += ch;
  }
  return out;
}

function basicTokens(text: string): string[] {
  const tokens: string[] = [];
  let buf = "";
  const flush = () => {
    if (buf.length > 0) tokens.push(buf);
    buf = "";
  };
  for (const ch of normalize(text)) {
    const cp = ch.codePointAt(0) ?? 0;
    if (isWhitespace(cp)) {
      flush();
      continue;
    }
    if (isPunct(cp)) {
      flush();
      tokens.push(ch);
      continue;
    }
    buf += ch;
  }
  flush();
  return tokens;
}

function wordPiece(token: string, words: Map<string, number>): number[] {
  const chars = [...token];
  if (chars.length > MAX_WORD_CHARS) return [UNK];
  const ids: number[] = [];
  let start = 0;
  while (start < chars.length) {
    let end = chars.length;
    let found: string | undefined;
    while (start < end) {
      const piece = (start === 0 ? "" : "##") + chars.slice(start, end).join("");
      if (words.has(piece)) {
        found = piece;
        break;
      }
      end -= 1;
    }
    if (!found) return [UNK];
    ids.push(words.get(found) ?? UNK);
    start = end;
  }
  return ids;
}

/** Token ids including [CLS] and [SEP], truncated to `maxLen`. */
export function encodeBert(text: string, maxLen: number): number[] {
  const words = vocabMap();
  const limit = Math.max(2, maxLen);
  const ids: number[] = [CLS];
  for (const token of basicTokens(text)) {
    if (ids.length >= limit - 1) break;
    for (const id of wordPiece(token, words)) {
      if (ids.length >= limit - 1) break;
      ids.push(id);
    }
  }
  ids.push(SEP);
  return ids;
}
