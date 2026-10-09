import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";
import type { HttpGet } from "./http.js";
import { fetchText } from "./http.js";
import type { IndexedDoc } from "./types.js";

const memory = new WeakMap<HttpGet, Map<string, IndexedDoc[]>>();

export function prebuiltDir(): string {
  return join(dirname(fileURLToPath(import.meta.url)), "..", "prebuilt");
}

export function prebuiltEnabled(http: HttpGet): boolean {
  if (http !== fetchText) return false;
  return process.env.RDK_DOCS_PREBUILT !== "0";
}

export function recallIndex(http: HttpGet, manualId: string): IndexedDoc[] | undefined {
  return memory.get(http)?.get(manualId);
}

export function rememberIndex(http: HttpGet, manualId: string, docs: IndexedDoc[]): void {
  let bucket = memory.get(http);
  if (!bucket) {
    bucket = new Map();
    memory.set(http, bucket);
  }
  bucket.set(manualId, docs);
}

export function readPrebuilt(manualId: string): IndexedDoc[] | undefined {
  const path = join(prebuiltDir(), `${manualId}.json.gz`);
  if (!existsSync(path)) return undefined;
  try {
    const parsed = JSON.parse(gunzipSync(readFileSync(path)).toString("utf8")) as unknown;
    if (!Array.isArray(parsed)) return undefined;
    return parsed as IndexedDoc[];
  } catch {
    return undefined;
  }
}
