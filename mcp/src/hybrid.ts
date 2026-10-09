import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { encodeBert, bgeModelDir } from "./bert-tokenizer.js";
import { decodeEmbeddingBlob, DENSE_DIM, docsFingerprint, passageText, type LoadedEmbeddings } from "./dense-store.js";
import { prebuiltDir } from "./index-store.js";
import { urlLooksLikeBoard, type BoardId } from "./products.js";
import type { IndexedDoc, SearchHit } from "./types.js";

/** Set to `1` to fuse BM25 with local bge-small-zh vectors. Default is BM25 only. */
export const HYBRID_ENV = "RDK_DOCS_HYBRID";

/** bge-small-zh-v1.5 query instruction. Passages are stored without it. */
export const QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章：";
export const QUERY_MAX_LEN = 128;
export const PASSAGE_MAX_LEN = 192;

const BOARDS: BoardId[] = ["x3", "x5", "s100", "s600"];

type OrtModule = typeof import("onnxruntime-node");
type Session = import("onnxruntime-node").InferenceSession;

let ortPromise: Promise<OrtModule> | undefined;
let sessionPromise: Promise<Session> | undefined;
let vectorPromise: Promise<Map<string, LoadedEmbeddings>> | undefined;
let warmPromise: Promise<boolean> | undefined;
let warmupSink = 0;
const warned = new Set<string>();

export function hybridEnabled(): boolean {
  return process.env[HYBRID_ENV] === "1";
}

function runtime(): Promise<OrtModule> {
  if (!ortPromise) ortPromise = import("onnxruntime-node");
  return ortPromise;
}

export function modelPath(): string {
  return join(bgeModelDir(), "model_quantized.onnx");
}

export async function openEmbedSession(threads = 2): Promise<Session> {
  const ort = await runtime();
  return ort.InferenceSession.create(modelPath(), {
    intraOpNumThreads: threads,
    interOpNumThreads: 1,
    graphOptimizationLevel: "all",
    executionProviders: ["cpu"],
  });
}

function session(): Promise<Session> {
  if (!sessionPromise) sessionPromise = openEmbedSession(2);
  return sessionPromise;
}

function i64(values: number[], at: number, into: BigInt64Array): void {
  for (let i = 0; i < values.length; i += 1) into[at + i] = BigInt(values[i] ?? 0);
}

/** L2-normalized CLS vectors, row-major `texts.length * dim`. */
export async function embedNormalized(
  texts: string[],
  maxLen: number,
  runner?: Session,
): Promise<{ dim: number; vectors: Float32Array }> {
  if (texts.length === 0) return { dim: DENSE_DIM, vectors: new Float32Array(0) };
  const ort = await runtime();
  const active = runner ?? (await session());
  const encoded = texts.map((text) => encodeBert(text, maxLen));
  let seq = 2;
  for (const ids of encoded) seq = Math.max(seq, ids.length);
  const n = texts.length;
  const inputIds = new BigInt64Array(n * seq);
  const mask = new BigInt64Array(n * seq);
  const types = new BigInt64Array(n * seq);
  for (let i = 0; i < n; i += 1) {
    const ids = encoded[i] ?? [];
    const at = i * seq;
    i64(ids, at, inputIds);
    for (let j = 0; j < ids.length; j += 1) mask[at + j] = 1n;
  }
  const out = await active.run({
    input_ids: new ort.Tensor("int64", inputIds, [n, seq]),
    attention_mask: new ort.Tensor("int64", mask, [n, seq]),
    token_type_ids: new ort.Tensor("int64", types, [n, seq]),
  });
  const hidden = out.last_hidden_state;
  if (!hidden) throw new Error("bge model did not return last_hidden_state");
  const data = hidden.data as Float32Array;
  const dim = Math.floor(data.length / (n * seq));
  const vectors = new Float32Array(n * dim);
  for (let i = 0; i < n; i += 1) {
    const src = i * seq * dim;
    let norm = 0;
    for (let d = 0; d < dim; d += 1) {
      const value = data[src + d] ?? 0;
      vectors[i * dim + d] = value;
      norm += value * value;
    }
    const inv = norm > 0 ? 1 / Math.sqrt(norm) : 0;
    for (let d = 0; d < dim; d += 1) vectors[i * dim + d] = (vectors[i * dim + d] ?? 0) * inv;
  }
  return { dim, vectors };
}

function loadVectors(): Promise<Map<string, LoadedEmbeddings>> {
  if (!vectorPromise) {
    vectorPromise = Promise.resolve().then(() => {
      const dir = prebuiltDir();
      const tables = new Map<string, LoadedEmbeddings>();
      if (!existsSync(dir)) return tables;
      for (const name of readdirSync(dir)) {
        if (!name.endsWith(".emb.gz")) continue;
        const manualId = name.replace(/\.emb\.gz$/, "");
        tables.set(manualId, decodeEmbeddingBlob(readFileSync(join(dir, name))));
      }
      return tables;
    });
  }
  return vectorPromise;
}

function dotAt(q: Float32Array, vecs: Float32Array, index: number, dim: number): number {
  const base = index * dim;
  let dot = 0;
  for (let d = 0; d < dim; d += 1) dot += (q[d] ?? 0) * (vecs[base + d] ?? 0);
  return dot;
}

/** Run the same cosine used at query time so the first search is not interpreted. */
function warmupDots(tables: Map<string, LoadedEmbeddings>): void {
  const q = new Float32Array(DENSE_DIM);
  q[0] = 1;
  for (const table of tables.values()) {
    const vecs = table.vectors;
    const dim = table.dim;
    for (let i = 0; i < table.count; i += 1) warmupSink += dotAt(q, vecs, i, dim);
  }
}

/** Load the ONNX session and the prebuilt vectors, then compile the embed and scan kernels. */
export function warmHybrid(): Promise<boolean> {
  if (!hybridEnabled()) return Promise.resolve(false);
  if (!warmPromise) {
    warmPromise = (async () => {
      if (!existsSync(modelPath())) throw new Error(`missing embedding model at ${modelPath()}`);
      await session();
      const tables = await loadVectors();
      if (tables.size === 0) throw new Error("no prebuilt .emb.gz vectors");
      await embedNormalized([`${QUERY_PREFIX}warmup`], QUERY_MAX_LEN);
      warmupDots(tables);
      return true;
    })().catch((error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`rdk-docs hybrid disabled: ${message}`);
      return false;
    });
  }
  return warmPromise;
}

function boardsOn(doc: IndexedDoc): BoardId[] {
  return BOARDS.filter((board) => urlLooksLikeBoard(doc.url, board) || urlLooksLikeBoard(doc.title, board));
}

function snippetOf(doc: IndexedDoc): string {
  const crumbs = doc.breadcrumbs?.filter(Boolean).join(" / ");
  const filled = doc.snippet || doc.answer?.slice(0, 180) || doc.text?.slice(0, 180) || crumbs || "";
  const text = filled.trim().slice(0, 240);
  if (text) return text;
  const path = doc.url.split("/").filter(Boolean);
  return path.at(-1) || doc.url;
}

function hitFromDoc(doc: IndexedDoc, score: number): SearchHit {
  const on = boardsOn(doc);
  return {
    title: doc.title,
    url: doc.url,
    manual: doc.manualId,
    snippet: snippetOf(doc),
    score,
    source: "docs",
    board: on.length === 1 ? on[0] : on.length > 1 ? "multiple" : undefined,
    quality: "good",
    confidence: Math.round(Math.max(0, Math.min(1, score)) * 1000) / 1000,
  };
}

export type DenseSearch = { hits: SearchHit[]; topCosine: number };

/** Brute-force cosine over prebuilt chunk vectors. One hit per page, best chunk. */
export async function denseHits(groups: IndexedDoc[][], query: string, limit = 60): Promise<DenseSearch> {
  const tables = await loadVectors();
  const embedded = await embedNormalized([`${QUERY_PREFIX}${query}`], QUERY_MAX_LEN);
  const q = embedded.vectors;
  const dim = embedded.dim;
  const best = new Map<string, { score: number; doc: IndexedDoc }>();
  for (const docs of groups) {
    if (docs.length === 0) continue;
    const manualId = docs[0]?.manualId;
    if (!manualId) continue;
    const table = tables.get(manualId);
    if (!table || table.dim !== dim || table.count !== docs.length || table.fingerprint !== docsFingerprint(docs)) {
      if (!warned.has(manualId)) {
        warned.add(manualId);
        console.error(`rdk-docs hybrid skipped ${manualId}: vectors do not match the loaded index`);
      }
      continue;
    }
    const vecs = table.vectors;
    for (let i = 0; i < docs.length; i += 1) {
      const doc = docs[i];
      if (!doc) continue;
      const dot = dotAt(q, vecs, i, dim);
      const key = doc.url.split("#")[0] ?? doc.url;
      const prev = best.get(key);
      if (!prev || dot > prev.score) best.set(key, { score: dot, doc });
    }
  }
  const ranked = [...best.values()].sort((a, b) => b.score - a.score || a.doc.url.localeCompare(b.doc.url));
  const top = ranked.slice(0, limit);
  return {
    hits: top.map((item) => hitFromDoc(item.doc, item.score)),
    topCosine: top[0]?.score ?? 0,
  };
}
