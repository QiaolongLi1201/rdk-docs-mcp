import { gzipSync, gunzipSync } from "node:zlib";
import type { IndexedDoc, SearchHit } from "./types.js";

export const EMB_MAGIC = "RDKE";
export const EMB_VERSION = 1;
export const DENSE_DIM = 512;
/** Reciprocal-rank fusion constant. Rank 0 is the best rank. */
export const RRF_K = 60;
export const RRF_LIST = 60;

const fingerprints = new WeakMap<IndexedDoc[], number>();

export function passageText(doc: IndexedDoc): string {
  const title = doc.title.replace(/\s+/g, " ").trim();
  const body = (doc.answer || doc.text || doc.snippet || "").replace(/\s+/g, " ").trim();
  if (!body || body === title) return title;
  return `${title} ${body}`;
}

export function docsFingerprint(docs: IndexedDoc[]): number {
  const cached = fingerprints.get(docs);
  if (cached !== undefined) return cached;
  let hash = 0x811c9dc5;
  for (const doc of docs) {
    const row = `${doc.url}\t${doc.title}\t${doc.kind}\n`;
    for (let i = 0; i < row.length; i += 1) {
      hash ^= row.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193);
    }
  }
  const out = hash >>> 0;
  fingerprints.set(docs, out);
  return out;
}

export function quantizeVectors(matrix: Float32Array, count: number, dim: number): { scales: Float32Array; bytes: Int8Array } {
  const scales = new Float32Array(count);
  const bytes = new Int8Array(count * dim);
  for (let i = 0; i < count; i += 1) {
    const base = i * dim;
    let max = 0;
    for (let d = 0; d < dim; d += 1) max = Math.max(max, Math.abs(matrix[base + d] ?? 0));
    const scale = max > 0 ? max / 127 : 1;
    scales[i] = scale;
    for (let d = 0; d < dim; d += 1) {
      const q = Math.round((matrix[base + d] ?? 0) / scale);
      bytes[base + d] = Math.max(-127, Math.min(127, q));
    }
  }
  return { scales, bytes };
}

export function dequantizeVectors(scales: Float32Array, bytes: Int8Array, dim: number): Float32Array {
  const count = scales.length;
  const out = new Float32Array(count * dim);
  for (let i = 0; i < count; i += 1) {
    const scale = scales[i] ?? 1;
    const base = i * dim;
    for (let d = 0; d < dim; d += 1) out[base + d] = (bytes[base + d] ?? 0) * scale;
  }
  return out;
}

export function encodeEmbeddingBlob(fingerprint: number, matrix: Float32Array, count: number, dim: number): Buffer {
  const { scales, bytes } = quantizeVectors(matrix, count, dim);
  const header = 16;
  const buf = Buffer.alloc(header + count * 4 + bytes.length);
  buf.write(EMB_MAGIC, 0, "ascii");
  buf.writeUInt16LE(EMB_VERSION, 4);
  buf.writeUInt16LE(dim, 6);
  buf.writeUInt32LE(count, 8);
  buf.writeUInt32LE(fingerprint >>> 0, 12);
  for (let i = 0; i < count; i += 1) buf.writeFloatLE(scales[i] ?? 1, header + i * 4);
  Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).copy(buf, header + count * 4);
  return buf;
}

export type LoadedEmbeddings = {
  fingerprint: number;
  count: number;
  dim: number;
  vectors: Float32Array;
};

export function decodeEmbeddingBlob(raw: Buffer): LoadedEmbeddings {
  const body = raw.length >= 2 && raw[0] === 0x1f && raw[1] === 0x8b ? gunzipSync(raw) : raw;
  const magic = body.toString("ascii", 0, 4);
  if (magic !== EMB_MAGIC) throw new Error("embedding file magic mismatch");
  const version = body.readUInt16LE(4);
  if (version !== EMB_VERSION) throw new Error(`embedding file version ${version}`);
  const dim = body.readUInt16LE(6);
  const count = body.readUInt32LE(8);
  const fingerprint = body.readUInt32LE(12);
  const header = 16;
  const scales = new Float32Array(count);
  for (let i = 0; i < count; i += 1) scales[i] = body.readFloatLE(header + i * 4);
  const start = header + count * 4;
  const bytes = new Int8Array(count * dim);
  for (let i = 0; i < bytes.length; i += 1) bytes[i] = body.readInt8(start + i);
  return { fingerprint, count, dim, vectors: dequantizeVectors(scales, bytes, dim) };
}

export function gzipEmbeddingBlob(blob: Buffer): Buffer {
  return gzipSync(blob);
}

/**
 * Equal-weight reciprocal rank fusion. A page present in only one list still
 * scores. When both lists contain a page, the BM25 hit keeps its metadata.
 */
export function fuseByRrf(bm: SearchHit[], dense: SearchHit[], k = RRF_K, list = RRF_LIST): SearchHit[] {
  const map = new Map<string, { hit: SearchHit; score: number }>();
  const add = (rows: SearchHit[], fromBm: boolean) => {
    const n = Math.min(rows.length, list);
    for (let rank = 0; rank < n; rank += 1) {
      const hit = rows[rank];
      if (!hit) continue;
      const key = hit.url.split("#")[0] ?? hit.url;
      const inc = 1 / (k + rank);
      const prev = map.get(key);
      if (!prev) {
        map.set(key, { hit, score: inc });
        continue;
      }
      prev.score += inc;
      if (fromBm) prev.hit = hit;
    }
  };
  add(dense, false);
  add(bm, true);
  const fused = [...map.values()].map(({ hit, score }) => ({ ...hit, score }));
  fused.sort((a, b) => b.score - a.score || a.url.localeCompare(b.url));
  return fused;
}
