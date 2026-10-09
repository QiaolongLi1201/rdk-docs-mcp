import { describe, expect, it } from "vitest";
import { decodeEmbeddingBlob, docsFingerprint, encodeEmbeddingBlob, fuseByRrf, gzipEmbeddingBlob } from "./dense-store.js";
import type { IndexedDoc, SearchHit } from "./types.js";

function doc(url: string, title: string): IndexedDoc {
  return { manualId: "rdk-x", title, url, kind: "heading" };
}

function hit(url: string, title: string, score: number): SearchHit {
  return { title, url, manual: "rdk-x", snippet: title, score, source: "docs" };
}

describe("dense store", () => {
  it("round-trips int8 vectors and keeps cosine high", () => {
    const count = 3;
    const dim = 8;
    const matrix = new Float32Array(count * dim);
    for (let i = 0; i < matrix.length; i += 1) matrix[i] = Math.sin(i + 1) * 0.2;
    const blob = gzipEmbeddingBlob(encodeEmbeddingBlob(0xabc, matrix, count, dim));
    const loaded = decodeEmbeddingBlob(blob);
    expect(loaded.count).toBe(count);
    expect(loaded.dim).toBe(dim);
    expect(loaded.fingerprint).toBe(0xabc);
    let dot = 0;
    let left = 0;
    let right = 0;
    for (let i = 0; i < matrix.length; i += 1) {
      const a = matrix[i] ?? 0;
      const b = loaded.vectors[i] ?? 0;
      dot += a * b;
      left += a * a;
      right += b * b;
    }
    const cosine = dot / Math.sqrt(left * right);
    expect(cosine).toBeGreaterThan(0.99);
  });

  it("fingerprints doc order", () => {
    const a = docsFingerprint([doc("https://example.test/a", "甲")]);
    const b = docsFingerprint([doc("https://example.test/a", "乙")]);
    expect(a).not.toBe(b);
  });

  it("fuses reciprocal ranks and keeps the BM25 hit", () => {
    const bm = [hit("https://example.test/a#q", "bm-title", 0.9), hit("https://example.test/b", "other", 0.2)];
    const dense = [hit("https://example.test/b", "dense-title", 0.8), hit("https://example.test/a", "dense-a", 0.7)];
    const fused = fuseByRrf(bm, dense, 60, 60);
    expect(fused.map((item) => item.url.split("#")[0])).toEqual(["https://example.test/a", "https://example.test/b"]);
    expect(fused[0]?.title).toBe("bm-title");
    expect(fused[0]?.score).toBeCloseTo(1 / 60 + 1 / 61, 6);
  });
});
