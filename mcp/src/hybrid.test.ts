import { afterEach, describe, expect, it } from "vitest";
import { embedNormalized } from "./hybrid.js";
import { readPrebuilt } from "./index-store.js";
import { searchManualsHybrid } from "./search.js";

function cosine(left: Float32Array, right: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < left.length; i += 1) dot += (left[i] ?? 0) * (right[i] ?? 0);
  return dot;
}

describe("bge-small-zh", () => {
  it("puts a black-screen sentence nearer a camera sentence than a GPIO sentence", async () => {
    const encoded = await embedNormalized(
      ["摄像头黑屏怎么办", "MIPI 摄像头没有画面，黑屏", "GPIO 引脚电平怎么读"],
      64,
    );
    const dim = encoded.dim;
    const row = (index: number) => encoded.vectors.subarray(index * dim, (index + 1) * dim);
    const related = cosine(row(0), row(1));
    const unrelated = cosine(row(0), row(2));
    expect(related).toBeGreaterThan(unrelated);
    expect(related).toBeGreaterThan(0.5);
  }, 30_000);

  it("fuses packaged vectors when RDK_DOCS_HYBRID=1", async () => {
    process.env.RDK_DOCS_HYBRID = "1";
    const docs = readPrebuilt("rdk-x") ?? [];
    expect(docs.length).toBeGreaterThan(1000);
    const hits = await searchManualsHybrid([docs], "摄像头黑屏", 3);
    expect(hits.length).toBeGreaterThan(0);
    // Reciprocal-rank scores stay near 1/60. The BM25-only head is about 0.1.
    expect(hits[0]?.score ?? 1).toBeLessThan(0.05);
  }, 30_000);
});

afterEach(() => {
  delete process.env.RDK_DOCS_HYBRID;
});
