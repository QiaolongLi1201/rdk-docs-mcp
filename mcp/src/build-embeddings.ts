/**
 * Encode packaged chunks with the local bge-small-zh ONNX model.
 * Reads mcp/prebuilt/*.json.gz and writes mcp/prebuilt/<manual>.emb.gz.
 * No network. Re-run after `npm run build:index` replaces those snapshots.
 *
 *   npx vite-node src/build-embeddings.ts
 */
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { listManuals } from "./catalog.js";
import { docsFingerprint, encodeEmbeddingBlob, gzipEmbeddingBlob, passageText } from "./dense-store.js";
import { embedNormalized, openEmbedSession, PASSAGE_MAX_LEN, QUERY_PREFIX } from "./hybrid.js";
import { prebuiltDir, readPrebuilt } from "./index-store.js";

const BATCH = 16;

async function main(): Promise<void> {
  const started = Date.now();
  const session = await openEmbedSession(4);
  const dir = prebuiltDir();
  const manuals = listManuals().filter((manual) => manual.searchable);
  const written: Array<{ manualId: string; count: number; bytes: number; fingerprint: number }> = [];
  for (const manual of manuals) {
    const docs = readPrebuilt(manual.id);
    if (!docs || docs.length === 0) {
      process.stderr.write(`skip\t${manual.id}\n`);
      continue;
    }
    const t0 = Date.now();
    const texts = docs.map(passageText);
    let dim = 0;
    const chunks: Float32Array[] = [];
    for (let at = 0; at < texts.length; at += BATCH) {
      const part = await embedNormalized(texts.slice(at, at + BATCH), PASSAGE_MAX_LEN, session);
      dim = part.dim;
      chunks.push(part.vectors);
      if (at > 0 && at % (BATCH * 40) === 0) process.stderr.write(`embed\t${manual.id}\t${at}/${texts.length}\n`);
    }
    const vectors = new Float32Array(docs.length * dim);
    let offset = 0;
    for (const chunk of chunks) {
      vectors.set(chunk, offset);
      offset += chunk.length;
    }
    const fingerprint = docsFingerprint(docs);
    const blob = gzipEmbeddingBlob(encodeEmbeddingBlob(fingerprint, vectors, docs.length, dim));
    writeFileSync(join(dir, `${manual.id}.emb.gz`), blob);
    written.push({ manualId: manual.id, count: docs.length, bytes: blob.length, fingerprint });
    process.stderr.write(`ok\t${manual.id}\t${docs.length}\t${blob.length}\t${Date.now() - t0}ms\n`);
  }
  const manifest = {
    model: "BAAI/bge-small-zh-v1.5",
    onnx: "Xenova/bge-small-zh-v1.5 onnx/model_quantized.onnx",
    pooling: "cls",
    normalized: true,
    dtype: "int8",
    queryPrefix: QUERY_PREFIX,
    passageMaxLen: PASSAGE_MAX_LEN,
    builtAt: new Date().toISOString(),
    manuals: written,
  };
  writeFileSync(join(dir, "embeddings.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  const bytes = written.reduce((sum, item) => sum + item.bytes, 0);
  process.stdout.write(`embeddings ${written.length} manuals ${bytes} bytes ${Date.now() - started}ms\n`);
}

main().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
});
