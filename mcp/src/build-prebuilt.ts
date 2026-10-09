import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { gzipSync } from "node:zlib";
import { listManuals } from "./catalog.js";
import { fetchText } from "./http.js";
import { prebuiltDir } from "./index-store.js";
import { loadIndexFromOrigin } from "./service.js";
import type { IndexedDoc } from "./types.js";

function compact(doc: IndexedDoc): IndexedDoc {
  const out: IndexedDoc = {
    manualId: doc.manualId,
    title: doc.title,
    url: doc.url,
    kind: doc.kind,
  };
  if (doc.snippet) out.snippet = doc.snippet;
  if (doc.text) out.text = doc.text;
  if (doc.breadcrumbs?.length) out.breadcrumbs = doc.breadcrumbs;
  return out;
}

async function main(): Promise<void> {
  const dir = prebuiltDir();
  mkdirSync(dir, { recursive: true });
  const manuals = listManuals().filter((manual) => manual.searchable);
  let failed = 0;
  for (const manual of manuals) {
    const started = Date.now();
    try {
      const docs = (await loadIndexFromOrigin(manual, fetchText)).map(compact);
      if (docs.length === 0) {
        process.stderr.write(`empty\t${manual.id}\n`);
        failed += 1;
        continue;
      }
      const body = gzipSync(Buffer.from(JSON.stringify(docs)));
      writeFileSync(join(dir, `${manual.id}.json.gz`), body);
      process.stderr.write(`ok\t${manual.id}\t${docs.length}\t${body.length}\t${Date.now() - started}ms\n`);
    } catch (error) {
      failed += 1;
      process.stderr.write(`fail\t${manual.id}\t${error instanceof Error ? error.message : String(error)}\n`);
    }
  }
  if (failed > 0) process.exitCode = 1;
}

main().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
});
