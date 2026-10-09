import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { gzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { prebuiltDir, readPrebuilt } from "./index-store.js";

describe("readPrebuilt", () => {
  it("returns undefined when the manual snapshot is absent", () => {
    expect(readPrebuilt("zz-missing-manual")).toBeUndefined();
  });

  it("inflates a gzipped snapshot", () => {
    const dir = prebuiltDir();
    mkdirSync(dir, { recursive: true });
    const id = "zz-test-manual";
    const path = join(dir, `${id}.json.gz`);
    const docs = [
      {
        manualId: id,
        title: "GPIO",
        url: "https://example.test/gpio",
        kind: "page",
        text: "gpio",
      },
    ];
    writeFileSync(path, gzipSync(Buffer.from(JSON.stringify(docs))));
    try {
      expect(readPrebuilt(id)?.[0]?.title).toBe("GPIO");
    } finally {
      rmSync(path, { force: true });
    }
  });
});
