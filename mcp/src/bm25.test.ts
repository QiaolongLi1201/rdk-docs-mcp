import { describe, expect, it } from "vitest";
import { ABSTAIN_COVERAGE, manualMatchesBoards } from "./bm25.js";
import { matchQuality, rankHits } from "./search.js";
import type { IndexedDoc } from "./types.js";

function doc(partial: Partial<IndexedDoc> & Pick<IndexedDoc, "title" | "url">): IndexedDoc {
  return { manualId: "rdk-x", kind: "page", ...partial };
}

describe("bm25", () => {
  it("ranks a title match above the same words buried in the body", () => {
    const hits = rankHits(
      [
        doc({
          title: "其他说明",
          url: "https://developer.d-robotics.cc/rdk_x_doc/other",
          text: "烧录 烧录 烧录 烧录",
        }),
        doc({ title: "系统烧录", url: "https://developer.d-robotics.cc/rdk_x_doc/flash" }),
      ],
      "烧录",
      5,
    );
    expect(hits[0]?.title).toBe("系统烧录");
  });

  it("matches an identifier with the underscores removed", () => {
    const hits = rankHits(
      [doc({ title: "hobot_dnn 推理", url: "https://developer.d-robotics.cc/rdk_x_doc/hobot_dnn" })],
      "hobotdnn",
      3,
    );
    expect(hits[0]?.url).toContain("hobot_dnn");
    expect(hits[0]?.quality).toBe("good");
  });

  it("drops an S-series OE page when the query names X5", () => {
    const hits = rankHits(
      [
        doc({
          manualId: "oe-s",
          title: "Docker 示例",
          url: "https://developer.d-robotics.cc/oe_s_doc/en/guide/docker",
          text: "docker iptables",
        }),
        doc({
          manualId: "rdk-x",
          title: "X5 快速开始",
          url: "https://developer.d-robotics.cc/rdk_x_doc/Quick_start/rdk_x5",
          text: "RDK X5",
        }),
      ],
      "x5 docker",
      5,
    );
    expect(hits.some((hit) => hit.manual === "oe-s")).toBe(false);
  });

  it("abstains when the query shares no real terms with the corpus", () => {
    const hits = rankHits(
      [doc({ title: "GPIO 应用", url: "https://developer.d-robotics.cc/rdk_x_doc/gpio", text: "40pin 电平" })],
      "excel 数据透视表怎么做",
      5,
    );
    expect(matchQuality(hits).noGoodMatch).toBe(true);
    expect(ABSTAIN_COVERAGE).toBeGreaterThan(0);
    expect(ABSTAIN_COVERAGE).toBeLessThan(1);
  });

  it("keeps one hit from each board when the query names neither", () => {
    const hits = rankHits(
      [
        doc({
          title: "BPU API",
          url: "https://developer.d-robotics.cc/rdk_x_doc/RDK_X3/bpu_api",
          text: "BPU inference",
        }),
        doc({
          title: "BPU API",
          url: "https://developer.d-robotics.cc/rdk_x_doc/RDK_X5/bpu_api",
          text: "BPU inference",
        }),
      ],
      "BPU inference",
      5,
    );
    expect(hits[0]?.board).not.toBe(hits[1]?.board);
    expect(hits.map((hit) => hit.board).sort()).toEqual(["x3", "x5"]);
  });

  it("blocks S-series manuals for an X-only board and the reverse", () => {
    expect(manualMatchesBoards("oe-s", ["x5"])).toBe(false);
    expect(manualMatchesBoards("oe-x5", ["x5"])).toBe(true);
    expect(manualMatchesBoards("oe-x5", ["x3"])).toBe(false);
    expect(manualMatchesBoards("oe-x3", ["x3"])).toBe(true);
    expect(manualMatchesBoards("rdk-x", ["s100"])).toBe(false);
    expect(manualMatchesBoards("tros", ["x3"])).toBe(true);
    expect(manualMatchesBoards("oe-s", ["x5", "s100"])).toBe(true);
  });
});
