import { describe, expect, it } from "vitest";
import { encodeBert } from "./bert-tokenizer.js";

const fixtures: Array<{ text: string; ids: number[] }> = [
  { text: "摄像头黑屏", ids: [101, 3029, 1008, 1928, 7946, 2242, 102] },
  {
    text: "为这个句子生成表示以用于检索相关文章：USB 摄像头没有 /dev/video 节点",
    ids: [
      101, 711, 6821, 702, 1368, 2094, 4495, 2768, 6134, 4850, 809, 4500, 754, 3466, 5164, 4685, 1068, 3152, 4995, 8038,
      100, 3029, 1008, 1928, 3766, 3300, 120, 8363, 8225, 120, 9539, 5688, 4157, 102,
    ],
  },
  { text: "WiFi 连不上", ids: [101, 100, 6825, 679, 677, 102] },
  { text: "hrt_model_exec", ids: [101, 8967, 8165, 142, 9264, 142, 9464, 8177, 102] },
  { text: "AttributeError", ids: [101, 100, 102] },
  { text: "Q1：开发板没有 /dev/video0", ids: [101, 100, 8038, 2458, 1355, 3352, 3766, 3300, 120, 8363, 8225, 120, 9539, 8129, 102] },
  { text: "hello, world!", ids: [101, 8701, 117, 8572, 106, 102] },
  { text: "GPIO 40pin", ids: [101, 100, 8164, 13248, 102] },
  { text: "  多个   空格\n换行 ", ids: [101, 1914, 702, 4958, 3419, 2940, 6121, 102] },
  { text: "sun55iw3", ids: [101, 8482, 8949, 8169, 8220, 8152, 102] },
];

describe("bge wordpiece", () => {
  it("matches the published Bert tokenizer, lowercase off", () => {
    for (const fixture of fixtures) {
      expect(encodeBert(fixture.text, 128), fixture.text).toEqual(fixture.ids);
    }
  });

  it("truncates to max length including specials", () => {
    const ids = encodeBert("摄像头黑屏怎么办", 4);
    expect(ids).toHaveLength(4);
    expect(ids[0]).toBe(101);
    expect(ids.at(-1)).toBe(102);
  });
});
