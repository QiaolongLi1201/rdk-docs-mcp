export type IndexedDoc = {
  manualId: string;
  title: string;
  url: string;
  snippet?: string;
  text?: string;
  breadcrumbs?: string[];
  kind: "page" | "heading" | "snippet";
};

export type HitRole = "official-start" | "related" | "forum-supplement";

export type SearchHit = {
  title: string;
  url: string;
  manual: string;
  snippet: string;
  score: number;
  source: "docs" | "forum";
  role?: HitRole;
  /** Board the URL or title is scoped to, when that is unambiguous. */
  board?: "x3" | "x5" | "s100" | "s600" | "multiple";
  quality?: "good" | "weak";
  /** Set when the hit came from an alias because the literal identifier is not indexed. */
  matchedVia?: "alias";
};
