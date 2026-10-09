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
  /** Fraction of query concepts found on this page. */
  coverage?: number;
  /**
   * 0–1 overlap of the query's distinctive terms with this page, mixed with
   * how strong the score is for this corpus. Callers can threshold this.
   * `noGoodMatch` stays conservative and does not follow a low value by itself.
   */
  confidence?: number;
  /** Set when the hit came from an alias because the literal identifier is not indexed. */
  matchedVia?: "alias";
};

export type ResultBoard = "x3" | "x5" | "s100" | "s600" | "agnostic" | "multiple";

export type BoardGroup = {
  board: ResultBoard;
  hits: SearchHit[];
};
