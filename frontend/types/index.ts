/**
 * types/index.ts
 *
 * Public types shared across components, hooks, and API helpers. Every type
 * here MUST match the FastAPI backend's response/request shapes — if you
 * change a field, update the corresponding Pydantic model in api/routes/.
 */

// ── Chat messages ─────────────────────────────────────────────────────────────

export type MessageRole = "user" | "assistant" | "system" | "tool";

export interface ChatMessage {
  /** Stable client-side id; backend doesn't currently assign one. */
  id:        string;
  role:      MessageRole;
  /** Markdown-formatted content. Streaming messages may have partial content. */
  content:   string;
  /** ISO 8601 timestamp (client-stamped on send / receive). */
  createdAt: string;
  /** Optional indicator that this message produced new plots. */
  hasPlots?: boolean;
  /** Skills/tools that fired while producing this message (badge display). */
  skills?:   SkillBadge[];
  /** Artifacts extracted from the message body (code, plot, table, mermaid). */
  artifacts?: Artifact[];
  /** True while the assistant is still streaming this message. */
  streaming?: boolean;
}

// ── Skill / tool routing badges ───────────────────────────────────────────────

export type SkillName =
  | "ingestion"
  | "biomarker"
  | "enrichment"
  | "visualization"
  | "code_reviewer"
  | "domain_expert"
  | "query_data"
  | "query_database"
  | "load_preview_data"
  | "complex_duckdb_query"
  | "simple_dataframe_query";

export interface SkillBadge {
  name:      SkillName;
  /** Display label, e.g. "Proteomics Analysis". */
  label:     string;
  /** ISO timestamp when the skill fired. */
  startedAt: string;
  /** Optional finish timestamp; absent while in progress. */
  endedAt?:  string;
  /** "running" | "done" | "error". */
  status:    "running" | "done" | "error";
}

// ── Artifacts (rendered in the right panel) ───────────────────────────────────

export type ArtifactKind = "code" | "plot" | "table" | "mermaid";

interface ArtifactBase {
  id:        string;
  kind:      ArtifactKind;
  /** Human-readable title shown above the artifact, e.g. "Volcano plot". */
  title?:    string;
  /** ISO 8601 timestamp. */
  createdAt: string;
}

export interface CodeArtifact extends ArtifactBase {
  kind:     "code";
  language: string;            // "python", "sql", "r", "json", …
  code:     string;
}

export interface PlotArtifact extends ArtifactBase {
  kind:        "plot";
  /** URL of the static image (PNG) served by the backend. */
  imageUrl?:   string;
  /** URL of the interactive HTML version. */
  htmlUrl?:    string;
  /** Plotly figure JSON for in-app rendering. */
  plotlyJson?: object;
}

export interface TableArtifact extends ArtifactBase {
  kind:     "table";
  columns:  string[];
  rows:     Array<Record<string, unknown>>;
}

export interface MermaidArtifact extends ArtifactBase {
  kind:    "mermaid";
  source:  string;
}

export type Artifact =
  | CodeArtifact
  | PlotArtifact
  | TableArtifact
  | MermaidArtifact;

// ── File upload ──────────────────────────────────────────────────────────────

export interface UploadedFile {
  /** Server-assigned id used to reference this file in subsequent chat turns. */
  fileId:     string;
  filename:   string;
  /** Bytes. */
  size:       number;
  /** Detected dataset type, e.g. "olink_npx" | "ms_lfq" | "ms_tmt" | "generic". */
  dataType?:  string;
  /** Detected vendor software, e.g. "MaxQuant" | "FragPipe". */
  software?:  string;
  rowCount?:  number;
  colCount?:  number;
  uploadedAt: string;
  /** True while the upload is in flight. */
  uploading?: boolean;
  /** If upload failed, the human-readable error. */
  error?:     string;
}

// ── Sessions ──────────────────────────────────────────────────────────────────

export interface SessionSummary {
  id:           string;
  /** First user message (truncated) used as a list-item title. */
  title:        string;
  createdAt:    string;
  lastActiveAt: string;
  messageCount: number;
}

export interface SessionDetail extends SessionSummary {
  messages: ChatMessage[];
  files:    UploadedFile[];
}

// ── SSE stream events (frontend ↔ backend) ────────────────────────────────────
//
// The backend emits these as named SSE events. Frontend code parses them
// into the union below. New events should be added here AND to the backend
// stream emitter in lockstep.

export type StreamEvent =
  | { type: "token";    sessionId: string; delta: string }
  | { type: "skill";    sessionId: string; badge: SkillBadge }
  | { type: "artifact"; sessionId: string; artifact: Artifact }
  | { type: "message_complete"; sessionId: string; message: ChatMessage }
  | { type: "error";    sessionId: string; error: string }
  | { type: "done";     sessionId: string };

// ── Theme ─────────────────────────────────────────────────────────────────────

export type Theme = "light" | "dark";
