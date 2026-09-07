/**
 * TypeScript mirrors of backend/app/schemas/notebook.py.
 * Keep field names/optionality in sync with those Pydantic models.
 */

export type TemperatureLevel = "precise" | "focused" | "balanced" | "creative";

export interface NotebookSummary {
  notebook_id: string;
  name: string;
  source_count: number;
  turn_count: number;
  source_names: string[];
  created_at: string;
  last_modified: string;
}

export interface SourceMeta {
  doc_id: string;
  filename: string;
  file_type: string;
  source_type: string;
  url: string;
  total_pages: number;
  total_chunks: number;
  content_md5: string;
  added_at: string;
}

export interface CitationItem {
  n: number;
  doc_name: string;
  page: number;
  page_label: string;
  snippet: string;
  url: string;
}

export interface ConversationTurn {
  role: string;
  content: string;
  timestamp: string;
  citations: CitationItem[] | null;
  suggested_questions: string[] | null;
}

export interface SavedReview {
  doc_id: string;
  doc_filename: string;
  review_text: string;
  external_refs: ExternalRef[];
  generated_at: string;
}

export interface ExternalRef {
  ref_num?: string;
  title?: string;
  authors?: string[];
  year?: number | null;
  url?: string;
  source?: string;
  abstract_snippet?: string;
}

export interface NotebookDetail {
  notebook_id: string;
  name: string;
  source_count: number;
  turn_count: number;
  sources: SourceMeta[];
  conversation: ConversationTurn[];
  created_at: string;
  last_modified: string;
  saved_reviews?: Record<string, SavedReview>;
  reviewer_chats?: Record<string, Array<{ role: string; content: string }>>;
}

export interface UploadSourceResult {
  added: boolean;
  duplicate: boolean;
  source: SourceMeta | null;
}

export interface RemoveSourceResult {
  removed: boolean;
}

export interface DeleteNotebookResult {
  deleted: boolean;
}

export interface ChatRequest {
  notebook_id: string;
  message: string;
  include_web_search?: boolean;
  model?: string | null;
  num_ctx?: number | null;
  embed_model?: string | null;
  top_k?: number | null;
  temperature_level?: TemperatureLevel | null;
}

export interface ChatResult {
  notebook_id: string;
  user_message: string;
  assistant_response: string;
  citations: CitationItem[];
  suggested_questions: string[];
  source_count: number;
  retrieval_mode: string;
  eval_result: Record<string, unknown>;
  rag_reflection_info: Record<string, unknown>;
  errors: string[];
  progress_pct: number;
}

export type JobStatusValue = "queued" | "running" | "done" | "error";

export interface JobCreated {
  job_id: string;
}

export interface ChatJobStatus {
  id: string;
  status: JobStatusValue;
  stage: string | null;
  stage_info: Record<string, unknown>;
  error: string | null;
  result: ChatResult | null;
}

export interface UploadJobStatus {
  id: string;
  status: JobStatusValue;
  stage: string | null;
  stage_info: Record<string, unknown>;
  error: string | null;
  result: UploadSourceResult | null;
}
