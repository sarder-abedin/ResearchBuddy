import { apiFetch } from "./client";
import type {
  ChatJobStatus,
  ChatRequest,
  ConversationTurn,
  DeleteNotebookResult,
  JobCreated,
  NotebookDetail,
  NotebookSummary,
  RemoveSourceResult,
  UploadJobStatus,
  UploadSourceResult,
} from "./notebookTypes";

const POLL_INTERVAL_MS = 700;
const BASE = "/api/notebook";

function pollUntilTerminal<T extends { status: string }>(
  fetchStatus: () => Promise<T>,
  onUpdate: (status: T) => void,
  signal?: AbortSignal,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    let timeoutId: ReturnType<typeof setTimeout> | undefined;

    const onAbort = () => {
      if (timeoutId !== undefined) clearTimeout(timeoutId);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort);
    const cleanup = () => signal?.removeEventListener("abort", onAbort);

    const tick = async () => {
      if (signal?.aborted) return;
      try {
        const status = await fetchStatus();
        if (signal?.aborted) return;
        onUpdate(status);
        if (status.status === "done" || status.status === "error") {
          cleanup();
          resolve(status);
          return;
        }
        timeoutId = setTimeout(tick, POLL_INTERVAL_MS);
      } catch (err) {
        cleanup();
        reject(err);
      }
    };

    void tick();
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Notebook CRUD
// ─────────────────────────────────────────────────────────────────────────────

export function createNotebook(name: string): Promise<NotebookSummary> {
  return apiFetch<NotebookSummary>(`${BASE}/notebooks`, {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

export function listNotebooks(): Promise<NotebookSummary[]> {
  return apiFetch<NotebookSummary[]>(`${BASE}/notebooks`);
}

export function getNotebook(notebookId: string): Promise<NotebookDetail> {
  return apiFetch<NotebookDetail>(`${BASE}/notebooks/${notebookId}`);
}

export function deleteNotebook(notebookId: string): Promise<DeleteNotebookResult> {
  return apiFetch<DeleteNotebookResult>(`${BASE}/notebooks/${notebookId}`, {
    method: "DELETE",
  });
}

export function renameNotebook(notebookId: string, newName: string): Promise<NotebookSummary> {
  return apiFetch<NotebookSummary>(`${BASE}/notebooks/${notebookId}/rename`, {
    method: "POST",
    body: JSON.stringify({ new_name: newName }),
  });
}

export function getHistory(notebookId: string, maxTurns = 8): Promise<ConversationTurn[]> {
  return apiFetch<ConversationTurn[]>(`${BASE}/notebooks/${notebookId}/history?max_turns=${maxTurns}`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Source upload / removal
// ─────────────────────────────────────────────────────────────────────────────

/** Multipart upload -- apiFetch's request() skips the default JSON Content-Type
 * for FormData bodies so the browser can set its own multipart boundary. */
export function uploadSource(
  notebookId: string,
  file: File,
  chunkSize?: number | null,
  chunkOverlap?: number | null,
  useDocling?: boolean,
  useOcr?: boolean,
  largeDocPageThreshold?: number,
  visionModel?: string,
): Promise<JobCreated> {
  const formData = new FormData();
  formData.append("file", file);
  if (chunkSize != null) formData.append("chunk_size", String(chunkSize));
  if (chunkOverlap != null) formData.append("chunk_overlap", String(chunkOverlap));
  if (useDocling != null) formData.append("use_docling", String(useDocling));
  if (useOcr != null) formData.append("use_ocr", String(useOcr));
  if (largeDocPageThreshold != null) formData.append("large_doc_page_threshold", String(largeDocPageThreshold));
  if (visionModel) formData.append("vision_model", visionModel);
  return apiFetch<JobCreated>(`${BASE}/notebooks/${notebookId}/sources`, {
    method: "POST",
    body: formData,
  });
}

export function getUploadJobStatus(
  notebookId: string,
  jobId: string,
): Promise<UploadJobStatus> {
  return apiFetch<UploadJobStatus>(
    `${BASE}/notebooks/${notebookId}/sources/jobs/${jobId}`,
  );
}

/** Poll an upload job to completion. Uploads run in the background because
 * Docling conversion plus per-figure vision captioning takes minutes -- far
 * too long to hold the HTTP request open. */
export function pollUploadJob(
  notebookId: string,
  jobId: string,
  onUpdate: (status: UploadJobStatus) => void,
  signal?: AbortSignal,
): Promise<UploadJobStatus> {
  return pollUntilTerminal(
    () => getUploadJobStatus(notebookId, jobId),
    onUpdate,
    signal,
  );
}

export function removeSource(notebookId: string, docId: string): Promise<RemoveSourceResult> {
  return apiFetch<RemoveSourceResult>(`${BASE}/notebooks/${notebookId}/sources/${docId}`, {
    method: "DELETE",
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat (background job + polling, same pattern as Mode 1 / Mode 3)
// ─────────────────────────────────────────────────────────────────────────────

export function sendChatMessage(req: ChatRequest): Promise<JobCreated> {
  return apiFetch<JobCreated>(`${BASE}/chat`, {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export function getChatJobStatus(jobId: string): Promise<ChatJobStatus> {
  return apiFetch<ChatJobStatus>(`${BASE}/jobs/${jobId}`);
}

/** Poll a notebook chat job until it reaches a terminal status ("done" | "error"). */
export function pollChatJob(
  jobId: string,
  onUpdate: (status: ChatJobStatus) => void,
  signal?: AbortSignal,
): Promise<ChatJobStatus> {
  return pollUntilTerminal(() => getChatJobStatus(jobId), onUpdate, signal);
}
