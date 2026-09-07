import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/client";
import type { ChatJobStatus, ChatResult, NotebookDetail, NotebookSummary, SourceMeta } from "../api/notebookTypes";
import NotebookPage from "./NotebookPage";

const createNotebookMock = vi.fn();
const deleteNotebookMock = vi.fn();
const getNotebookMock = vi.fn();
const listNotebooksMock = vi.fn();
const pollChatJobMock = vi.fn();
const removeSourceMock = vi.fn();
const renameNotebookMock = vi.fn();
const sendChatMessageMock = vi.fn();
const uploadSourceMock = vi.fn();
const pollUploadJobMock = vi.fn();

vi.mock("../api/notebook", () => ({
  createNotebook: (...args: unknown[]) => createNotebookMock(...args),
  deleteNotebook: (...args: unknown[]) => deleteNotebookMock(...args),
  getNotebook: (...args: unknown[]) => getNotebookMock(...args),
  listNotebooks: (...args: unknown[]) => listNotebooksMock(...args),
  pollChatJob: (...args: unknown[]) => pollChatJobMock(...args),
  removeSource: (...args: unknown[]) => removeSourceMock(...args),
  renameNotebook: (...args: unknown[]) => renameNotebookMock(...args),
  sendChatMessage: (...args: unknown[]) => sendChatMessageMock(...args),
  pollUploadJob: (...args: unknown[]) => pollUploadJobMock(...args),
  uploadSource: (...args: unknown[]) => uploadSourceMock(...args),
}));

vi.mock("../context/SettingsContext", () => ({
  useSettings: () => ({
    model: null,
    numCtx: 8192,
    temperatureLevel: "focused",
    embedModel: null,
    hybridTopK: 8,
    maxResults: 6,
    includeCrossref: true,
    chunkSize: 800,
    chunkOverlap: 150,
    useDocling: true,
    useOcr: false,
    largeDocPageThreshold: 50,
  }),
}));

function makeSummary(overrides: Partial<NotebookSummary> = {}): NotebookSummary {
  return {
    notebook_id: "nb-1",
    name: "My Notebook",
    source_count: 0,
    turn_count: 0,
    source_names: [],
    created_at: "2024-01-01T00:00:00Z",
    last_modified: "2024-01-01T00:00:00Z",
    ...overrides,
  };
}

function makeSource(overrides: Partial<SourceMeta> = {}): SourceMeta {
  return {
    doc_id: "doc-1",
    filename: "paper.pdf",
    file_type: "pdf",
    source_type: "upload",
    url: "",
    total_pages: 5,
    total_chunks: 12,
    content_md5: "abc123",
    added_at: "2024-01-01T00:00:00Z",
    ...overrides,
  };
}

function makeDetail(overrides: Partial<NotebookDetail> = {}): NotebookDetail {
  return {
    notebook_id: "nb-1",
    name: "My Notebook",
    source_count: 0,
    turn_count: 0,
    sources: [],
    conversation: [],
    created_at: "2024-01-01T00:00:00Z",
    last_modified: "2024-01-01T00:00:00Z",
    ...overrides,
  };
}

function makeChatResult(overrides: Partial<ChatResult> = {}): ChatResult {
  return {
    notebook_id: "nb-1",
    user_message: "What is the main finding?",
    assistant_response: "The main finding is X [1].",
    citations: [],
    suggested_questions: [],
    source_count: 1,
    retrieval_mode: "hybrid",
    eval_result: {},
    rag_reflection_info: {},
    errors: [],
    progress_pct: 100,
    ...overrides,
  };
}

/** Returns a controllable pollChatJob double for the NEXT call to pollChatJob(). */
function controllablePoll() {
  let onUpdateRef: ((s: ChatJobStatus) => void) | undefined;
  let resolveRef: ((s: ChatJobStatus) => void) | undefined;
  let rejectRef: ((e: unknown) => void) | undefined;
  pollChatJobMock.mockImplementationOnce((_jobId: string, onUpdate: (s: ChatJobStatus) => void) => {
    onUpdateRef = onUpdate;
    return new Promise<ChatJobStatus>((resolve, reject) => {
      resolveRef = resolve;
      rejectRef = reject;
    });
  });
  return {
    update: async (s: ChatJobStatus) => {
      await act(async () => {
        onUpdateRef?.(s);
      });
    },
    settle: async (s: ChatJobStatus) => {
      await act(async () => {
        resolveRef?.(s);
      });
    },
    fail: async (e: unknown) => {
      await act(async () => {
        rejectRef?.(e);
      });
    },
  };
}

describe("NotebookPage", () => {
  beforeEach(() => {
    createNotebookMock.mockReset();
    deleteNotebookMock.mockReset();
    getNotebookMock.mockReset();
    listNotebooksMock.mockReset();
    pollChatJobMock.mockReset();
    removeSourceMock.mockReset();
    renameNotebookMock.mockReset();
    sendChatMessageMock.mockReset();
    uploadSourceMock.mockReset();
    pollUploadJobMock.mockReset();
  });

  it("renders the header, intro, and a default '+ New notebook' selector option", async () => {
    listNotebooksMock.mockResolvedValue([]);
    render(<NotebookPage />);

    expect(screen.getByRole("heading", { name: "Mode 2 — Research Notebook" })).toBeInTheDocument();
    expect(await screen.findByText("+ New notebook")).toBeInTheDocument();
    expect(screen.getByLabelText("Notebook name")).toHaveValue("");
    expect(screen.getByText("Create or select a notebook on the left to begin.")).toBeInTheDocument();
  });

  it("loads notebooks on mount and lists them in the selector", async () => {
    listNotebooksMock.mockResolvedValue([
      makeSummary({ notebook_id: "nb-1", name: "Climate Papers", source_count: 3, turn_count: 5 }),
    ]);
    render(<NotebookPage />);

    expect(await screen.findByText("Climate Papers (3 src, 5 turns)")).toBeInTheDocument();
  });

  it("creates a notebook, switches to it, and loads its empty detail", async () => {
    const user = userEvent.setup();
    listNotebooksMock.mockResolvedValue([]);
    createNotebookMock.mockResolvedValue(makeSummary({ notebook_id: "nb-new", name: "Fresh Notebook" }));
    getNotebookMock.mockResolvedValue(makeDetail({ notebook_id: "nb-new", name: "Fresh Notebook" }));
    render(<NotebookPage />);

    await user.type(screen.getByLabelText("Notebook name"), "Fresh Notebook");
    await user.click(screen.getByRole("button", { name: "Create notebook" }));

    expect(createNotebookMock).toHaveBeenCalledWith("Fresh Notebook");
    expect(await screen.findByRole("heading", { name: "Fresh Notebook" })).toBeInTheDocument();
    expect(screen.getByText(/No sources yet/)).toBeInTheDocument();
  });

  it("shows an error message when the notebook list fails to load", async () => {
    listNotebooksMock.mockRejectedValue(new ApiError(500, "Database unavailable"));
    render(<NotebookPage />);

    expect(await screen.findByText("Database unavailable")).toBeInTheDocument();
  });

  it("warns instead of sending when the notebook has no sources and web search is off", async () => {
    const user = userEvent.setup();
    listNotebooksMock.mockResolvedValue([makeSummary()]);
    getNotebookMock.mockResolvedValue(makeDetail());
    render(<NotebookPage />);

    await user.selectOptions(await screen.findByLabelText("Select a notebook"), "nb-1");
    await screen.findByRole("heading", { name: "My Notebook" });

    await user.type(screen.getByLabelText("Message"), "What does the paper say?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(
      await screen.findByText(/Add at least one source before asking questions/),
    ).toBeInTheDocument();
    expect(sendChatMessageMock).not.toHaveBeenCalled();
  });

  it("uploads a file and lists it as a new source", async () => {
    const user = userEvent.setup();
    listNotebooksMock.mockResolvedValue([makeSummary()]);
    getNotebookMock
      .mockResolvedValueOnce(makeDetail())
      .mockResolvedValueOnce(makeDetail({ sources: [makeSource()], source_count: 1 }));
    uploadSourceMock.mockResolvedValue({ job_id: "job-1" });
    pollUploadJobMock.mockResolvedValue({
      id: "job-1", status: "done", stage: "done", stage_info: {}, error: null,
      result: { added: true, duplicate: false, source: makeSource() },
    });
    render(<NotebookPage />);

    await user.selectOptions(await screen.findByLabelText("Select a notebook"), "nb-1");
    await screen.findByText(/No sources yet/);

    const file = new File(["content"], "paper.pdf", { type: "application/pdf" });
    await user.upload(screen.getByLabelText("Upload sources"), file);

    expect(uploadSourceMock).toHaveBeenCalledWith("nb-1", file, 800, 150, true, false, 50, undefined);
    expect(await screen.findByText("paper.pdf")).toBeInTheDocument();
    expect(screen.getByText("(12 chunks)")).toBeInTheDocument();
  });

  it("shows an info notice when an uploaded file is already in the notebook", async () => {
    const user = userEvent.setup();
    listNotebooksMock.mockResolvedValue([makeSummary()]);
    getNotebookMock.mockResolvedValue(makeDetail());
    uploadSourceMock.mockResolvedValue({ job_id: "job-2" });
    pollUploadJobMock.mockResolvedValue({
      id: "job-2", status: "done", stage: "done", stage_info: {}, error: null,
      result: { added: false, duplicate: true, source: null },
    });
    render(<NotebookPage />);

    await user.selectOptions(await screen.findByLabelText("Select a notebook"), "nb-1");
    await screen.findByText(/No sources yet/);

    const file = new File(["content"], "dup.pdf", { type: "application/pdf" });
    await user.upload(screen.getByLabelText("Upload sources"), file);

    expect(await screen.findByText("Already in this notebook (skipped): dup.pdf")).toBeInTheDocument();
  });

  it("removes a source via its Remove button", async () => {
    const user = userEvent.setup();
    listNotebooksMock.mockResolvedValue([makeSummary()]);
    getNotebookMock
      .mockResolvedValueOnce(makeDetail({ sources: [makeSource()], source_count: 1 }))
      .mockResolvedValueOnce(makeDetail({ sources: [], source_count: 0 }));
    removeSourceMock.mockResolvedValue({ removed: true });
    render(<NotebookPage />);

    await user.selectOptions(await screen.findByLabelText("Select a notebook"), "nb-1");
    await screen.findByText("paper.pdf");

    await user.click(screen.getByRole("button", { name: "Remove" }));

    expect(removeSourceMock).toHaveBeenCalledWith("nb-1", "doc-1");
    expect(await screen.findByText(/No sources yet/)).toBeInTheDocument();
  });

  it("runs the happy path: live stage labels, then the grounded assistant turn with citations, follow-ups, eval and RAG panels", async () => {
    const user = userEvent.setup();
    listNotebooksMock.mockResolvedValue([makeSummary({ source_count: 1 })]);
    getNotebookMock.mockResolvedValue(makeDetail({ sources: [makeSource()], source_count: 1 }));
    sendChatMessageMock.mockResolvedValue({ job_id: "job-1" });
    const poll = controllablePoll();
    render(<NotebookPage />);

    await user.selectOptions(await screen.findByLabelText("Select a notebook"), "nb-1");
    await screen.findByText("paper.pdf");

    await user.type(screen.getByLabelText("Message"), "What is the main finding?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Searching your notebook…")).toBeInTheDocument();
    expect(sendChatMessageMock).toHaveBeenCalledWith({
      notebook_id: "nb-1",
      message: "What is the main finding?",
      include_web_search: false,
      model: null,
      num_ctx: 8192,
      embed_model: null,
      top_k: 8,
      temperature_level: "focused",
    });

    await poll.update({
      id: "job-1",
      status: "running",
      stage: "retrieving",
      stage_info: { label: "Retrieving relevant chunks…" },
      error: null,
      result: null,
    });
    expect(await screen.findByText("Retrieving relevant chunks…")).toBeInTheDocument();

    const result = makeChatResult({
      citations: [{ n: 1, doc_name: "paper.pdf", page: 2, page_label: "p. 2", snippet: "", url: "" }],
      suggested_questions: ["What methodology was used?"],
      eval_result: { overall: 4, summary: "Solid, well-grounded answer." },
      rag_reflection_info: { total_retrieved: 5, total_relevant: 3, query: "main finding" },
    });
    await poll.settle({
      id: "job-1",
      status: "done",
      stage: "done",
      stage_info: {},
      error: null,
      result,
    });

    expect(await screen.findByText("The main finding is X [1].")).toBeInTheDocument();
    expect(screen.getByText("Sources (1)")).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "What methodology was used?" }).length).toBeGreaterThan(0);
    expect(screen.getByText("Quality Score: 4/5 — Good — Solid, well-grounded answer.")).toBeInTheDocument();
    expect(screen.getByText("Self-Reflective RAG — 3/5 items passed grading (60%)")).toBeInTheDocument();
  });

  it("clicking a follow-up question immediately re-asks it", async () => {
    const user = userEvent.setup();
    listNotebooksMock.mockResolvedValue([makeSummary({ source_count: 1 })]);
    getNotebookMock.mockResolvedValue(makeDetail({ sources: [makeSource()], source_count: 1 }));
    sendChatMessageMock.mockResolvedValueOnce({ job_id: "job-1" }).mockResolvedValueOnce({ job_id: "job-2" });
    const poll1 = controllablePoll();
    render(<NotebookPage />);

    await user.selectOptions(await screen.findByLabelText("Select a notebook"), "nb-1");
    await screen.findByText("paper.pdf");
    await user.type(screen.getByLabelText("Message"), "What is the main finding?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await poll1.settle({
      id: "job-1",
      status: "done",
      stage: "done",
      stage_info: {},
      error: null,
      result: makeChatResult({ suggested_questions: ["What methodology was used?"] }),
    });
    const followupButton = (await screen.findAllByRole("button", { name: "What methodology was used?" }))[0];

    const poll2 = controllablePoll();
    await user.click(followupButton);

    expect(sendChatMessageMock).toHaveBeenLastCalledWith({
      notebook_id: "nb-1",
      message: "What methodology was used?",
      include_web_search: false,
      model: null,
      num_ctx: 8192,
      embed_model: null,
      top_k: 8,
      temperature_level: "focused",
    });

    await poll2.settle({
      id: "job-2",
      status: "done",
      stage: "done",
      stage_info: {},
      error: null,
      result: makeChatResult({
        user_message: "What methodology was used?",
        assistant_response: "A randomized trial [1].",
      }),
    });
    expect(await screen.findByText("A randomized trial [1].")).toBeInTheDocument();
  });

  it("shows a Failed status line when the chat job ends in error", async () => {
    const user = userEvent.setup();
    listNotebooksMock.mockResolvedValue([makeSummary({ source_count: 1 })]);
    getNotebookMock.mockResolvedValue(makeDetail({ sources: [makeSource()], source_count: 1 }));
    sendChatMessageMock.mockResolvedValue({ job_id: "job-err" });
    const poll = controllablePoll();
    render(<NotebookPage />);

    await user.selectOptions(await screen.findByLabelText("Select a notebook"), "nb-1");
    await screen.findByText("paper.pdf");
    await user.type(screen.getByLabelText("Message"), "q");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await poll.settle({
      id: "job-err",
      status: "error",
      stage: null,
      stage_info: {},
      error: "Ollama not reachable",
      result: null,
    });

    expect(await screen.findByText("Failed: Ollama not reachable")).toBeInTheDocument();
  });

  it("shows a Failed status line when sendChatMessage itself rejects", async () => {
    const user = userEvent.setup();
    listNotebooksMock.mockResolvedValue([makeSummary({ source_count: 1 })]);
    getNotebookMock.mockResolvedValue(makeDetail({ sources: [makeSource()], source_count: 1 }));
    sendChatMessageMock.mockRejectedValue(new Error("network down"));
    render(<NotebookPage />);

    await user.selectOptions(await screen.findByLabelText("Select a notebook"), "nb-1");
    await screen.findByText("paper.pdf");
    await user.type(screen.getByLabelText("Message"), "q");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Failed: network down")).toBeInTheDocument();
  });

  it("renames the notebook from the Rename / Delete panel", async () => {
    const user = userEvent.setup();
    listNotebooksMock.mockResolvedValue([makeSummary({ name: "Old Name" })]);
    getNotebookMock
      .mockResolvedValueOnce(makeDetail({ name: "Old Name" }))
      .mockResolvedValueOnce(makeDetail({ name: "New Name" }));
    renameNotebookMock.mockResolvedValue(makeSummary({ name: "New Name" }));
    render(<NotebookPage />);

    await user.selectOptions(await screen.findByLabelText("Select a notebook"), "nb-1");
    await screen.findByRole("heading", { name: "Old Name" });

    await user.click(screen.getByText("Rename / Delete"));
    const renameInput = screen.getByLabelText("Notebook name");
    await user.clear(renameInput);
    await user.type(renameInput, "New Name");
    await user.click(screen.getByRole("button", { name: "Save name" }));

    expect(renameNotebookMock).toHaveBeenCalledWith("nb-1", "New Name");
    expect(await screen.findByRole("heading", { name: "New Name" })).toBeInTheDocument();
  });

  it("deletes the notebook and returns to the empty state", async () => {
    const user = userEvent.setup();
    listNotebooksMock.mockResolvedValue([makeSummary()]);
    getNotebookMock.mockResolvedValue(makeDetail());
    deleteNotebookMock.mockResolvedValue({ deleted: true });
    render(<NotebookPage />);

    await user.selectOptions(await screen.findByLabelText("Select a notebook"), "nb-1");
    await screen.findByRole("heading", { name: "My Notebook" });

    await user.click(screen.getByText("Rename / Delete"));
    await user.click(screen.getByRole("button", { name: "Delete notebook" }));

    expect(deleteNotebookMock).toHaveBeenCalledWith("nb-1");
    expect(await screen.findByText("Create or select a notebook on the left to begin.")).toBeInTheDocument();
  });

  it("switches to the Analysis Pipeline tab and shows its no-sources guard when the notebook is empty", async () => {
    const user = userEvent.setup();
    listNotebooksMock.mockResolvedValue([makeSummary()]);
    getNotebookMock.mockResolvedValue(makeDetail());
    render(<NotebookPage />);

    await user.selectOptions(await screen.findByLabelText("Select a notebook"), "nb-1");
    await screen.findByRole("heading", { name: "My Notebook" });

    await user.click(screen.getByRole("tab", { name: "Analysis Pipeline" }));

    expect(
      await screen.findByText("Add at least one source in the Sources panel before running the analysis pipeline."),
    ).toBeInTheDocument();
    expect(screen.queryByLabelText("Message")).not.toBeInTheDocument();
  });

  it("switches to the Analysis Pipeline tab and shows the Run button once a source is present", async () => {
    const user = userEvent.setup();
    listNotebooksMock.mockResolvedValue([makeSummary({ source_count: 1 })]);
    getNotebookMock.mockResolvedValue(makeDetail({ sources: [makeSource()], source_count: 1 }));
    render(<NotebookPage />);

    await user.selectOptions(await screen.findByLabelText("Select a notebook"), "nb-1");
    await screen.findByText("paper.pdf");

    await user.click(screen.getByRole("tab", { name: "Analysis Pipeline" }));
    expect(await screen.findByRole("button", { name: "Run Full Pipeline" })).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Chat" }));
    expect(screen.getByLabelText("Message")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Run Full Pipeline" })).not.toBeInTheDocument();
  });
});
