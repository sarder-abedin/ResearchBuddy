import { useEffect, useRef, useState } from "react";
import { ApiError } from "../../api/client";
import { createCollection, expandCollection, pollExpandJob } from "../../api/paperGraph";
import type { ExpandRelationship, GraphData, PaperNode } from "../../api/paperGraphTypes";
import "../sr/sr-common.css";
import ForceGraph from "./ForceGraph";
import PaperIndexPanel from "./PaperIndexPanel";
import "./PaperGraph.css";

type InitStatus = "idle" | "creating" | "ready" | "error";
type ExpandStatus = "idle" | "expanding" | "error";

export default function DiscoveryNetworkPanel() {
  const [seedInput, setSeedInput] = useState("");
  const [seeds, setSeeds] = useState<string[]>([]);
  const [initStatus, setInitStatus] = useState<InitStatus>("idle");
  const [collectionId, setCollectionId] = useState<string | null>(null);
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [initError, setInitError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<PaperNode | null>(null);
  const [expandStatus, setExpandStatus] = useState<ExpandStatus>("idle");
  const [expandError, setExpandError] = useState<string | null>(null);
  const [expandRelationship, setExpandRelationship] = useState<ExpandRelationship>("later");
  const abortRef = useRef<AbortController | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [graphWidth, setGraphWidth] = useState(680);

  useEffect(() => {
    if (!containerRef.current) return;
    const obs = new ResizeObserver((entries) => {
      setGraphWidth(Math.max(200, entries[0].contentRect.width));
    });
    obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  function addSeed() {
    const t = seedInput.trim();
    if (t && !seeds.includes(t)) setSeeds((prev) => [...prev, t]);
    setSeedInput("");
  }

  function removeSeed(s: string) {
    setSeeds((prev) => prev.filter((x) => x !== s));
  }

  async function handleCreate() {
    if (seeds.length === 0) return;
    setInitStatus("creating");
    setInitError(null);
    try {
      const resp = await createCollection({ seed_paper_ids: seeds });
      setCollectionId(resp.collection_id);
      setGraph(resp.graph);
      setInitStatus("ready");
    } catch (err) {
      setInitError(err instanceof ApiError ? err.detail : (err as Error).message);
      setInitStatus("error");
    }
  }

  async function handleExpand(nodeId: string, relationship: ExpandRelationship) {
    if (!collectionId) return;
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setExpandStatus("expanding");
    setExpandError(null);

    try {
      const { job_id } = await expandCollection(collectionId, { node_id: nodeId, relationship });
      const final = await pollExpandJob(collectionId, job_id, () => {}, ctrl.signal);
      if (final.status === "done" && final.result) {
        setGraph(final.result.graph);
        setExpandStatus("idle");
      } else {
        setExpandError(final.error ?? "Expand failed");
        setExpandStatus("error");
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setExpandError(err instanceof ApiError ? err.detail : (err as Error).message);
      setExpandStatus("error");
    }
  }

  function handleReset() {
    abortRef.current?.abort();
    setInitStatus("idle");
    setGraph(null);
    setCollectionId(null);
    setSeeds([]);
    setSelectedNode(null);
    setExpandStatus("idle");
    setExpandError(null);
    setSeedInput("");
  }

  return (
    <div className="pg-panel">
      <h3>Discovery Network</h3>
      <p>
        Build a persistent paper collection and incrementally explore its neighborhood by
        relationship type. Click any node to expand it.
      </p>

      {initStatus !== "ready" && (
        <>
          <div className="pg-input-row">
            <input
              type="text"
              placeholder="S2 ID, arXiv ID, DOI, PubMed ID, URL, or title…"
              value={seedInput}
              onChange={(e) => setSeedInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addSeed()}
            />
            <button type="button" className="sr-button" onClick={addSeed}>
              Add seed
            </button>
          </div>

          {seeds.length > 0 && (
            <div className="pg-seeds">
              {seeds.map((s) => (
                <span key={s} className="pg-seed-chip">
                  {s.length > 40 ? s.slice(0, 38) + "…" : s}
                  <button
                    type="button"
                    aria-label={`Remove ${s}`}
                    style={{
                      marginLeft: "0.3rem",
                      background: "none",
                      border: "none",
                      cursor: "pointer",
                      padding: 0,
                      color: "inherit",
                      lineHeight: 1,
                    }}
                    onClick={() => removeSeed(s)}
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          )}

          <div style={{ marginTop: "0.5rem" }}>
            <button
              type="button"
              className="sr-button"
              disabled={seeds.length === 0 || initStatus === "creating"}
              onClick={handleCreate}
            >
              {initStatus === "creating" ? "Creating…" : "Create Collection"}
            </button>
          </div>

          {initStatus === "error" && <p className="pg-error">{initError}</p>}
        </>
      )}

      {graph && (() => {
        const nodeNumbers = new Map(graph.nodes.map((n, i) => [n.id, i + 1]));
        return (
          <>
          <div style={{ display: "flex", gap: "1rem", alignItems: "flex-start" }}>
            <div
              ref={containerRef}
              className="pg-graph-container"
              style={{ flex: "1 1 0", minWidth: 0, position: "relative" }}
            >
              {graph.partial && graph.notice && (
                <div className="pg-graph-notice">{graph.notice}</div>
              )}
              <ForceGraph
                graph={graph}
                onNodeClick={setSelectedNode}
                selectedNodeId={selectedNode?.id}
                nodeNumbers={nodeNumbers}
                width={graphWidth}
                height={480}
              />
            </div>
            <PaperIndexPanel
              nodes={graph.nodes}
              nodeNumbers={nodeNumbers}
              selectedNodeId={selectedNode?.id ?? null}
              onSelect={setSelectedNode}
              onExpand={handleExpand}
              expandRelationship={expandRelationship}
              onExpandRelationshipChange={setExpandRelationship}
              expanding={expandStatus === "expanding"}
              height={480}
            />
          </div>

          {expandStatus === "expanding" && (
            <p className="pg-status">Fetching related papers…</p>
          )}
          {expandStatus === "error" && <p className="pg-error">{expandError}</p>}

          <p className="pg-status">
            {graph.nodes.length} papers · {graph.edges.length} edges
            {collectionId && ` · Collection ${collectionId.slice(0, 8)}…`}
          </p>

          <button
            type="button"
            className="sr-button"
            style={{ marginTop: "0.5rem", fontSize: "0.8rem" }}
            onClick={handleReset}
          >
            New collection
          </button>
          </>
        );
      })()}
    </div>
  );
}
