import { useEffect, useRef, useState } from "react";
import { ApiError } from "../../api/client";
import { pollSimilarityGraphJob, runSimilarityGraph } from "../../api/paperGraph";
import type { GraphData, PaperNode } from "../../api/paperGraphTypes";
import "../sr/sr-common.css";
import ForceGraph from "./ForceGraph";
import PaperIndexPanel from "./PaperIndexPanel";
import "./PaperGraph.css";

type RunStatus = "idle" | "running" | "done" | "error";

export default function SimilarityGraphPanel() {
  const [paperInput, setPaperInput] = useState("");
  const [topN, setTopN] = useState(30);
  const [bcWeight, setBcWeight] = useState(0.5);
  const [status, setStatus] = useState<RunStatus>("idle");
  const [stage, setStage] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [selectedNode, setSelectedNode] = useState<PaperNode | null>(null);
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

  const ccWeight = parseFloat((1 - bcWeight).toFixed(1));

  async function runGraph(paperId: string) {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setStatus("running");
    setStage("Starting…");
    setError(null);
    setGraph(null);
    setSelectedNode(null);

    try {
      const { job_id } = await runSimilarityGraph({
        paper_id: paperId.trim(),
        top_n: topN,
        bc_weight: bcWeight,
        cc_weight: ccWeight,
      });
      const final = await pollSimilarityGraphJob(
        job_id,
        (s) => {
          if (s.stage_info?.step) setStage(s.stage_info.step as string);
          else if (s.stage) setStage(s.stage);
        },
        ctrl.signal,
      );
      if (final.status === "done" && final.result) {
        setGraph(final.result.graph);
        setStatus("done");
        setStage("Done.");
      } else {
        setError(final.error ?? "Unknown error");
        setStatus("error");
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(err instanceof ApiError ? err.detail : (err as Error).message);
      setStatus("error");
    }
  }

  function handleBuild() {
    const t = paperInput.trim();
    if (!t) return;
    void runGraph(t);
  }

  function handleSetOrigin(node: PaperNode) {
    setPaperInput(node.id);
    void runGraph(node.id);
  }

  return (
    <div className="pg-panel">
      <h3>Similarity Graph</h3>
      <p>
        Enter a Semantic Scholar paper ID or title to map its neighborhood via bibliographic
        coupling and co-citation.
      </p>

      <div className="pg-input-row">
        <input
          type="text"
          placeholder="S2 ID, arXiv ID, DOI, PubMed ID, URL, or title…"
          value={paperInput}
          onChange={(e) => setPaperInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleBuild()}
        />
        <label>
          Top N
          <br />
          <input
            type="range"
            min={5}
            max={100}
            step={5}
            value={topN}
            onChange={(e) => setTopN(Number(e.target.value))}
          />
          {" "}{topN}
        </label>
        <label>
          BC weight
          <br />
          <input
            type="range"
            min={0}
            max={1}
            step={0.1}
            value={bcWeight}
            onChange={(e) => setBcWeight(Number(e.target.value))}
          />
          {" "}{bcWeight.toFixed(1)} / CC {ccWeight.toFixed(1)}
        </label>
        <button
          type="button"
          className="sr-button"
          disabled={status === "running" || !paperInput.trim()}
          onClick={handleBuild}
        >
          {status === "running" ? "Building…" : "Build Graph"}
        </button>
      </div>

      {status === "running" && <p className="pg-status">{stage}</p>}
      {status === "error" && <p className="pg-error">{error}</p>}

      {graph && (() => {
        const nodeNumbers = new Map(graph.nodes.map((n, i) => [n.id, i + 1]));
        return (
          <>
            <div style={{ display: "flex", gap: "1rem", alignItems: "flex-start" }}>
              <div ref={containerRef} className="pg-graph-container" style={{ flex: "1 1 0", minWidth: 0 }}>
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
                onSetAsOrigin={handleSetOrigin}
                height={480}
              />
            </div>

            <div className="pg-legend">
              <span className="pg-legend-item">
                <span className="pg-legend-swatch" style={{ background: "rgb(107,111,106)" }} />
                Older papers
              </span>
              <span className="pg-legend-item">
                <span className="pg-legend-swatch" style={{ background: "rgb(184,134,11)" }} />
                Recent papers
              </span>
              <span className="pg-legend-item">
                Node size = citation count
              </span>
            </div>
            <p className="pg-status">
              {graph.nodes.length} papers · {graph.edges.length} edges
            </p>
          </>
        );
      })()}
    </div>
  );
}
