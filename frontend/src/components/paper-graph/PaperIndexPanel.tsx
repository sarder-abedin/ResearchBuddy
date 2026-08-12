import { useEffect, useRef } from "react";
import type { ExpandRelationship, PaperNode } from "../../api/paperGraphTypes";

interface PaperIndexPanelProps {
  nodes: PaperNode[];
  nodeNumbers: Map<string, number>;
  selectedNodeId: string | null;
  onSelect: (paper: PaperNode) => void;
  height?: number;
  /** Feature 1: re-centre the graph on this paper */
  onSetAsOrigin?: (node: PaperNode) => void;
  /** Feature 2: expand this node's neighbourhood */
  onExpand?: (nodeId: string, relationship: ExpandRelationship) => void;
  expandRelationship?: ExpandRelationship;
  onExpandRelationshipChange?: (r: ExpandRelationship) => void;
  expanding?: boolean;
}

const RELATIONSHIP_LABELS: Record<ExpandRelationship, string> = {
  earlier: "Earlier work",
  later: "Later work",
  similar: "Similar papers",
  authors: "Author network",
};

// Match the same gradient used in ForceGraph so badges are consistent
function yearToColor(year: number | null): string {
  if (year == null) return "#999999";
  const minY = 2000;
  const maxY = new Date().getFullYear();
  const t = Math.max(0, Math.min(1, (year - minY) / (maxY - minY)));
  const r = Math.round(107 + t * (184 - 107));
  const g = Math.round(111 + t * (134 - 111));
  const b = Math.round(106 + t * (11 - 106));
  return `rgb(${r},${g},${b})`;
}

export default function PaperIndexPanel({
  nodes,
  nodeNumbers,
  selectedNodeId,
  onSelect,
  height = 480,
  onSetAsOrigin,
  onExpand,
  expandRelationship = "later",
  onExpandRelationshipChange,
  expanding = false,
}: PaperIndexPanelProps) {
  const itemRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  // Scroll selected item into view when selection changes from the graph side
  useEffect(() => {
    if (selectedNodeId) {
      itemRefs.current.get(selectedNodeId)?.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    }
  }, [selectedNodeId]);

  return (
    <div className="pg-index-panel" style={{ height }}>
      {nodes.map((paper) => {
        const num = nodeNumbers.get(paper.id) ?? "?";
        const isSelected = paper.id === selectedNodeId;
        const meta = [
          paper.authors.slice(0, 2).join(", ") + (paper.authors.length > 2 ? " et al." : ""),
          paper.year ?? "n.d.",
        ]
          .filter(Boolean)
          .join(" · ");

        return (
          <div
            key={paper.id}
            ref={(el) => {
              if (el) itemRefs.current.set(paper.id, el);
              else itemRefs.current.delete(paper.id);
            }}
            className={`pg-index-item${isSelected ? " pg-index-item--selected" : ""}`}
            onClick={() => onSelect(paper)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && onSelect(paper)}
            aria-pressed={isSelected}
          >
            <span
              className="pg-index-num"
              style={{ background: yearToColor(paper.year) }}
            >
              {num}
            </span>

            <div className="pg-index-body">
              <div className="pg-index-title">
                {paper.title || "(Untitled)"}
              </div>
              <div className="pg-index-meta">
                {meta}
                {paper.citation_count != null && (
                  <> · {paper.citation_count.toLocaleString()} citations</>
                )}
              </div>

              {isSelected && (
                <div className="pg-index-detail">
                  {paper.abstract ? (
                    <p className="pg-index-abstract">
                      {paper.abstract.slice(0, 300)}
                      {paper.abstract.length > 300 ? "…" : ""}
                    </p>
                  ) : (
                    <p className="pg-index-abstract pg-index-abstract--na">Abstract unavailable</p>
                  )}

                  <div className="pg-index-actions">
                    {paper.url && (
                      <a
                        href={paper.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="pg-index-link"
                        onClick={(e) => e.stopPropagation()}
                      >
                        Open ↗
                      </a>
                    )}
                    {onSetAsOrigin && (
                      <button
                        type="button"
                        className="sr-button sr-button--sm"
                        onClick={(e) => { e.stopPropagation(); onSetAsOrigin(paper); }}
                      >
                        Set as origin
                      </button>
                    )}
                  </div>

                  {onExpand && (
                    <div className="pg-expand-row">
                      <label htmlFor={`pg-rel-${paper.id}`}>Explore:</label>
                      <select
                        id={`pg-rel-${paper.id}`}
                        value={expandRelationship}
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => {
                          e.stopPropagation();
                          onExpandRelationshipChange?.(e.target.value as ExpandRelationship);
                        }}
                      >
                        {(Object.keys(RELATIONSHIP_LABELS) as ExpandRelationship[]).map((r) => (
                          <option key={r} value={r}>{RELATIONSHIP_LABELS[r]}</option>
                        ))}
                      </select>
                      <button
                        type="button"
                        className="sr-button sr-button--sm"
                        disabled={expanding}
                        onClick={(e) => {
                          e.stopPropagation();
                          onExpand(paper.id, expandRelationship);
                        }}
                      >
                        {expanding ? "Expanding…" : "Expand"}
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
