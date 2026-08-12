import { useCallback } from "react";
import ForceGraph2D from "react-force-graph-2d";
import type { GraphData, PaperNode } from "../../api/paperGraphTypes";

interface FGNode {
  id: string;
  title: string;
  year: number | null;
  citation_count: number | null;
  _paper: PaperNode;
  // Populated by react-force-graph-2d during layout
  x?: number;
  y?: number;
}

interface FGLink {
  source: string;
  target: string;
  weight: number;
  edge_type: string;
}

interface ForceGraphProps {
  graph: GraphData;
  onNodeClick: (paper: PaperNode) => void;
  selectedNodeId?: string | null;
  nodeNumbers?: Map<string, number>;
  width?: number;
  height?: number;
}

// Year → colour gradient: older = muted blue, newer = warm amber (accent colour)
function yearToColor(year: number | null): string {
  if (year == null) return "#999999";
  const minY = 2000;
  const maxY = new Date().getFullYear();
  const t = Math.max(0, Math.min(1, (year - minY) / (maxY - minY)));
  // Interpolate from #6b6f6a (muted) to #b8860b (accent)
  const r = Math.round(107 + t * (184 - 107));
  const g = Math.round(111 + t * (134 - 111));
  const b = Math.round(106 + t * (11 - 106));
  return `rgb(${r},${g},${b})`;
}

// Node radius scaled by log(citation_count) so highly-cited papers are larger
function nodeRadius(citationCount: number | null): number {
  if (citationCount == null || citationCount <= 0) return 5;
  return Math.min(20, 4 + Math.log10(citationCount + 1) * 4);
}

export default function ForceGraph({
  graph,
  onNodeClick,
  selectedNodeId,
  nodeNumbers,
  width = 680,
  height = 480,
}: ForceGraphProps) {
  const fgNodes: FGNode[] = graph.nodes.map((p) => ({
    id: p.id,
    title: p.title,
    year: p.year,
    citation_count: p.citation_count,
    _paper: p,
  }));

  const fgLinks: FGLink[] = graph.edges.map((e) => ({
    source: e.source,
    target: e.target,
    weight: e.weight,
    edge_type: e.edge_type,
  }));

  const paintNode = useCallback(
    (node: FGNode, ctx: CanvasRenderingContext2D) => {
      const baseR = nodeRadius(node.citation_count);
      // Enforce a minimum so the number always fits inside
      const r = Math.max(8, baseR);
      const isSelected = node.id === selectedNodeId;
      const cx = node.x ?? 0;
      const cy = node.y ?? 0;

      // Circle fill
      ctx.beginPath();
      ctx.arc(cx, cy, isSelected ? r + 3 : r, 0, 2 * Math.PI);
      ctx.fillStyle = yearToColor(node.year);
      ctx.fill();

      // Selection ring
      if (isSelected) {
        ctx.strokeStyle = "#b8860b";
        ctx.lineWidth = 2.5;
        ctx.stroke();
      }

      // Number label centered inside the circle
      const num = nodeNumbers ? (nodeNumbers.get(node.id) ?? "") : "";
      if (num !== "") {
        const digits = String(num).length;
        const fontSize = digits > 2 ? Math.max(6, r * 0.6) : Math.max(7, r * 0.75);
        ctx.font = `bold ${fontSize}px sans-serif`;
        ctx.fillStyle = "#1a1a1a";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(num), cx, cy);
        ctx.textBaseline = "alphabetic";
      }
    },
    [selectedNodeId, nodeNumbers],
  );

  const linkColor = useCallback((link: FGLink) => {
    switch (link.edge_type) {
      case "reference": return "rgba(100,130,200,0.5)";
      case "citation":  return "rgba(80,160,100,0.5)";
      case "co_author": return "rgba(180,80,180,0.5)";
      case "recommendation": return "rgba(200,140,50,0.5)";
      default:          return `rgba(184,134,11,${Math.max(0.1, link.weight * 0.7)})`;
    }
  }, []);

  const linkWidth = useCallback(
    (link: FGLink) => (link.edge_type === "similarity" ? Math.max(0.5, link.weight * 3) : 1),
    [],
  );

  return (
    <ForceGraph2D
      graphData={{ nodes: fgNodes, links: fgLinks }}
      width={width}
      height={height}
      nodeId="id"
      nodeCanvasObject={paintNode}
      nodeCanvasObjectMode={() => "replace"}
      linkColor={linkColor}
      linkWidth={linkWidth}
      linkDirectionalArrowLength={(link: FGLink) =>
        link.edge_type === "reference" || link.edge_type === "citation" ? 4 : 0
      }
      linkDirectionalArrowRelPos={1}
      onNodeClick={(node: FGNode) => onNodeClick(node._paper)}
      nodeLabel={(node: FGNode) => node.title}
      d3VelocityDecay={0.3}
      cooldownTicks={100}
    />
  );
}
