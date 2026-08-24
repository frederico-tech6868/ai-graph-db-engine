# Graph DB Engine — Web UI

A FastAPI backend + single-page management console for the graph database
engine. It runs fully offline using the deterministic `LocalEmbedder` (no API
key needed) and lets you set up and manage the graph from the browser.

## Features

- **Dashboard** — live stats + interactive graph visualization (vis-network),
  color-coded by label. Click a node to inspect it.
- **Nodes** — create / list / delete nodes; attach a `text` field that is
  embedded automatically for similarity search.
- **Edges** — add edges with a **label-scoped similarity scan before adding**;
  any near-duplicate existing edges are surfaced with their combined score.
- **Similarity Search** — cosine-similarity vector search, optionally scoped to
  a single label (e.g. compare a `User` only to other `User` nodes).
- **AI Memory** — remember / recall / reflect over agent memories, with
  near-duplicate detection and an entity list.
- **Admin** — seed demo data, save/load to disk, reset.

## Run

```bash
cd /home/ubuntu/graphdb
pip install -r requirements.txt          # includes fastapi, uvicorn, pydantic
python -m webui.run                      # serves on http://0.0.0.0:3000
```

Then open <http://localhost:3000>. Click **Seed demo** to populate example data.

The graph is persisted to `webui_graph.json` (override with the
`GRAPHDB_WEB_PATH` environment variable).

## REST API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/stats` | Node/edge/label counts + memory stats |
| GET | `/api/graph?label=&limit=` | Full graph for visualization |
| GET | `/api/nodes?label=&limit=` | List nodes |
| GET | `/api/nodes/{id}` | Node detail + edges |
| POST | `/api/nodes` | Create node (`{label, properties, text?}`) |
| PUT | `/api/nodes/{id}` | Update node properties |
| DELETE | `/api/nodes/{id}` | Delete node (cascades edges) |
| GET | `/api/edges` | List edges |
| POST | `/api/edges` | Add edge — runs the label-scoped similarity scan |
| DELETE | `/api/edges/{id}` | Delete edge |
| POST | `/api/search` | Label-scoped vector search (`{text, label?, k?}`) |
| GET | `/api/traverse/{id}?edge_label=&max_depth=` | BFS traversal |
| GET | `/api/path?src_id=&dst_id=` | Shortest path |
| GET | `/api/memory/stats` | Memory stats |
| POST | `/api/memory/remember` | Store a memory (dedup-checked) |
| POST | `/api/memory/recall` | Recall memories |
| POST | `/api/memory/reflect` | Synthesize a reflection |
| GET | `/api/memory/entities` | List entities |
| POST | `/api/save` `/api/load` `/api/reset` `/api/seed` | Admin |

## Architecture

```
webui/
  server.py    FastAPI app + REST endpoints
  service.py   GraphService — wraps GraphStore + AgentMemory + embedder
  models.py    Pydantic request models
  run.py       uvicorn entry point (0.0.0.0:3000)
  static/      index.html + styles.css + app.js  (no build step)
```

The frontend is dependency-free vanilla JS; the only CDN dependency is
`vis-network` for the graph view.
