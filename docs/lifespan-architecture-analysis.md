# MCP Server Lifespan Architecture: Bundled vs Decoupled Initialization

## The Current Design: One Lifespan, All Dependencies

The MCP server uses a single `crawl4ai_lifespan` context manager that initializes **every dependency** before any tool can respond:

```
crawl4ai_lifespan
  ├── AsyncWebCrawler (headless browser)  ← slow, ~2-5s
  ├── Supabase client (HTTP client)       ← fast, <100ms
  ├── CrossEncoder model (ML model)       ← slow, ~3-4s (now module-level)
  ├── Neo4j validator (DB connection)     ← moderate, ~500ms
  └── Neo4j repo extractor (DB connection)← moderate, ~500ms
```

All five are bundled into a single `Crawl4AIContext` dataclass. Every tool receives the entire bundle, even if it only needs one piece.

## The Problem: Not Every Tool Needs Everything

Here's what each tool actually depends on:

| Tool                           | Browser | Supabase | CrossEncoder | Neo4j |
|--------------------------------|---------|----------|--------------|-------|
| `get_available_sources`        |         | x        |              |       |
| `perform_rag_query`            |         | x        | optional     |       |
| `search_code_examples`         |         | x        | optional     |       |
| `crawl_single_page`           | x       | x        |              |       |
| `smart_crawl_url`             | x       | x        |              |       |
| `check_ai_script_hallucinations` |      |          |              | x     |
| `query_knowledge_graph`        |         |          |              | x     |
| `parse_github_repository`      |         |          |              | x     |

Three of the eight tools (the most commonly called ones for RAG queries) need **only Supabase**. Yet every SSE connection must wait for a headless browser to launch before any of these tools can respond.

## Why Bundling Hurts

### 1. Unnecessary Latency on Lightweight Queries

A `get_available_sources` call is a single Supabase HTTP request — it should resolve in under 200ms. Instead, it waits 2-5 seconds for a Chromium process to spawn, which it will never use.

### 2. Per-Connection Initialization (SSE Transport)

The MCP SDK's SSE transport runs the lifespan **per SSE connection**, not once for the server. This means:

- Every client reconnect triggers a full browser launch
- Each connection holds its own browser process in memory
- The server accumulates browser processes across concurrent clients

### 3. Race Condition on Startup

The bundled lifespan takes long enough that the MCP client sends its first request before initialization completes. The server crashes with:

```
RuntimeError: Received request before initialization was complete
```

This happens because the browser launch is the bottleneck — the client has no way to know the server isn't ready yet.

### 4. Resource Waste

Each SSE connection allocates a full headless browser instance even if the client only ever calls `perform_rag_query`. Browser processes consume ~50-150MB of RAM each.

## How Decoupling Helps

### Module-Level Init for Stateless/Shared Resources

Resources that are thread-safe and don't need per-connection state can be initialized once at module load:

```python
# Loaded once when the Python module is imported
_supabase_client = get_supabase_client()      # HTTP client, thread-safe
_reranking_model = CrossEncoder(...)           # Read-only model, thread-safe
```

These are ready **before the server even starts accepting connections**. Zero wait time per connection.

### Lazy Init for Heavy Resources

The browser doesn't need to exist until a crawl tool is actually called:

```python
_crawler = None

async def _get_crawler():
    global _crawler
    if _crawler is None:
        config = BrowserConfig(headless=True, verbose=False)
        _crawler = AsyncWebCrawler(config=config)
        await _crawler.__aenter__()
    return _crawler
```

This way:
- RAG-only sessions never pay the browser cost
- The browser launches on first crawl request, not on connection
- Only one browser instance exists regardless of connection count

### The Result

```
Before (bundled):
  Client connects → browser launches (3s) → Supabase inits → tools ready
  Total: ~3-5 seconds before first tool response

After (decoupled):
  Client connects → tools ready immediately (Supabase already initialized)
  First crawl request → browser launches on demand
  Total: <200ms for Supabase-only tools
```

## Tradeoffs of Decoupling

| Aspect | Bundled (current) | Decoupled (proposed) |
|--------|-------------------|----------------------|
| Code simplicity | Single context, easy to reason about | Multiple init paths, slightly more complex |
| Cleanup guarantees | Lifespan `finally` block handles all teardown | Must handle cleanup for module-level and lazy resources separately |
| Per-connection isolation | Each connection gets its own crawler instance | Shared crawler means concurrent crawls share one browser |
| Error handling | One place to catch init failures | Init failures can happen at module load or mid-request |
| First-tool latency | Always slow (browser must start) | Fast for non-browser tools, first crawl is slow |
| Memory usage | Browser per connection | Single shared browser |

## Recommendation

Move initialization out of the lifespan in stages:

1. **Already done**: Cross-encoder model moved to module level
2. **Next**: Move Supabase client to module level — fixes the race condition for RAG queries
3. **Optional**: Lazy-init the browser — eliminates the startup bottleneck entirely

The lifespan can remain as a thin wrapper that references the module-level resources, preserving the `Crawl4AIContext` interface so no tool code needs to change.
