# Plan: Decouple MCP Server Lifespan Initialization

**Goal**: Eliminate the per-connection browser startup bottleneck so Supabase-only tools respond instantly, and the "Received request before initialization was complete" race condition no longer occurs.

**Reference**: See `lifespan-architecture-analysis.md` for full background.

---

## Task 1: Move Supabase client to module level

**File**: `src/crawl4ai_mcp.py`

Move `get_supabase_client()` out of `crawl4ai_lifespan` to module level, same pattern as the cross-encoder fix.

```python
# After load_dotenv and _reranking_model initialization
_supabase_client = get_supabase_client()
```

In `crawl4ai_lifespan`, replace `supabase_client = get_supabase_client()` with:

```python
supabase_client = _supabase_client
```

**Why**: The Supabase client is a stateless HTTP client. No reason to create one per connection. This alone fixes the race condition for the three most-called tools (`get_available_sources`, `perform_rag_query`, `search_code_examples`).

---

## Task 2: Lazy-init the AsyncWebCrawler

**File**: `src/crawl4ai_mcp.py`

Replace eager browser launch in the lifespan with a lazy singleton:

```python
_crawler = None
_crawler_lock = asyncio.Lock()

async def _get_crawler() -> AsyncWebCrawler:
    global _crawler
    if _crawler is None:  # fast path — no lock on every call once initialised
        async with _crawler_lock:
            if _crawler is None:  # double-check after acquiring lock
                print("Launching browser (lazy init)...")
                config = BrowserConfig(headless=True, verbose=False)
                _crawler = AsyncWebCrawler(config=config)
                await _crawler.__aenter__()
    return _crawler
```

Update crawl tools (`crawl_single_page`, `smart_crawl_url`) to call `await _get_crawler()` instead of reading from context.

**Why**: The browser is only needed by 2 of 8 tools. Lazy init means:
- Supabase-only sessions never pay the 2-5s browser cost
- Only one browser instance exists across all connections
- The lifespan becomes near-instant, eliminating the race condition entirely

**Tradeoff**: The first crawl request will still take 2-5s extra for browser startup. This is acceptable since crawling is already a long operation.

---

## Task 3: Slim down the lifespan to a thin passthrough

**File**: `src/crawl4ai_mcp.py`

After Tasks 1-2, the lifespan only needs to:
- Reference module-level resources
- Optionally init Neo4j (if enabled)
- Handle Neo4j cleanup on disconnect

```python
@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    knowledge_validator = None
    repo_extractor = None

    if os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true":
        # ... Neo4j init (unchanged) ...

    try:
        yield Crawl4AIContext(
            supabase_client=_supabase_client,
            reranking_model=_reranking_model,
            knowledge_validator=knowledge_validator,
            repo_extractor=repo_extractor,
        )
    finally:
        # Neo4j cleanup only — browser and supabase are module-level singletons
        if knowledge_validator:
            await knowledge_validator.close()
        if repo_extractor:
            await repo_extractor.close()
```

**Why**: The lifespan completes in milliseconds. No browser, no model load, no Supabase init. The race condition is gone.

---

## Task 4: Update Crawl4AIContext dataclass

**File**: `src/crawl4ai_mcp.py`

Remove `crawler` from the dataclass since crawl tools now get it via `_get_crawler()`:

```python
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    supabase_client: Client
    reranking_model: Optional[CrossEncoder] = None
    knowledge_validator: Optional[Any] = None
    repo_extractor: Optional[Any] = None
```

Update `crawl_single_page` and `smart_crawl_url` to use:
```python
crawler = await _get_crawler()
```
instead of:
```python
crawler = ctx.request_context.lifespan_context.crawler
```

---

## Task 5: Browser cleanup via lifespan teardown

**File**: `src/crawl4ai_mcp.py`

~~`atexit` is the wrong tool here.~~ `atexit` runs synchronously after the event loop may already be closed, causing `RuntimeError: Event loop is closed`. Since the lifespan is still present (just slimmed down), use its teardown section instead — it runs in the live async context when the server shuts down (Ctrl+C).

Add browser cleanup to the lifespan's `finally` block (from Task 3):

```python
    finally:
        # Neo4j cleanup
        if knowledge_validator:
            await knowledge_validator.close()
        if repo_extractor:
            await repo_extractor.close()
        # Browser cleanup — only if a crawl was ever requested
        global _crawler
        if _crawler is not None:
            print("Closing browser...")
            await _crawler.__aexit__(None, None, None)
            _crawler = None  # reset so a future connection can re-init if needed
```

**Why `_crawler = None` after close**: The lifespan runs per-connection. Without the reset, a subsequent connection would receive the already-closed crawler object from `_get_crawler()` (the `is None` check would pass it straight through). Resetting ensures the next caller re-initialises cleanly.

**Why not `atexit`**: `atexit` callbacks run synchronously; calling `run_until_complete` on a loop that is already stopped or closed raises an error and leaves Chromium running anyway.

**Multi-connection note**: For a personal MCP server with a single active client at a time this is safe. In a scenario with overlapping connections, the first teardown closes the browser while another connection may still be mid-crawl. That edge case is out of scope for this project.

## Task 6: Test and verify

1. Start the server: `uv run src/crawl4ai_mcp.py`
2. Confirm startup logs show no "Loading weights" and no browser launch
3. Call `get_available_sources` — should respond in <500ms with no browser started
4. Call `crawl_single_page` — should trigger browser launch on first call
5. Connect multiple SSE clients — confirm no duplicate browser/model/supabase inits
6. Verify the "Received request before initialization was complete" error no longer occurs

---

## Execution Order

```
Task 1 (Supabase module-level)
  └── Task 2 (Lazy crawler)
        └── Task 3 (Slim lifespan)
              └── Task 4 (Update dataclass)
                    └── Task 5 (Browser cleanup)
                          └── Task 6 (Test)
```

Tasks 1 and 2 are independent of each other and could be done in parallel. Tasks 2, 3, and 5 are tightly coupled — the lazy crawler, the slimmed lifespan, and the teardown cleanup must land together as one coherent change. Task 4 (dataclass) is a straightforward follow-on once Task 2 is in.
