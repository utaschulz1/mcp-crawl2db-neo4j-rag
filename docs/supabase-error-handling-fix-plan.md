# Implementation Plan: Supabase Error Handling Fix

## Goal

Replace silent swallowing of Supabase errors with meaningful return values so that
`crawl_single_page` (and other callers) can report actual storage outcomes instead of
hardcoded success.

## Scope

- `src/utils.py` — three storage functions
- `src/crawl4ai_mcp.py` — `crawl_single_page` response block (and the equivalent block
  in `smart_crawl_url` at ~line 652–722)

---

## Step 1 — `update_source_info()` → return `bool`

**File:** `src/utils.py`, lines 597–628

**Change:** Return `True` on success, `False` on exception.

```python
def update_source_info(client, source_id, summary, word_count) -> bool:
    try:
        result = client.table('sources').update({...}).eq('source_id', source_id).execute()
        if not result.data:
            client.table('sources').insert({...}).execute()
            print(f"Created new source: {source_id}")
        else:
            print(f"Updated source: {source_id}")
        return True          # <-- add
    except Exception as e:
        print(f"Error updating source {source_id}: {e}")
        return False         # <-- change from implicit None
```

---

## Step 2 — `add_documents_to_supabase()` → return `tuple[int, bool]`

**File:** `src/utils.py`, lines 167–316

**Return value:** `(successful_inserts: int, delete_ok: bool)`

### 2a — Track delete outcome

Wrap the existing batch-delete block and set `delete_ok`:

```python
delete_ok = True
try:
    if unique_urls:
        client.table("crawled_pages").delete().in_("url", unique_urls).execute()
except Exception as e:
    print(f"Batch delete failed: {e}. Trying one-by-one deletion as fallback.")
    delete_ok = True   # reset; track individual failures below
    for url in unique_urls:
        try:
            client.table("crawled_pages").delete().eq("url", url).execute()
        except Exception as inner_e:
            print(f"Error deleting record for URL {url}: {inner_e}")
            delete_ok = False   # at least one delete failed
```

### 2b — Track insert count

Add `successful_inserts = 0` before the batch loop. In the existing retry/fallback
logic, increment it wherever an insert succeeds:

- Batch insert succeeds → `successful_inserts += len(batch_data)`
- Individual insert fallback succeeds → `successful_inserts += 1`

### 2c — Return

Change the function signature and add a return at the end:

```python
def add_documents_to_supabase(...) -> tuple[int, bool]:
    ...
    return successful_inserts, delete_ok
```

---

## Step 3 — `add_code_examples_to_supabase()` → return `tuple[int, bool]`

**File:** `src/utils.py`, lines 488–595

Same pattern as Step 2.

### 3a — Track delete outcome

```python
delete_ok = True
for url in unique_urls:
    try:
        client.table('code_examples').delete().eq('url', url).execute()
    except Exception as e:
        print(f"Error deleting existing code examples for {url}: {e}")
        delete_ok = False   # <-- was just print, now also sets flag
```

### 3b — Track insert count

Add `successful_inserts = 0` before the batch loop. In the retry block:

- Batch insert succeeds → `successful_inserts += len(batch_data)`
- Individual insert fallback succeeds → `successful_inserts += 1`

### 3c — Return

```python
def add_code_examples_to_supabase(...) -> tuple[int, bool]:
    ...
    return successful_inserts, delete_ok
```

---

## Step 4 — Update `crawl_single_page` response

**File:** `src/crawl4ai_mcp.py`, lines ~458–524

### 4a — Capture return values

```python
source_ok = update_source_info(supabase_client, source_id, source_summary, total_word_count)

chunks_stored, docs_delete_ok = add_documents_to_supabase(
    supabase_client, urls, chunk_numbers, contents, metadatas, url_to_full_document
)

code_stored, code_delete_ok = 0, True
if extract_code_examples and code_blocks:
    code_stored, code_delete_ok = add_code_examples_to_supabase(
        supabase_client, code_urls, code_chunk_numbers,
        code_examples, code_summaries, code_metadatas
    )

delete_ok = docs_delete_ok and code_delete_ok
```

### 4b — Replace hardcoded success response

```python
return json.dumps({
    "success": source_ok and chunks_stored > 0,
    "url": url,
    "chunks_stored": chunks_stored,                         # actual DB count
    "code_examples_stored": code_stored,                    # actual DB count
    "delete_ok": delete_ok,                                 # new field
    "content_length": len(result.markdown),
    "total_word_count": total_word_count,
    "source_id": source_id,
    "links_count": {
        "internal": len(result.links.get("internal", [])),
        "external": len(result.links.get("external", []))
    }
}, indent=2)
```

---

## Step 5 — Apply same fix to `smart_crawl_url`

**File:** `src/crawl4ai_mcp.py`, lines ~652–722

`smart_crawl_url` calls the same three storage functions with the same hardcoded
success pattern. Apply Steps 4a–4b identically to that response block.

---

## Summary of Changes

| File | Lines affected | Change |
|---|---|---|
| `src/utils.py` | ~193–205 | Track `delete_ok` in batch delete |
| `src/utils.py` | ~212–316 | Track `successful_inserts`, return `tuple[int, bool]` |
| `src/utils.py` | ~514–518 | Track `delete_ok` in code example delete loop |
| `src/utils.py` | ~568–594 | Track `successful_inserts`, return `tuple[int, bool]` |
| `src/utils.py` | ~626–627 | Return `bool` from `update_source_info` |
| `src/crawl4ai_mcp.py` | ~460–524 | Capture return values, fix response in `crawl_single_page` |
| `src/crawl4ai_mcp.py` | ~652–722 | Same fix in `smart_crawl_url` |

**Estimated lines changed:** ~35

---

## What is not changed

- Exception handling philosophy — errors are still caught and printed, not re-raised
- Function signatures for callers that ignore the return value (they will continue to work)
- Delete failures still do not abort the insert; `delete_ok: false` in the response is
  informational only
