# Supabase Error Handling Issues in crawl4ai-rag MCP Server

## Problem Summary

The `crawl_single_page` tool reports `"success": true` and `"chunks_stored": 9` even when Supabase is completely unreachable. All Supabase operations silently swallow exceptions, and the response is hardcoded to report success based on in-memory chunk counts rather than actual database inserts.

## Root Cause: Hardcoded Success Response

**File:** `src/crawl4ai_mcp.py`, lines ~502-514

The return value always reports success regardless of whether anything was actually stored:

```python
return json.dumps({
    "success": True,              # Always True — never set to False on storage failure
    "chunks_stored": len(chunks), # Based on in-memory list, not actual DB inserts
    "code_examples_stored": len(code_blocks) if code_blocks else 0,  # Same issue
    ...
})
```

## Three Functions That Silently Swallow Errors

### 1. `update_source_info()` — `src/utils.py`, lines ~597-628

```python
def update_source_info(client, source_id, summary, word_count):
    try:
        result = client.table('sources').update({...}).eq('source_id', source_id).execute()
        if not result.data:
            client.table('sources').insert({...}).execute()
    except Exception as e:
        print(f"Error updating source {source_id}: {e}")  # Swallowed — no re-raise, returns None
```

- Returns `None` always (no return value)
- Exception is only printed to console
- Caller at `crawl4ai_mcp.py:450` has no way to detect failure

### 2. `add_documents_to_supabase()` — `src/utils.py`, lines ~167-316

- Delete operations fail silently (lines ~193-205): catches exception, prints, continues
- Batch insert retries, then falls back to individual inserts (lines ~290-315)
- If ALL inserts fail (batch and individual), just prints and returns `None`
- No return value indicating how many records were actually inserted
- Called at `crawl4ai_mcp.py:453` with no error check

### 3. `add_code_examples_to_supabase()` — `src/utils.py`, lines ~488-595

- Same pattern as `add_documents_to_supabase`
- Delete failures silently continue (lines ~515-518)
- Insert failures caught and never propagated (lines ~573-593)
- Called at `crawl4ai_mcp.py:493-500` with no error check

## The Call Chain in `crawl_single_page`

```python
# Line ~450 — no error check
update_source_info(supabase_client, source_id, ...)

# Line ~453 — no error check
add_documents_to_supabase(supabase_client, urls, ...)

# Lines ~493-500 — no error check
add_code_examples_to_supabase(supabase_client, ...)

# Lines ~502-514 — always reports success
return json.dumps({"success": True, "chunks_stored": len(chunks), ...})
```

## Impact

When Supabase is unreachable (wrong URL, bad credentials, network down, missing tables):

- All inserts fail silently
- Tool reports `"success": true` and `"chunks_stored": 9`
- No data reaches Supabase
- No way for the caller (or user) to detect the failure
- Misleading success metrics prevent debugging

## Suggested Fixes

1. **Storage functions should return actual insert counts** (e.g., `add_documents_to_supabase` returns the number of successfully inserted records)
2. **`crawl_single_page` should use actual counts** in its response instead of in-memory list lengths
3. **Set `"success": false`** if zero records were stored when records were expected
4. **Optionally propagate exceptions** or include error messages in the response JSON
