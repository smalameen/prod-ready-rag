-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table: stores chunked documents with embeddings
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    embedding   VECTOR(384) NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for cosine similarity search
CREATE INDEX IF NOT EXISTS idx_documents_embedding
    ON documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Metadata index for filtering
CREATE INDEX IF NOT EXISTS idx_documents_metadata
    ON documents USING GIN (metadata);

-- Index for session_id lookups (common filter)
CREATE INDEX IF NOT EXISTS idx_documents_session_id
    ON documents ((metadata->>'session_id'));

-- Index for source_file lookups
CREATE INDEX IF NOT EXISTS idx_documents_source_file
    ON documents ((metadata->>'source_file'));

-- Match documents by cosine similarity
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(384),
    match_threshold FLOAT,
    match_count INT,
    filter_where JSONB DEFAULT NULL
)
RETURNS TABLE (
    id TEXT,
    text TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    IF filter_where IS NULL THEN
        RETURN QUERY
        SELECT d.id, d.text, d.metadata, 1 - (d.embedding <=> query_embedding) AS similarity
        FROM documents d
        WHERE 1 - (d.embedding <=> query_embedding) > match_threshold
        ORDER BY d.embedding <=> query_embedding
        LIMIT match_count;
        RETURN;
    END IF;

    -- Handle $or syntax: {"$or": [{"session_id": {"$eq": "val1"}}, {"session_id": {"$eq": "val2"}}]}
    IF filter_where ? '$or' THEN
        RETURN QUERY
        SELECT d.id, d.text, d.metadata, 1 - (d.embedding <=> query_embedding) AS similarity
        FROM documents d
        WHERE
            EXISTS (
                SELECT 1
                FROM jsonb_array_elements(filter_where->'$or') AS cond,
                     jsonb_each(cond) AS j(k, v)
                WHERE d.metadata @>
                    jsonb_build_object(j.k, CASE WHEN j.v ? '$eq' THEN j.v->'$eq' ELSE j.v END)
            )
            AND 1 - (d.embedding <=> query_embedding) > match_threshold
        ORDER BY d.embedding <=> query_embedding
        LIMIT match_count;
        RETURN;
    END IF;

    -- Standard filter: metadata must contain all key-value pairs
    RETURN QUERY
    SELECT d.id, d.text, d.metadata, 1 - (d.embedding <=> query_embedding) AS similarity
    FROM documents d
    WHERE
        d.metadata @> filter_where
        AND 1 - (d.embedding <=> query_embedding) > match_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Delete documents matching a metadata filter
CREATE OR REPLACE FUNCTION delete_documents_by_metadata(filter_where JSONB)
RETURNS INT
LANGUAGE plpgsql
AS $$
DECLARE
    deleted_count INT;
BEGIN
    DELETE FROM documents
    WHERE metadata @> filter_where;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$;

-- Insert or update a document
CREATE OR REPLACE FUNCTION upsert_document(
    doc_id TEXT,
    doc_text TEXT,
    doc_embedding VECTOR(384),
    doc_metadata JSONB
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO documents (id, text, embedding, metadata)
    VALUES (doc_id, doc_text, doc_embedding, doc_metadata)
    ON CONFLICT (id)
    DO UPDATE SET
        text = doc_text,
        embedding = doc_embedding,
        metadata = doc_metadata;
END;
$$;
