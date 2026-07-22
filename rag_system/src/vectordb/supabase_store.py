import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from src.loaders.base import Document

try:
    from supabase import create_client
except ImportError:
    create_client = None


logger = logging.getLogger(__name__)


_MIGRATION_SQL_PATH = Path(__file__).parent.parent.parent / "supabase" / "migration.sql"


class SupabaseVectorStore:
    def __init__(self, collection_name: str = "user_docs_en"):
        self.collection_name = collection_name
        supabase_url = os.environ["SUPABASE_URL"]
        supabase_key = os.environ["SUPABASE_SERVICE_KEY"]
        self._sb = create_client(supabase_url, supabase_key)
        self._dimension = 384
        self._table = "documents"
        self._ensure_migration(supabase_url, supabase_key)
        logger.info(f"SupabaseVectorStore initialized: {self._table}")

    def _ensure_migration(self, supabase_url: str, supabase_key: str):
        if not _MIGRATION_SQL_PATH.exists():
            logger.warning(f"Migration SQL not found at {_MIGRATION_SQL_PATH}")
            return
        try:
            self._sb.rpc("match_documents", {
                "query_embedding": [0.0] * self._dimension,
                "match_threshold": 0.0,
                "match_count": 1,
                "filter_where": None,
            }).execute()
            logger.info("match_documents function already exists")
            return
        except Exception:
            logger.info("match_documents function not found, running migration...")
        try:
            import psycopg2
            db_password = os.environ.get("SUPABASE_DB_PASSWORD")
            if not db_password:
                logger.warning("SUPABASE_DB_PASSWORD not set, skipping auto-migration. "
                               "Run rag_system/supabase/migration.sql manually in Supabase SQL Editor.")
                return
            m = re.match(r"https://([^.]+)\.supabase\.co", supabase_url)
            if not m:
                logger.warning("Could not parse project ref from SUPABASE_URL")
                return
            project_ref = m.group(1)
            conn = psycopg2.connect(
                host=f"db.{project_ref}.supabase.co",
                port=5432,
                user="postgres",
                password=db_password,
                dbname="postgres",
                connect_timeout=10,
                sslmode="require",
            )
            conn.autocommit = True
            sql = _MIGRATION_SQL_PATH.read_text(encoding="utf-8")
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.close()
            logger.info("Migration completed successfully")
        except Exception as e:
            logger.warning(f"Auto-migration failed: {e}. "
                           f"Run rag_system/supabase/migration.sql manually in Supabase SQL Editor.")

    @property
    def collection(self):
        return self

    def get(self, where: dict[str, Any] | None = None, include: list[str] | None = None):
        q = self._sb.table(self._table).select("*")
        eq_filters, or_groups = self._build_filters(where or {})
        for f in eq_filters:
            q = q.filter(f["column"], f["operator"], f["value"])
        for or_str in or_groups:
            q = q.or_(or_str)
        resp = q.execute()
        rows = resp.data if resp.data else []
        result: dict[str, Any] = {"ids": [], "metadatas": [], "documents": []}
        for r in rows:
            result["ids"].append(r["id"])
            result["documents"].append(r["text"])
            result["metadatas"].append(r.get("metadata", {}))
        if include:
            result = {k: v for k, v in result.items() if k in include or k == "ids"}
        return result

    def count(self) -> int:
        resp = self._sb.table(self._table).select("id", count="exact").execute()
        return resp.count or 0

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> list[str]:
        ids: list[str] = []
        rows: list[dict] = []
        for doc, emb in zip(documents, embeddings):
            chunk_id = doc.metadata.get("chunk_id", "")
            ids.append(chunk_id)
            meta = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                    for k, v in doc.metadata.items()}
            rows.append({
                "id": chunk_id,
                "text": doc.text.replace("\x00", ""),
                "embedding": emb,
                "metadata": meta,
            })
        if rows:
            self._sb.table(self._table).upsert(rows, ignore_duplicates=False).execute()
        logger.info(f"Added {len(ids)} documents to Supabase vector store")
        return ids

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        similarity_threshold: float = 0.75,
        metadata_filter: dict[str, Any] | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        filter_where = where or metadata_filter
        if filter_where is None:
            filter_json = None
        elif "$or" in filter_where:
            all_hits: dict[str, dict[str, Any]] = {}
            for sub in filter_where["$or"]:
                flat = self._flatten_filter(sub)
                resp = self._sb.rpc(
                    "match_documents",
                    {
                        "query_embedding": query_embedding,
                        "match_threshold": similarity_threshold,
                        "match_count": top_k,
                        "filter_where": json.dumps(flat) if flat else None,
                    },
                ).execute()
                for row in resp.data or []:
                    cid = row["id"]
                    if cid not in all_hits:
                        all_hits[cid] = {
                            "id": cid,
                            "text": row["text"],
                            "metadata": row.get("metadata", {}),
                            "score": row["similarity"],
                        }
            hits = sorted(all_hits.values(), key=lambda x: x["score"], reverse=True)[:top_k]
            return hits
        else:
            filter_json = json.dumps(self._flatten_filter(filter_where))

        resp = self._sb.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": similarity_threshold,
                "match_count": top_k,
                "filter_where": filter_json,
            },
        ).execute()

        hits: list[dict[str, Any]] = []
        for row in resp.data or []:
            hits.append({
                "id": row["id"],
                "text": row["text"],
                "metadata": row.get("metadata", {}),
                "score": row["similarity"],
            })
        return hits

    @staticmethod
    def _flatten_filter(filter_: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for k, v in filter_.items():
            if isinstance(v, dict) and "$eq" in v:
                result[k] = v["$eq"]
            elif isinstance(v, dict) and "$ne" in v:
                continue
            else:
                result[k] = v
        return result

    def delete_documents(self, ids: list[str]):
        if not ids:
            return
        for doc_id in ids:
            self._sb.table(self._table).delete().eq("id", doc_id).execute()
        logger.info(f"Deleted {len(ids)} documents")

    def delete_by_metadata(self, where: dict[str, Any]) -> int:
        eq_filters, or_groups = self._build_filters(where)
        q = self._sb.table(self._table).delete()
        for f in eq_filters:
            q = q.filter(f["column"], f["operator"], f["value"])
        for or_str in or_groups:
            q = q.or_(or_str)
        resp = q.execute()
        count = len(resp.data) if resp.data else 0
        logger.info(f"Deleted {count} documents matching {where}")
        return count

    @staticmethod
    def _build_filters(where: dict[str, Any]) -> tuple[list[dict], list[str]]:
        eq_filters: list[dict] = []
        or_groups: list[str] = []
        for key, condition in where.items():
            if key == "$or":
                parts: list[str] = []
                for sub in condition:
                    for k, v in sub.items():
                        if isinstance(v, dict) and "$eq" in v:
                            parts.append(f"metadata->>{k}.eq.{v['$eq']}")
                if parts:
                    or_groups.append(",".join(parts))
            elif isinstance(condition, dict):
                col = f"metadata->>{key}"
                if "$eq" in condition:
                    eq_filters.append({"column": col, "operator": "eq", "value": condition["$eq"]})
                elif "$ne" in condition:
                    eq_filters.append({"column": col, "operator": "neq", "value": condition["$ne"]})
            else:
                eq_filters.append({"column": f"metadata->>{key}", "operator": "eq", "value": condition})
        return eq_filters, or_groups
