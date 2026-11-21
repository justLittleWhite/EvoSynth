"""冻结知识世界与确定性查询索引。

本模块不认识任何领域或工具名称。它从已经冻结的 ``FrozenObservation``
中提取文档、标识符和文本，提供两种只读模式：精确观测重放，以及在同一
冻结语料上进行确定性词项检索。索引模式仍然不能访问 8822，也不会临时
生成网页；未命中的查询明确失败。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import re
from typing import Any, Mapping, Sequence

from evosynth.task_gen.common.json_utils import canonical_json
from evosynth.task_gen.world.controlled import FrozenObservation


_URL_KEYS = {"url", "uri", "link", "web_url"}
_ID_KEYS = {"id", "pmid", "pmcid", "doi", "identifier", "record_id"}
_TITLE_KEYS = {"title", "name", "headline"}
_TEXT_KEYS = {"text", "body", "content", "snippet", "summary", "abstract",
              "description", "indications_and_usage"}
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def _tokens(value: Any) -> set[str]:
    return {item.casefold() for item in _TOKEN_RE.findall(str(value or ""))
            if len(item) > 1}


def _walk(value: Any, path: str = ""):
    if isinstance(value, Mapping):
        yield path, value
        for key, child in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield from _walk(child, path + "/" + escaped)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, path + f"/{index}")


def _first(mapping: Mapping[str, Any], keys: set[str]) -> Any:
    for key, value in mapping.items():
        if str(key).casefold() in keys and value not in (None, "", [], {}):
            return value
    return None


@dataclass(frozen=True)
class FrozenDocument:
    doc_id: str
    url: str | None
    title: str
    text: str
    source_digest: str
    source_path: str
    raw: Any

    @property
    def tokens(self) -> set[str]:
        return _tokens(" ".join((self.doc_id, self.url or "", self.title, self.text)))


@dataclass
class FrozenEpisodeState:
    """所有状态都来自本次 episode，不写回共享世界。"""

    visited: list[str]
    queries: list[str]
    pages: dict[str, int]

    def snapshot(self) -> dict[str, Any]:
        return {"visited": list(self.visited), "queries": list(self.queries),
                "pages": dict(self.pages)}


class FrozenKnowledgeIndex:
    """从冻结观测构建的领域无关、只读、可重放索引。"""

    def __init__(self, rows: Mapping[str, FrozenObservation], tools: Sequence[Any] = ()):
        self.rows = dict(rows)
        self.tools = {str(getattr(tool, "tool_id", "")): tool for tool in tools}
        self.documents: dict[str, FrozenDocument] = {}
        self._by_url: dict[str, FrozenDocument] = {}
        self._by_id: dict[str, FrozenDocument] = {}
        self._search_rows: list[FrozenObservation] = []
        self._detail_rows: list[FrozenObservation] = []
        self._build()

    def _build(self) -> None:
        for row in self.rows.values():
            tool = self.tools.get(f"{row.server}/{row.tool}")
            kind = str(getattr(tool, "kind", ""))
            if kind == "search":
                self._search_rows.append(row)
            elif kind == "detail":
                self._detail_rows.append(row)
            for path, mapping in _walk(row.observation):
                url = _first(mapping, _URL_KEYS)
                identifier = _first(mapping, _ID_KEYS)
                title = _first(mapping, _TITLE_KEYS)
                text = _first(mapping, _TEXT_KEYS)
                if url is None and identifier is None:
                    continue
                doc_id = str(identifier or url)
                document = FrozenDocument(
                    doc_id=doc_id, url=str(url) if url is not None else None,
                    title=str(title or ""), text=str(text or ""),
                    source_digest=row.query_digest, source_path=path, raw=mapping)
                # 首次出现者是冻结语料的 canonical 版本，避免不同观测覆盖。
                self.documents.setdefault(doc_id, document)
                if document.url:
                    self._by_url.setdefault(document.url, document)
                if identifier is not None:
                    self._by_id.setdefault(str(identifier), document)

    def template(self, server: str, tool: str) -> FrozenObservation | None:
        """返回同一工具的冻结观测模板，用于索引模式的未命中参数查询。"""
        return next((row for row in self.rows.values()
                     if row.server == server and row.tool == tool), None)

    @staticmethod
    def _digest(row: FrozenObservation, arguments: Mapping[str, Any]) -> str:
        return sha256(canonical_json({
            "server": row.server, "tool": row.tool, "arguments": dict(arguments)
        }).encode()).hexdigest()

    def _state_after(self, state: FrozenEpisodeState, tool: Any,
                     arguments: Mapping[str, Any], rows: Sequence[FrozenDocument]) -> None:
        query = next((str(value) for key, value in arguments.items()
                      if "query" in str(key).casefold()), None)
        if query:
            state.queries.append(query)
            page = next((int(value) for key, value in arguments.items()
                         if str(key).casefold() in {"page", "page_number"}
                         and str(value).isdigit()), 1)
            state.pages[query] = page
        for document in rows:
            if document.url and document.url not in state.visited:
                state.visited.append(document.url)

    def _row_with_observation(self, row: FrozenObservation, arguments: Mapping[str, Any],
                              observation: Any) -> FrozenObservation:
        encoded = canonical_json(observation)
        return replace(row, arguments=dict(arguments), observation=observation,
                       query_digest=self._digest(row, arguments),
                       observation_hash=sha256(encoded.encode()).hexdigest())

    def _search(self, row: FrozenObservation, arguments: Mapping[str, Any],
                state: FrozenEpisodeState) -> FrozenObservation | None:
        query = next((str(value) for key, value in arguments.items()
                      if "query" in str(key).casefold()), "")
        query_tokens = _tokens(query)
        if not query_tokens or not self.documents:
            return None
        ranked = sorted(
            ((len(query_tokens & document.tokens), document)
             for document in self.documents.values()),
            key=lambda item: (-item[0], item[1].doc_id))
        selected = [document for score, document in ranked if score > 0]
        if not selected:
            return None
        selected_ids = {document.doc_id for document in selected}

        def filter_value(value: Any) -> Any:
            if isinstance(value, list):
                filtered = []
                for item in value:
                    if isinstance(item, Mapping):
                        identifier = _first(item, _ID_KEYS)
                        url = _first(item, _URL_KEYS)
                        if (str(identifier) in selected_ids or str(url) in
                                {doc.url for doc in selected if doc.url}):
                            filtered.append(item)
                    else:
                        filtered.append(item)
                return filtered
            if isinstance(value, Mapping):
                return {key: filter_value(item) for key, item in value.items()}
            return value

        observation = filter_value(row.observation)
        self._state_after(state, self.tools.get(f"{row.server}/{row.tool}"), arguments,
                          selected)
        return self._row_with_observation(row, arguments, observation)

    def _detail(self, row: FrozenObservation, arguments: Mapping[str, Any],
                state: FrozenEpisodeState) -> FrozenObservation | None:
        values = {str(value) for value in arguments.values() if value not in (None, "")}
        matches = [document for document in self.documents.values()
                   if (document.url and document.url in values) or document.doc_id in values]
        if not matches:
            return None
        chosen = matches[0]
        for candidate in self._detail_rows:
            if any(str(value) == chosen.url or str(value) == chosen.doc_id
                   for value in candidate.arguments.values()):
                self._state_after(state, self.tools.get(f"{row.server}/{row.tool}"),
                                  arguments, [chosen])
                return self._row_with_observation(candidate, arguments, candidate.observation)
        return None

    def resolve(self, row: FrozenObservation, arguments: Mapping[str, Any],
                state: FrozenEpisodeState) -> FrozenObservation | None:
        digest = self._digest(row, arguments)
        exact = self.rows.get(digest)
        if exact is not None:
            self._state_after(state, self.tools.get(f"{row.server}/{row.tool}"),
                              arguments, [])
            return exact
        tool = self.tools.get(f"{row.server}/{row.tool}")
        kind = str(getattr(tool, "kind", ""))
        if kind == "search":
            return self._search(row, arguments, state)
        if kind == "detail":
            return self._detail(row, arguments, state)
        return None

    def state(self, state: FrozenEpisodeState | None = None) -> dict[str, Any]:
        return (state or FrozenEpisodeState([], [], {})).snapshot()


__all__ = ["FrozenDocument", "FrozenEpisodeState", "FrozenKnowledgeIndex"]
