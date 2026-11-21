"""可机械重算的跨领域逻辑图。

节点只操作已经审计的 claim 值，不包含领域、工具或调用顺序规则。有限操作
代数是任务本体的一部分；任何未声明操作、悬空输入或类型不匹配都失效关闭。
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import re
from typing import Any, Mapping, Sequence


SUBSTANTIVE_OPERATIONS = frozenset({
    "compare", "aggregate", "order", "filter", "rank", "derive",
    "intersect", "join", "branch", "scan", "synthesize",
    "extract",
})


_NUMBER_TOKEN = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])")
_ISO_DATE_TOKEN = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ][0-9:.]+(?:Z|[+-]\d{2}:?\d{2})?)?")


def _number(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise TypeError("numeric logic operation received a non-number")
    return Decimal(str(value))


def _json_number(value: Decimal) -> int | float:
    return int(value) if value == value.to_integral_value() else float(value)


def _date_value(value: Any) -> datetime:
    """Parse a complete ISO date/time without guessing locale-specific dates."""
    if not isinstance(value, str) or not _ISO_DATE_TOKEN.fullmatch(value.strip()):
        raise TypeError("ordered date operation received a non-ISO date")
    text = value.strip().replace(" ", "T", 1)
    try:
        if "T" not in text:
            parsed = datetime.combine(date.fromisoformat(text), datetime.min.time())
        else:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TypeError("ordered date operation received an invalid ISO date") from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _ordered(value: Any) -> tuple[str, Decimal | datetime]:
    """Return a typed ordering key for numbers or explicit ISO dates."""
    if not isinstance(value, bool) and isinstance(value, (int, float, Decimal)):
        return "number", Decimal(str(value))
    if isinstance(value, str) and _ISO_DATE_TOKEN.fullmatch(value.strip()):
        return "date", _date_value(value)
    # Preserve the established numeric type error for arbitrary strings,
    # containers and booleans.  Only complete ISO values opt into date order.
    return "number", _number(value)


def _ordered_values(values: Sequence[Any]) -> tuple[str, list[Decimal | datetime]]:
    converted = [_ordered(value) for value in values]
    kinds = {kind for kind, _value in converted}
    if len(kinds) != 1:
        raise TypeError("ordered logic operation received incompatible types")
    return converted[0][0], [value for _kind, value in converted]


def _type_compatible(left: Any, right: Any) -> bool:
    """Whether two values share a meaningful JSON comparison type."""
    numeric = (int, float, Decimal)
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right)
    if isinstance(left, numeric) and isinstance(right, numeric):
        return True
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return True
    if isinstance(left, (list, tuple, set)) and isinstance(
            right, (list, tuple, set)):
        return True
    return type(left) is type(right)


def _diagnostic_value(value: Any) -> str:
    """Return a bounded, domain-neutral type/value summary for DAG repair."""
    if isinstance(value, bool):
        kind = "boolean"
    elif isinstance(value, str):
        kind = "string"
    elif isinstance(value, int):
        kind = "integer"
    elif isinstance(value, (float, Decimal)):
        kind = "number"
    elif isinstance(value, Mapping):
        kind = "object"
    elif isinstance(value, (list, tuple, set)):
        kind = "array"
    elif value is None:
        kind = "null"
    else:
        kind = type(value).__name__
    rendered = repr(value)
    if len(rendered) > 160:
        rendered = rendered[:157] + "..."
    return f"{kind}={rendered}"


def _validate_output_value(value: Any) -> Any:
    """Apply domain-neutral answer-output invariants.

    A logic node is part of an auditable answer graph, so an empty value or a
    collection containing duplicate entries is never a useful certificate.
    Enforcing this at graph evaluation keeps the rule independent of map,
    finance, or any other domain recipe.
    """
    if value is None:
        raise ValueError("logic output must be non-empty")
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("logic output must be non-empty")
        return value
    if isinstance(value, Mapping):
        if not value:
            raise ValueError("logic output mapping must be non-empty")
        return value
    if isinstance(value, (list, tuple, set)):
        if not value:
            raise ValueError("logic output collection must be non-empty")
        items = list(value)
        if any(left == right for index, left in enumerate(items)
               for right in items[index + 1:]):
            raise ValueError("logic output collection must contain unique items")
        return value
    return value


def _labels(parameters: Mapping[str, Any], count: int) -> list[str]:
    labels = parameters.get("labels")
    if not isinstance(labels, list) or len(labels) != count:
        raise ValueError("logic node labels must align with inputs")
    normalized = [str(item) for item in labels]
    if len(set(normalized)) != len(normalized):
        raise ValueError("logic node labels must be unique")
    return normalized


def _predicate(value: Any, operator: str, target: Any) -> bool:
    if operator in {"eq", "ne"}:
        # JSON permits comparing arbitrary types, but cross-type equality is
        # a constant false/true predicate and therefore cannot express a
        # grounded reasoning step.  Reject it instead of letting a generator
        # manufacture a non-empty answer from labels alone.
        if not _type_compatible(value, target):
            raise TypeError("equality logic operation received incompatible types")
        return value == target if operator == "eq" else value != target
    if operator == "contains":
        if isinstance(value, str) and not isinstance(target, str):
            raise TypeError("string contains requires a string target")
        if not isinstance(value, (str, Mapping, list, tuple, set)):
            raise TypeError("contains logic operation received a non-container")
        return target in value
    _kind, (left, right) = _ordered_values((value, target))
    return {"gt": left > right, "gte": left >= right,
            "lt": left < right, "lte": left <= right}[operator]


def _normalize_operator(value: Any) -> str:
    """Normalize common symbolic and textual comparison spellings."""
    operator = str(value or "").strip().casefold()
    return {
        "=": "eq", "==": "eq", "equals": "eq",
        "!=": "ne", "<>": "ne", "not_equals": "ne",
        ">": "gt", "greater_than": "gt",
        ">=": "gte", "ge": "gte", "greater_equal": "gte",
        "greater_than_or_equal": "gte",
        "<": "lt", "less_than": "lt",
        "<=": "lte", "le": "lte", "less_equal": "lte",
        "less_than_or_equal": "lte",
        "if": "eq",
    }.get(operator, operator)


def evaluate_operation(operation: str, inputs: Sequence[Any],
                       parameters: Mapping[str, Any]) -> Any:
    """执行一个类型受限的通用逻辑节点。"""
    op = str(operation).casefold()
    mode = str(parameters.get("mode", "")).casefold()
    if op == "compare":
        if len(inputs) < 2:
            raise ValueError("compare requires at least two inputs")
        if mode in {"argmax", "argmin"}:
            labels = _labels(parameters, len(inputs))
            _kind, values = _ordered_values(inputs)
            index = (max if mode == "argmax" else min)(range(len(values)), key=values.__getitem__)
            return labels[index]
        if mode == "difference" and len(inputs) == 2:
            return _json_number(_number(inputs[0]) - _number(inputs[1]))
        if mode == "relation" and len(inputs) == 2:
            _kind, (left, right) = _ordered_values(inputs)
            return "greater" if left > right else "less" if left < right else "equal"
        if mode == "equality" and len(inputs) == 2:
            if not _type_compatible(inputs[0], inputs[1]):
                raise TypeError("compare equality received incompatible types")
            return inputs[0] == inputs[1]
        raise ValueError("unsupported compare mode")
    if op == "aggregate":
        if not inputs:
            raise ValueError("aggregate requires inputs")
        if mode == "count":
            return len(inputs)
        values = [_number(item) for item in inputs]
        if mode == "sum":
            result = sum(values, Decimal(0))
        elif mode == "mean":
            result = sum(values, Decimal(0)) / len(values)
        elif mode == "min":
            result = min(values)
        elif mode == "max":
            result = max(values)
        else:
            raise ValueError("unsupported aggregate mode")
        return _json_number(result)
    if op in {"rank", "order"}:
        if len(inputs) < 2:
            raise ValueError(f"{op} requires at least two inputs")
        labels = _labels(parameters, len(inputs))
        direction = str(parameters.get("direction", "descending")).casefold()
        if direction in {"descending", "desc", "decreasing", "highest_first"}:
            descending = True
        elif direction in {"ascending", "asc", "increasing", "lowest_first"}:
            descending = False
        else:
            raise ValueError("unsupported rank/order direction")
        _kind, values = _ordered_values(inputs)
        indices = sorted(range(len(values)), key=lambda index: values[index], reverse=descending)
        top_k = int(parameters.get("top_k", len(indices)))
        if top_k < 1 or top_k > len(indices):
            raise ValueError("invalid top_k")
        return [labels[index] for index in indices[:top_k]]
    if op == "filter":
        if len(inputs) < 2:
            raise ValueError("filter requires at least two candidates")
        labels = _labels(parameters, len(inputs))
        # ``mode`` 是其他算子的通用参数名，结构化生成器偶尔会
        # 把筛选／分支的比较符放在该键中。这里只做语法别名归一，
        # 后续允许集合、阈值、类型和输出门禁保持不变。
        operator = _normalize_operator(
            parameters.get("operator", parameters.get("mode")))
        if operator not in {"eq", "ne", "contains", "gt", "gte", "lt", "lte"}:
            raise ValueError("unsupported filter operator")
        if "threshold" not in parameters:
            raise ValueError("filter threshold is required")
        target = parameters.get("threshold")
        raw_excluded = parameters.get("exclude_labels", [])
        if not isinstance(raw_excluded, list):
            raise TypeError("filter exclude_labels must be a list")
        excluded = {str(item) for item in raw_excluded}
        unknown = excluded - set(labels)
        if unknown:
            raise ValueError("filter exclude_labels must be drawn from labels")
        return [label for label, value in zip(labels, inputs)
                if label not in excluded and _predicate(value, operator, target)]
    if op == "derive":
        values = [_number(item) for item in inputs]
        if mode == "add" and len(values) >= 2:
            result = sum(values, Decimal(0))
        elif mode == "subtract" and len(values) == 2:
            result = values[0] - values[1]
        elif mode == "multiply" and len(values) >= 2:
            result = Decimal(1)
            for value in values:
                result *= value
        elif mode == "divide" and len(values) == 2:
            result = values[0] / values[1]
        elif mode == "percent_change" and len(values) == 2:
            result = (values[1] - values[0]) / values[0] * Decimal(100)
        else:
            raise ValueError("unsupported derive mode")
        digits = int(parameters.get("digits", 6))
        return round(float(result), digits)
    if op == "extract":
        if len(inputs) != 1 or not isinstance(inputs[0], str):
            raise TypeError("extract requires exactly one string input")
        source = inputs[0]
        exact_quote = parameters.get("exact_quote")
        if not isinstance(exact_quote, str) or not exact_quote.strip():
            raise ValueError("extract exact_quote is required")
        if exact_quote not in source:
            raise ValueError("extract exact_quote is not present in the source text")
        unit = parameters.get("unit")
        if unit is not None:
            if not isinstance(unit, str) or not unit.strip():
                raise ValueError("extract unit must be a non-empty string")
            if unit.casefold() not in exact_quote.casefold():
                raise ValueError("extract unit is not present in exact_quote")
        matches = [Decimal(item.group(0).replace(",", ""))
                   for item in _NUMBER_TOKEN.finditer(exact_quote)]
        if not matches:
            raise ValueError("extract found no numeric token")
        if mode == "first_number":
            result = matches[0]
        elif mode == "last_number":
            result = matches[-1]
        elif mode == "min_number":
            result = min(matches)
        elif mode == "max_number":
            result = max(matches)
        elif mode == "number_at":
            index = int(parameters.get("index", -1))
            if index < 0 or index >= len(matches):
                raise ValueError("extract number_at index is out of range")
            result = matches[index]
        else:
            raise ValueError("unsupported extract mode")
        return _json_number(result)
    if op == "intersect":
        if len(inputs) < 2 or not all(isinstance(item, (list, tuple, set)) for item in inputs):
            raise TypeError("intersect requires at least two collections")
        common = set(inputs[0])
        for item in inputs[1:]:
            common &= set(item)
        return sorted(common, key=repr)
    if op in {"join", "synthesize", "scan"}:
        if len(inputs) < 2:
            raise ValueError(f"{op} requires at least two inputs")
        if op == "synthesize":
            if mode == "consistency":
                # 一致性判断只能比较同类观测。把筛选结果列表与路线文本、
                # 数值与名称等异构事实放进同一节点只会产生没有语义的 False，
                # 不能作为“验证失败”或其它任务结论。数值允许整数/浮点混用，
                # 其余值必须具有相同 JSON 顶层类型。
                if not all(_type_compatible(inputs[0], value)
                           for value in inputs[1:]):
                    raise TypeError(
                        "synthesize consistency requires type-compatible inputs")
                return all(value == inputs[0] for value in inputs[1:])
            if mode != "bundle":
                raise ValueError("synthesize mode must be consistency or bundle")
        labels = parameters.get("labels")
        if isinstance(labels, list):
            labels = _labels(parameters, len(inputs))
            return {label: value for label, value in zip(labels, inputs)}
        return list(inputs)
    if op == "branch":
        if len(inputs) < 3:
            raise ValueError("branch requires condition, true value and false value")
        operator = _normalize_operator(
            parameters.get("operator", parameters.get("mode")))
        if operator not in {"eq", "ne", "contains", "gt", "gte", "lt", "lte"}:
            raise ValueError("unsupported branch operator")
        if "threshold" not in parameters:
            raise ValueError("branch threshold is required")
        return inputs[1] if _predicate(inputs[0], operator, parameters.get("threshold")) else inputs[2]
    raise ValueError(f"unsupported logic operation:{op}")


def evaluate_logic_graph(nodes: Sequence[Mapping[str, Any]],
                         claim_values: Mapping[str, Any],
                         required_operations: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按显式依赖拓扑重算逻辑图，并验证目标操作覆盖。"""
    values = dict(claim_values)
    normalized: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    for index, raw in enumerate(nodes, 1):
        if not isinstance(raw, Mapping):
            raise TypeError("logic node must be an object")
        node_id = str(raw.get("node_id") or f"logic_{index}")
        operation = str(raw.get("operation") or "").casefold()
        if not node_id or node_id in values or node_id in seen_nodes:
            raise ValueError(f"invalid logic node id:{node_id}")
        if operation not in SUBSTANTIVE_OPERATIONS:
            raise ValueError(f"unsupported logic operation:{operation}")
        inputs = [str(item) for item in raw.get("inputs", ())]
        minimum_inputs = 3 if operation == "branch" else 1 if operation == "extract" else 2
        if len(inputs) < minimum_inputs:
            raise ValueError(
                f"logic node has insufficient inputs:{node_id}:"
                f"{len(inputs)}<{minimum_inputs}")
        if operation == "branch" and len(inputs) != 3:
            raise ValueError(
                f"branch requires exactly three inputs:{node_id}:{len(inputs)}")
        if operation == "extract" and len(inputs) != 1:
            raise ValueError(
                f"extract requires exactly one input:{node_id}:{len(inputs)}")
        if any(item not in values for item in inputs):
            missing = sorted({item for item in inputs if item not in values})
            raise ValueError(
                f"logic node references missing inputs:{node_id}:" + ",".join(missing))
        # A branch may legitimately return the same claim it inspected as its
        # true or false value.  Other operators need distinct evidence inputs;
        # repeated IDs would only inflate apparent complexity.
        if operation != "branch" and len(set(inputs)) != len(inputs):
            raise ValueError(f"logic node repeats an input:{node_id}")
        parameters = dict(raw.get("parameters") or {})
        input_values = [values[item] for item in inputs]
        try:
            output = evaluate_operation(operation, input_values, parameters)
            output = _validate_output_value(output)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            # 返修 agent 需要知道首个非法节点究竟收到了什么类型。只返回
            # 已审计输入的有界摘要和已有参数，不补值、不猜领域语义，也不
            # 改变编译结果；异常类型保持不变，门禁继续失效关闭。
            raw_labels = parameters.get("labels")
            diagnostic_labels = (
                [str(item) for item in raw_labels]
                if isinstance(raw_labels, list) and len(raw_labels) == len(inputs)
                else [""] * len(inputs))
            diagnostic = ", ".join(
                f"{input_id}" + (f"(label={label})" if label else "")
                + f":{_diagnostic_value(value)}"
                for input_id, label, value
                in zip(inputs, diagnostic_labels, input_values))
            parameters_text = repr(parameters)
            if len(parameters_text) > 500:
                parameters_text = parameters_text[:497] + "..."
            raise type(exc)(
                f"logic node {node_id} operation={operation} failed; "
                f"inputs=[{diagnostic}]; parameters={parameters_text}; {exc}"
            ) from exc
        values[node_id] = output
        seen_nodes.add(node_id)
        normalized.append({"node_id": node_id, "operation": operation,
                           "inputs": inputs, "parameters": parameters,
                           "output": output})
    required = {str(item).casefold() for item in required_operations} & SUBSTANTIVE_OPERATIONS
    covered = {item["operation"] for item in normalized}
    missing = sorted(required - covered)
    if missing:
        raise ValueError("logic graph missing operations:" + ",".join(missing))
    return normalized, {node_id: values[node_id] for node_id in seen_nodes}
