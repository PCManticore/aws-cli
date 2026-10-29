# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""``aws athena run-query`` -- run one Athena statement and return every row.

Callers otherwise hand-roll the Athena lifecycle:
``StartQueryExecution`` -> ``GetQueryExecution`` (poll) -> ``GetQueryResults``
(paginate). This command runs it once, read-only, and returns the full, typed
result set in one call.

Procedure:

0. Validate the input. Exactly one row-returning statement is allowed; writes and
   DDL are refused (this command is strictly read-only). The size limit, S3 URI
   and polling parameters are also checked, all before any AWS call.
1. ``athena:GetWorkGroup`` -- honour the workgroup's enforced output location and
   engine version. When no workgroup is given, auto-resolve one whose effective
   configuration actually has an output location.
2. ``athena:StartQueryExecution``.
3. ``athena:GetQueryExecution`` -- poll with backoff to SUCCEEDED, FAILED or
   CANCELLED, using the shared bounded poller.
4. ``athena:GetQueryResults`` -- paginate on NextToken, drop the header row, and
   return column types, ``DataScannedInBytes`` and an exact resume token if
   ``--max-rows`` truncates the result.
"""

from __future__ import annotations

import re
import time
from typing import Any

from awscli.customizations.commands import BasicCommand
from awscli.customizations.compositereadonly.helpers import (
    CONFLICT,
    CREDENTIALS_EXPIRED,
    INVALID_INPUT,
    NOT_FOUND,
    NOT_READY,
    PERMISSION_DENIED,
    SERVICE_ERROR,
    THROTTLED,
    TIMEOUT,
    call,
    check_max_size,
    check_non_empty_str,
    check_positive_int,
    check_positive_number,
    create_read_client,
    emit_result,
    err_result,
    invalid_input,
    ok_result,
)
from awscli.customizations.compositereadonly.poll import drive_poll
from awscli.customizations.compositereadonly.resolve import (
    resolve_workgroup_with_output,
)

_TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})
_ROW_RETURNING = frozenset(
    {
        "SELECT",
        "WITH",
        "VALUES",
        "SHOW",
        "DESCRIBE",
        "DESC",
        "EXPLAIN",
        "TABLE",
    }
)
_TXN = "Athena does not support transactions (BEGIN/START TRANSACTION/COMMIT/ROLLBACK)."
_NEVER_SUPPORTED: dict[str, str] = {
    "USE": "Athena has no session state; pass --database (and --catalog) instead of 'USE <db>'.",
    "SET": "Athena does not support session properties via SET; remove the SET statement.",
    "RESET": "Athena does not support session properties via RESET; remove the RESET statement.",
    "BEGIN": _TXN,
    "START": _TXN,
    "COMMIT": _TXN,
    "ROLLBACK": _TXN,
    "GRANT": "Athena permissions come from IAM / Lake Formation; GRANT is not supported.",
    "REVOKE": "Athena permissions come from IAM / Lake Formation; REVOKE is not supported.",
}
# DDL/write statement keywords. This command is read-only, so these are always
# refused (there is no write escape hatch).
_WRITES_OR_DDL = frozenset(
    {
        "CREATE",
        "DROP",
        "ALTER",
        "INSERT",
        "DELETE",
        "UPDATE",
        "MERGE",
        "MSCK",
        "UNLOAD",
        "OPTIMIZE",
        "VACUUM",
        "EXECUTE",
        "PREPARE",
        "DEALLOCATE",
    }
)
_INTEGER_TYPES = frozenset({"tinyint", "smallint", "integer", "int", "bigint"})
# Athena StartQueryExecution QueryString limit (256 KB).
_MAX_QUERY_BYTES = 262_144
_MAX_PAGE_SIZE = 1000
_MAX_POLL_INTERVAL = 5.0
_MAX_POLLS = 10_000
_MAX_TRANSIENT_POLL_ERRORS = 5
_NOT_SUPPORTED_MSG = "queries of this type are not supported"

_VERBATIM_GUIDANCE = (
    "Cell values are verbatim Athena strings (timestamps such as "
    "'2024-01-15 00:00:00.000' and ids are untouched); print them exactly as "
    "returned, do not round, abbreviate or relabel them."
)
_NULL_GUIDANCE = "SQL NULL is returned as JSON null; render it explicitly as NULL in any report."


def _scan_statements(sql: str) -> tuple[list[tuple[str, str]], str | None]:
    """Split on top-level ';' respecting quotes/comments -> ([(original, code_only)], unterminated)."""
    segments: list[tuple[str, str]] = []
    orig: list[str] = []
    code: list[str] = []
    unterminated: str | None = None
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch in ("'", '"', "`"):
            j = i + 1
            closed = False
            while j < n:
                if sql[j] == ch:
                    if j + 1 < n and sql[j + 1] == ch:
                        j += 2
                        continue
                    closed = True
                    break
                j += 1
            if not closed:
                unterminated = (
                    f"unterminated {ch} quote starting at character {i}"
                )
            chunk = sql[i : j + 1]
            orig.append(chunk)
            code.append(chunk)
            i = j + 1
            continue
        if sql.startswith("--", i):
            j = sql.find("\n", i)
            j = n if j == -1 else j
            orig.append(sql[i:j])
            code.append(" ")
            i = j
            continue
        if sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            if j == -1:
                unterminated = (
                    f"unterminated /* comment starting at character {i}"
                )
                j = n
            else:
                j += 2
            orig.append(sql[i:j])
            code.append(" ")
            i = j
            continue
        if ch == ";":
            segments.append(("".join(orig), "".join(code)))
            orig, code = [], []
            i += 1
            continue
        orig.append(ch)
        code.append(ch)
        i += 1
    segments.append(("".join(orig), "".join(code)))
    return [
        (o.strip(), c.strip()) for o, c in segments if c.strip()
    ], unterminated


# NOTE: This keyword check is the sole read-only enforcement point for Athena queries. StartQueryExecution itself passes through the call bridge, so this guard must not be weakened without adding an alternative.
def _leading_keyword(code_text: str) -> str:
    m = re.match(r"[\s(]*([A-Za-z_]+)", code_text)
    return m.group(1).upper() if m else ""


def _validate_params(
    workgroup: Any,
    database: Any,
    catalog: Any,
    output_location: Any,
    max_rows: Any,
    poll_interval: Any,
    max_wait: Any,
) -> dict[str, Any] | None:
    # workgroup may be None here (auto-resolved later); only reject an explicit
    # empty string.
    err = check_non_empty_str("workgroup", workgroup, required=False)
    if err is not None:
        return invalid_input(
            err,
            ["Pass --workgroup <name> or omit it to auto-resolve one."],
        )
    err = check_non_empty_str("catalog", catalog, required=True)
    if err is not None:
        return invalid_input(
            err,
            ["Pass --catalog AwsDataCatalog unless the task names another."],
        )
    err = check_non_empty_str("database", database, required=False)
    if err is not None:
        return invalid_input(
            err,
            ["Omit --database or pass the Glue database name."],
        )
    if output_location is not None and (
        not isinstance(output_location, str)
        or not output_location.startswith("s3://")
        or len(output_location) <= 5
    ):
        return invalid_input(
            f"output_location {output_location!r} is not an s3:// URI.",
            [
                "Pass --output-location s3://<bucket>/<prefix>/ or omit it to use "
                "the workgroup's location."
            ],
        )
    err = check_positive_int("max_rows", max_rows)
    if err is not None:
        return invalid_input(
            err,
            ["Pass e.g. --max-rows 10000."],
        )
    for name, val in (
        ("poll_interval", poll_interval),
        ("max_wait", max_wait),
    ):
        err = check_positive_number(name, val)
        if err is not None:
            return invalid_input(
                err,
                [f"Pass a positive {name}."],
            )
    return None


def _check_sql(sql: Any) -> tuple[dict[str, Any] | None, str]:
    """Return (error_or_None, statement_to_submit). Makes no AWS calls. Read-only."""
    if not isinstance(sql, str) or not sql.strip():
        return (
            invalid_input(
                "sql is empty.",
                [
                    "Pass exactly one SQL statement, e.g. 'SELECT ... FROM db.table'."
                ],
            ),
            "",
        )
    size_err = check_max_size("sql", sql, _MAX_QUERY_BYTES)
    if size_err is not None:
        return (
            invalid_input(
                f"{size_err} (Athena's QueryString limit).",
                [
                    "Shorten the statement (e.g. move large IN-lists into a table or a VALUES join)."
                ],
            ),
            "",
        )
    statements, unterminated = _scan_statements(sql)
    if unterminated:
        return (
            invalid_input(
                f"sql has an {unterminated}.",
                ["Close the quote/comment; Athena would reject it."],
            ),
            "",
        )
    if not statements:
        return (
            invalid_input(
                "sql contains only comments/whitespace.",
                ["Pass a real SQL statement."],
            ),
            "",
        )
    if len(statements) > 1:
        kinds = ", ".join(_leading_keyword(c) or "?" for _, c in statements)
        return (
            invalid_input(
                f"sql contains {len(statements)} statements ({kinds}); Athena runs exactly one "
                "statement per StartQueryExecution and rejects batches with 'Queries of this type "
                "are not supported'.",
                [
                    "Run run-query once per statement (they can be issued in parallel).",
                    "To combine result sets, use UNION ALL or a single WITH ... SELECT.",
                ],
            ),
            "",
        )
    original, code_text = statements[0]
    keyword = _leading_keyword(code_text)
    if keyword in _NEVER_SUPPORTED:
        return (
            invalid_input(
                f"'{keyword}' statements are not supported by Athena (it would answer 'Queries of "
                f"this type are not supported'). {_NEVER_SUPPORTED[keyword]}",
                [_NEVER_SUPPORTED[keyword]],
            ),
            "",
        )
    if keyword in _WRITES_OR_DDL:
        # Read-only command: DDL/write statements are always refused.
        return (
            invalid_input(
                f"'{keyword}' is a DDL/write statement; run-query is read-only and refuses writes.",
                [
                    "Rewrite as SELECT/WITH/SHOW/DESCRIBE/EXPLAIN to read data.",
                ],
            ),
            "",
        )
    if keyword not in _ROW_RETURNING:
        return (
            invalid_input(
                f"Unrecognized leading keyword {keyword!r}; Athena would likely reject this statement.",
                [
                    "Start the statement with SELECT, WITH, VALUES, SHOW, DESCRIBE or EXPLAIN (check for typos)."
                ],
            ),
            "",
        )
    return None, original


def _read_workgroup(
    call_op, workgroup: str
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    info: dict[str, Any] = {
        "read": False,
        "output_location": None,
        "enforced": False,
        "managed_results": False,
        "engine_version": None,
        "state": None,
        "note": None,
    }
    res = call_op("GetWorkGroup", {"WorkGroup": workgroup})
    if res["ok"]:
        wg = (res.get("value") or {}).get("WorkGroup") or {}
        cfg = wg.get("Configuration") or {}
        info["read"] = True
        info["state"] = wg.get("State")
        info["output_location"] = (cfg.get("ResultConfiguration") or {}).get(
            "OutputLocation"
        )
        info["enforced"] = bool(cfg.get("EnforceWorkGroupConfiguration"))
        info["managed_results"] = bool(
            (cfg.get("ManagedQueryResultsConfiguration") or {}).get("Enabled")
        )
        ev = cfg.get("EngineVersion") or {}
        info["engine_version"] = ev.get("EffectiveEngineVersion") or ev.get(
            "SelectedEngineVersion"
        )
        if info["state"] == "DISABLED":
            return (
                err_result(
                    NOT_READY,
                    f"Workgroup '{workgroup}' is DISABLED; queries cannot run in it.",
                    key=workgroup,
                    guidance=[
                        "Use another workgroup (athena:ListWorkGroups) or ask an admin to enable it."
                    ],
                ),
                info,
            )
        return None, info
    err = res.get("error") or ""
    low = err.lower()
    if res.get("code") == NOT_FOUND or (
        "not found" in low and "workgroup" in low
    ):
        return (
            err_result(
                NOT_FOUND,
                f"Workgroup '{workgroup}' not found: {err}",
                aws_code=res.get("aws_code"),
                key=workgroup,
                guidance=[
                    "List workgroups with athena:ListWorkGroups, or use the workgroup named by the task."
                ],
            ),
            info,
        )
    if res.get("code") == CREDENTIALS_EXPIRED:
        return (
            err_result(
                CREDENTIALS_EXPIRED,
                err,
                aws_code=res.get("aws_code"),
                retriable=True,
                guidance=[
                    "Refresh the AWS credentials (re-assume the role) and run the command again."
                ],
            ),
            info,
        )
    info["note"] = (
        f"Could not read workgroup '{workgroup}' ({res.get('code')}: {err}); proceeding without its "
        "enforced output location / engine version."
    )
    return None, info


def _coerce(value: str | None, col_type: str | None) -> Any:
    if value is None:
        return None
    t = (col_type or "").lower()
    if t in _INTEGER_TYPES:
        try:
            return int(value)
        except ValueError:
            return value
    if t == "boolean" and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def _row_keys(columns: list[dict[str, Any]]) -> tuple[list[str], bool]:
    keys: list[str] = []
    seen: dict[str, int] = {}
    renamed = False
    for idx, col in enumerate(columns):
        base = col.get("label") or col.get("name") or f"_col{idx}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        if count > 1:
            renamed = True
            keys.append(f"{base}_{count}")
        else:
            keys.append(base)
    return keys, renamed


def _stats(qe: dict[str, Any]) -> dict[str, Any]:
    st = qe.get("Statistics") or {}
    return {
        "data_scanned_bytes": st.get("DataScannedInBytes"),
        "engine_execution_time_ms": st.get("EngineExecutionTimeInMillis"),
        "total_execution_time_ms": st.get("TotalExecutionTimeInMillis"),
        "queue_time_ms": st.get("QueryQueueTimeInMillis"),
    }


def _failure_code(qe: dict[str, Any], reason: str) -> tuple[str, bool]:
    low = reason.lower()
    if _NOT_SUPPORTED_MSG in low:
        return INVALID_INPUT, False
    if (
        "access denied" in low
        or "not authorized" in low
        or "accessdenied" in low
    ):
        return PERMISSION_DENIED, False
    athena_error = (qe.get("Status") or {}).get("AthenaError") or {}
    category = athena_error.get("ErrorCategory")
    if category == 2:  # USER error: bad SQL, missing column/table
        return INVALID_INPUT, False
    if category == 1:  # SYSTEM error: retriable
        return SERVICE_ERROR, True
    return SERVICE_ERROR, False


def _resolve_location(
    wg: dict[str, Any],
    workgroup: str,
    output_location: str | None,
    notes: list[str],
) -> tuple[dict[str, Any] | None, str | None, str, bool]:
    """Return (error, effective_location, source, send_location)."""
    if wg["enforced"]:
        loc = wg["output_location"]
        source = (
            "workgroup_managed"
            if (wg["managed_results"] and not loc)
            else "workgroup_enforced"
        )
        if output_location and output_location != loc:
            notes.append(
                f"Workgroup '{workgroup}' enforces its configuration; output_location "
                f"{output_location!r} was ignored and results go to {loc or 'managed query results'}."
            )
        return None, loc, source, False
    if output_location:
        return None, output_location, "caller", True
    if wg["output_location"]:
        return None, wg["output_location"], "workgroup_default", False
    if wg["managed_results"]:
        return None, None, "workgroup_managed", False
    if wg["read"]:
        return (
            invalid_input(
                f"Workgroup '{workgroup}' has no result output location and none was passed.",
                [
                    "Pass --output-location s3://<athena-results-bucket>/<prefix>/ (a bucket the role can write to)."
                ],
                value={"workgroup": workgroup},
            ),
            None,
            "",
            False,
        )
    return None, None, "unknown", False


def _start_failure(start: dict[str, Any]) -> dict[str, Any]:
    err = start.get("error") or ""
    if _NOT_SUPPORTED_MSG in err.lower():
        return err_result(
            INVALID_INPUT,
            f"Athena rejected the statement: {err}. Usual causes: more than one statement, an "
            "unsupported statement type (USE/SET/transactions/GRANT), or DDL for a table "
            "format/catalog that does not support it.",
            aws_code=start.get("aws_code"),
            guidance=[
                "Send exactly one SELECT/WITH/SHOW/DESCRIBE statement per call.",
                "Pass --database/--catalog instead of USE; drop SET/BEGIN/COMMIT.",
                "For DDL, check the target catalog/table format supports the operation in this engine version.",
            ],
        )
    by_code = {
        CREDENTIALS_EXPIRED: [
            "Refresh the AWS credentials (re-assume the role) and run again."
        ],
        PERMISSION_DENIED: [
            "The role lacks athena:StartQueryExecution on this workgroup, or the glue/s3 access it "
            "needs; use the role named by the task or another workgroup."
        ],
        THROTTLED: [
            "Athena throttled StartQueryExecution; wait a few seconds and run again."
        ],
    }
    code = start.get("code") or SERVICE_ERROR
    guidance = by_code.get(
        code, [f"StartQueryExecution failed ({code}); fix and retry."]
    )
    return err_result(
        code,
        err,
        aws_code=start.get("aws_code"),
        retriable=bool(start.get("retriable"))
        or code in (THROTTLED, CREDENTIALS_EXPIRED),
        guidance=guidance,
    )


def run_athena_query(
    call_op,
    sql: str,
    *,
    workgroup: str | None = None,
    database: str | None = None,
    catalog: str = "AwsDataCatalog",
    output_location: str | None = None,
    max_rows: int = 10_000,
    coerce_types: bool = True,
    poll_interval: float = 1.0,
    max_wait: float = 300.0,
    cancel_on_timeout: bool = False,
    sleep=None,
    clock=None,
) -> dict[str, Any]:
    """Run one Athena SQL statement to a terminal state and return all rows.

    ``call_op(operation, params)`` is the read-only call bridge bound to an
    Athena client. Returns the plain-dict result convention; never raises. When
    ``workgroup`` is ``None`` a workgroup with an output location is auto-resolved.
    """
    sleep = time.sleep if sleep is None else sleep
    clock = time.monotonic if clock is None else clock
    try:
        bad = _validate_params(
            workgroup,
            database,
            catalog,
            output_location,
            max_rows,
            poll_interval,
            max_wait,
        )
        if bad is not None:
            return bad
        bad, statement = _check_sql(sql)
        if bad is not None:
            return bad
        resolve_guidance: list[str] = []
        if not workgroup:
            resolved = resolve_workgroup_with_output(call_op, None)
            # resolve never fails hard; it always yields a workgroup to try.
            workgroup = (resolved.get("value") or {}).get(
                "workgroup", "primary"
            )
            resolve_guidance = resolved.get("guidance") or []
        return _run(
            call_op,
            statement,
            workgroup=workgroup,
            database=database,
            catalog=catalog,
            output_location=output_location,
            max_rows=max_rows,
            coerce_types=coerce_types,
            poll_interval=float(poll_interval),
            max_wait=float(max_wait),
            cancel_on_timeout=cancel_on_timeout,
            sleep=sleep,
            clock=clock,
            extra_guidance=resolve_guidance,
        )
    except Exception as exc:  # noqa: BLE001 - the command contract is "never raises"
        return err_result(
            SERVICE_ERROR,
            f"run-query internal error: {exc!r}",
            guidance=[
                "Unexpected response shape; retry once, then fall back to athena:GetQueryExecution by hand."
            ],
        )


def _run(
    call_op,
    statement: str,
    *,
    workgroup: str,
    database: str | None,
    catalog: str,
    output_location: str | None,
    max_rows: int,
    coerce_types: bool,
    poll_interval: float,
    max_wait: float,
    cancel_on_timeout: bool,
    sleep,
    clock,
    extra_guidance: list[str],
) -> dict[str, Any]:
    notes: list[str] = list(extra_guidance)

    # 1. Workgroup.
    wg_err, wg = _read_workgroup(call_op, workgroup)
    if wg_err is not None:
        return wg_err
    if wg["note"]:
        notes.append(wg["note"])
    loc_err, effective_location, source, send_location = _resolve_location(
        wg, workgroup, output_location, notes
    )
    if loc_err is not None:
        return loc_err

    # 2. Start.
    ctx: dict[str, str] = {"Catalog": catalog}
    if database:
        ctx["Database"] = database
    params: dict[str, Any] = {
        "QueryString": statement,
        "WorkGroup": workgroup,
        "QueryExecutionContext": ctx,
    }
    if send_location and effective_location:
        params["ResultConfiguration"] = {"OutputLocation": effective_location}
    start = call_op("StartQueryExecution", params)
    if not start["ok"]:
        return _start_failure(start)
    qid = (start.get("value") or {}).get("QueryExecutionId")
    if not qid:
        return err_result(
            SERVICE_ERROR,
            "StartQueryExecution returned no QueryExecutionId.",
            guidance=[
                "Retry the call; if it persists, check the workgroup in the Athena console."
            ],
        )

    base: dict[str, Any] = {
        "query_execution_id": qid,
        "workgroup": workgroup,
        "database": database,
        "catalog": catalog,
        "engine_version": wg["engine_version"],
        "output_location": effective_location,
        "output_location_source": source,
        "submitted_sql": statement,
    }

    # 3. Poll to a terminal state using the shared bounded poller.
    qe: dict[str, Any] = {}
    state = "QUEUED"

    def _terminal(res: dict[str, Any]) -> str | None:
        nonlocal qe, state
        qe = (res.get("value") or {}).get("QueryExecution") or {}
        # State persists across polls (defaults to the last observed value).
        state = (qe.get("Status") or {}).get("State") or state
        return state if state in _TERMINAL_STATES else None

    def _is_retriable(res: dict[str, Any]) -> bool:
        return bool(res.get("retriable")) or res.get("code") in (
            THROTTLED,
            "NETWORK_ERROR",
        )

    cancelled = False

    def _on_timeout() -> None:
        nonlocal cancelled
        if cancel_on_timeout:
            # StopQueryExecution cancels a running query (read-only w.r.t.
            # your data; it does not create or modify resources).
            stop = call_op("StopQueryExecution", {"QueryExecutionId": qid})
            cancelled = stop["ok"]

    poll_result = drive_poll(
        call_op,
        "GetQueryExecution",
        {"QueryExecutionId": qid},
        terminal=_terminal,
        is_retriable=_is_retriable,
        on_timeout=_on_timeout,
        interval=poll_interval,
        max_wait=max_wait,
        max_polls=_MAX_POLLS,
        max_transient=_MAX_TRANSIENT_POLL_ERRORS,
        clock=clock,
        sleep=sleep,
        key=qid,
    )
    polls = poll_result.get("polls_count", 0)

    if not poll_result["ok"]:
        if poll_result.get("code") == TIMEOUT:
            if cancelled:
                guidance = [
                    f"Query {qid} was cancelled after {max_wait:g}s (--cancel-on-timeout); narrow the "
                    "query (partition/time predicates) and run again."
                ]
            else:
                guidance = [
                    f"Query {qid} is still {state}; call athena:GetQueryExecution later or re-run with a "
                    "larger --max-wait. Do not re-submit the same SQL (it would be billed twice)."
                ]
            return err_result(
                TIMEOUT,
                f"Query {qid} did not finish within {max_wait:g}s (last state {state}).",
                retriable=not cancelled,
                key=qid,
                value={
                    **base,
                    "state": "CANCELLED" if cancelled else state,
                    "polls": polls,
                    "cancelled": cancelled,
                    **_stats(qe),
                },
                guidance=guidance,
            )
        return err_result(
            poll_result.get("code") or SERVICE_ERROR,
            poll_result.get("error") or "GetQueryExecution failed.",
            aws_code=poll_result.get("aws_code"),
            retriable=bool(poll_result.get("retriable")),
            key=qid,
            value={**base, "state": state, "polls": polls},
            guidance=[
                f"Query {qid} was started; poll athena:GetQueryExecution for it later instead of "
                "starting a new one."
            ],
        )

    stmt_type = qe.get("StatementType")
    effective_location = (qe.get("ResultConfiguration") or {}).get(
        "OutputLocation"
    ) or effective_location
    base["output_location"] = effective_location
    base["engine_version"] = (qe.get("EngineVersion") or {}).get(
        "EffectiveEngineVersion"
    ) or base["engine_version"]
    base.update(
        {
            "state": state,
            "statement_type": stmt_type,
            "polls": polls,
            **_stats(qe),
        }
    )

    if state == "FAILED":
        reason = (qe.get("Status") or {}).get(
            "StateChangeReason"
        ) or "no StateChangeReason returned"
        code, retriable = _failure_code(qe, reason)
        return err_result(
            code,
            f"Athena query {qid} FAILED: {reason}",
            retriable=retriable,
            key=qid,
            aws_code="QueryFailed",
            value={**base, "state_change_reason": reason},
            guidance=[
                f"Fix the cause named in StateChangeReason: {reason}",
                "Use DESCRIBE <table> / SHOW COLUMNS to check column and table names before re-running.",
            ],
        )
    if state == "CANCELLED":
        reason = (qe.get("Status") or {}).get(
            "StateChangeReason"
        ) or "cancelled"
        return err_result(
            CONFLICT,
            f"Athena query {qid} was CANCELLED: {reason}",
            key=qid,
            value={**base, "state_change_reason": reason},
            guidance=[
                "The query was cancelled (by a user or the workgroup's BytesScannedCutoffPerQuery); "
                "narrow it with partition predicates and run again."
            ],
        )

    # 4. Paginate results.
    columns: list[dict[str, Any]] | None = None
    keys: list[str] = []
    rows: list[dict[str, Any]] = []
    header_dropped = False
    pages = 0
    token: str | None = None
    update_count: int | None = None
    seen_tokens: set[str] = set()
    expect_header = stmt_type == "DML"
    while True:
        remaining = max_rows - len(rows)
        page_size = min(
            _MAX_PAGE_SIZE,
            remaining + (1 if pages == 0 and expect_header else 0),
        )
        rparams: dict[str, Any] = {
            "QueryExecutionId": qid,
            "MaxResults": page_size,
        }
        if token:
            rparams["NextToken"] = token
        res = call_op("GetQueryResults", rparams)
        if not res["ok"]:
            return err_result(
                res.get("code") or SERVICE_ERROR,
                f"GetQueryResults failed after {pages} page(s) for query {qid}: {res.get('error')}",
                aws_code=res.get("aws_code"),
                retriable=bool(res.get("retriable")),
                key=qid,
                is_exhaustive=False,
                value={
                    **base,
                    "columns": columns or [],
                    "rows": rows,
                    "row_count": len(rows),
                    "pages_fetched": pages,
                    "next_token": token,
                    "is_truncated": True,
                },
                guidance=[
                    f"Resume with athena:GetQueryResults QueryExecutionId={qid} NextToken={token!r}, or "
                    f"read the CSV at {effective_location}."
                ],
            )
        pages += 1
        body = res.get("value") or {}
        if update_count is None and body.get("UpdateCount") is not None:
            update_count = body.get("UpdateCount")
        rs = body.get("ResultSet") or {}
        if columns is None:
            columns = [
                {
                    "name": c.get("Name"),
                    "label": c.get("Label") or c.get("Name"),
                    "type": c.get("Type"),
                    "nullable": c.get("Nullable"),
                    "precision": c.get("Precision"),
                    "scale": c.get("Scale"),
                }
                for c in (rs.get("ResultSetMetadata") or {}).get("ColumnInfo")
                or []
            ]
            keys, renamed = _row_keys(columns)
            if renamed:
                notes.append(
                    "Duplicate column labels were suffixed (_2, _3, ...) in row keys; alias them in SQL."
                )
        raw_rows = rs.get("Rows") or []
        if pages == 1 and raw_rows:
            first = [
                d.get("VarCharValue") for d in raw_rows[0].get("Data") or []
            ]
            labels = [c["label"] for c in columns]
            names = [c["name"] for c in columns]
            if expect_header or (
                columns and (first == labels or first == names)
            ):
                raw_rows = raw_rows[1:]
                header_dropped = True
        for raw in raw_rows:
            if len(rows) >= max_rows:
                break
            cells = [d.get("VarCharValue") for d in raw.get("Data") or []]
            cells += [None] * (len(keys) - len(cells))
            rows.append(
                {
                    k: (_coerce(v, columns[i]["type"]) if coerce_types else v)
                    for i, (k, v) in enumerate(zip(keys, cells))
                }
            )
        token = body.get("NextToken")
        if not token:
            break
        if token in seen_tokens:
            notes.append(
                "GetQueryResults repeated a NextToken; pagination stopped to avoid a loop."
            )
            token = None
            break
        seen_tokens.add(token)
        if len(rows) >= max_rows:
            break

    truncated = bool(token)
    guidance = [_VERBATIM_GUIDANCE, _NULL_GUIDANCE, *notes]
    if truncated:
        guidance.insert(
            0,
            (
                f"Result truncated at max_rows={max_rows}; more rows exist. Re-run with a larger "
                f"--max-rows, add a tighter WHERE, or resume with athena:GetQueryResults "
                f"QueryExecutionId={qid} NextToken={token!r}."
            ),
        )
    value = {
        **base,
        "columns": columns or [],
        "rows": rows,
        "row_count": len(rows),
        "header_row_dropped": header_dropped,
        "pages_fetched": pages,
        "is_truncated": truncated,
        "next_token": token if truncated else None,
        "max_rows": max_rows,
        "update_count": update_count,
        "null_representation": "SQL NULL -> JSON null (render as NULL)",
    }
    return ok_result(
        value, key=qid, is_exhaustive=not truncated, guidance=guidance
    )


class RunQueryCommand(BasicCommand):
    NAME = "run-query"
    DESCRIPTION = (
        "Run a single read-only Athena statement to completion and return every "
        "row in one call. Submits the query, polls it to a terminal state, and "
        "paginates all result pages. Multi-statement batches, writes and DDL are "
        "refused. When no --workgroup is given, a workgroup with a result output "
        "location is auto-resolved."
    )
    EXAMPLES = (
        "To run a query in the default workgroup::\n\n"
        "  aws athena run-query --sql \"SELECT * FROM mydb.mytable LIMIT 100\"\n\n"
        "To target a workgroup and database with a row cap::\n\n"
        "  aws athena run-query --sql \"SELECT count(*) FROM logs\" "
        "--workgroup analytics --database mydb --max-rows 1\n"
    )
    ARG_TABLE = [
        {
            "name": "sql",
            "required": True,
            "help_text": "The single SQL statement to run (read-only).",
            "cli_type_name": "string",
        },
        {
            "name": "workgroup",
            "help_text": "Athena workgroup. Omit to auto-resolve one with an output location.",
            "cli_type_name": "string",
        },
        {
            "name": "database",
            "help_text": "Glue database for unqualified names.",
            "cli_type_name": "string",
        },
        {
            "name": "catalog",
            "default": "AwsDataCatalog",
            "help_text": "Data catalog (default AwsDataCatalog).",
            "cli_type_name": "string",
        },
        {
            "name": "output-location",
            "help_text": "s3:// URI for results; overrides the workgroup default unless enforced.",
            "cli_type_name": "string",
        },
        {
            "name": "max-rows",
            "default": 10000,
            "cli_type_name": "integer",
            "help_text": "Maximum rows to return (default 10000).",
        },
        {
            "name": "poll-interval",
            "default": 1.0,
            "cli_type_name": "double",
            "help_text": "Initial seconds between GetQueryExecution polls (default 1.0).",
        },
        {
            "name": "max-wait",
            "default": 300.0,
            "cli_type_name": "double",
            "help_text": "Maximum seconds to wait for the query (default 300).",
        },
        {
            "name": "cancel-on-timeout",
            "action": "store_true",
            "default": False,
            "help_text": "Cancel the query (StopQueryExecution) if --max-wait is exceeded.",
        },
        {
            "name": "no-coerce-types",
            "action": "store_true",
            "default": False,
            "help_text": "Return all cells as strings instead of coercing integer/boolean columns.",
        },
    ]

    def __init__(
        self,
        session,
        client=None,
        stdout=None,
        stderr=None,
        sleep=None,
        clock=None,
    ):
        super().__init__(session)
        self._client = client
        self._stdout = stdout
        self._stderr = stderr
        self._sleep = sleep
        self._clock = clock

    def _run_main(self, parsed_args, parsed_globals):
        client = self._client or create_read_client(
            self._session, "athena", parsed_globals
        )

        def call_op(operation, params):
            return call(client, operation, params)

        result = run_athena_query(
            call_op,
            parsed_args.sql,
            workgroup=parsed_args.workgroup,
            database=parsed_args.database,
            catalog=parsed_args.catalog,
            output_location=parsed_args.output_location,
            max_rows=parsed_args.max_rows,
            coerce_types=not parsed_args.no_coerce_types,
            poll_interval=parsed_args.poll_interval,
            max_wait=parsed_args.max_wait,
            cancel_on_timeout=parsed_args.cancel_on_timeout,
            sleep=self._sleep,
            clock=self._clock,
        )
        return emit_result(
            self.NAME,
            result,
            parsed_globals,
            self._session,
            self._stdout,
            self._stderr,
        )
