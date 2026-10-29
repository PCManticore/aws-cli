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
"""Name resolvers for the read-only aggregating commands.

Two resolvers live here because both trade a little extra read traffic for a
much more forgiving command:

* :func:`resolve_workgroup_with_output` -- Athena results need an output
  location. If the caller names a workgroup it is honoured verbatim; otherwise we
  pick a workgroup whose *effective* configuration actually has one (including an
  enforced configuration), rather than blindly defaulting to ``primary`` which
  may have none.
* :func:`resolve_alarm` -- a CloudWatch alarm lookup that tries an exact name
  first, then a bounded name-prefix query, then a bounded, deterministic fuzzy
  match over those prefix candidates. The fuzzy step never issues an unbounded
  scan; it only ranks names already fetched.
"""

from __future__ import annotations

import difflib
from typing import Any, Callable

from awscli.customizations.compositereadonly.helpers import (
    NOT_FOUND,
    err_result,
    ok_result,
)

# Bounded paging for workgroup/alarm enumeration so a resolver can never run
# away on an account with thousands of workgroups or alarms.
_MAX_WORKGROUP_PAGES = 10
_MAX_ALARM_PAGES = 5
_FUZZY_CUTOFF = 0.6

CallFn = Callable[[str, dict[str, Any]], dict[str, Any]]


def _workgroup_output(cfg: dict[str, Any]) -> tuple[str | None, bool]:
    """Return (effective_output_location, has_managed_results) for a WG config."""
    loc = (cfg.get("ResultConfiguration") or {}).get("OutputLocation")
    managed = bool(
        (cfg.get("ManagedQueryResultsConfiguration") or {}).get("Enabled")
    )
    return loc, managed


def resolve_workgroup_with_output(
    call_op: CallFn, requested: str | None
) -> dict[str, Any]:
    """Resolve a workgroup whose effective config can place query results.

    ``call_op(operation, params)`` is the read-only call bridge bound to an
    Athena client. When ``requested`` is given it is returned untouched (the
    caller's explicit choice always wins). When it is ``None`` we read
    ``primary`` first and, only if it has neither an output location nor managed
    results, enumerate other enabled workgroups (bounded) and pick the first that
    does -- reporting the choice in ``guidance``.

    The value is ``{"workgroup": name, "auto_resolved": bool,
    "output_location": loc, "managed_results": bool}``.
    """
    if requested:
        # Honour an explicit workgroup verbatim; its output location (or an
        # enforced one) is resolved later when the query runs.
        return ok_result(
            {
                "workgroup": requested,
                "auto_resolved": False,
                "output_location": None,
                "managed_results": False,
            }
        )

    guidance: list[str] = []
    primary = call_op("GetWorkGroup", {"WorkGroup": "primary"})
    if primary["ok"]:
        cfg = ((primary.get("value") or {}).get("WorkGroup") or {}).get(
            "Configuration"
        ) or {}
        loc, managed = _workgroup_output(cfg)
        if loc or managed:
            return ok_result(
                {
                    "workgroup": "primary",
                    "auto_resolved": False,
                    "output_location": loc,
                    "managed_results": managed,
                }
            )

    # 'primary' cannot place results (or could not be read); look for a workgroup
    # that can. Enumerate enabled workgroups and read each effective config.
    token: str | None = None
    for _ in range(_MAX_WORKGROUP_PAGES):
        params: dict[str, Any] = {}
        if token:
            params["NextToken"] = token
        listing = call_op("ListWorkGroups", params)
        if not listing["ok"]:
            # Fall back to 'primary'; the query step will surface a precise
            # "no output location" error if it truly has none.
            guidance.append(
                "Could not list workgroups to auto-resolve an output location; "
                "falling back to 'primary'. Pass --workgroup or --output-location."
            )
            return ok_result(
                {
                    "workgroup": "primary",
                    "auto_resolved": False,
                    "output_location": None,
                    "managed_results": False,
                },
                guidance=guidance,
            )
        summaries = (listing.get("value") or {}).get("WorkGroups") or []
        for summ in summaries:
            name = summ.get("Name")
            if not name or summ.get("State") == "DISABLED":
                continue
            wg = call_op("GetWorkGroup", {"WorkGroup": name})
            if not wg["ok"]:
                continue
            cfg = ((wg.get("value") or {}).get("WorkGroup") or {}).get(
                "Configuration"
            ) or {}
            loc, managed = _workgroup_output(cfg)
            if loc or managed:
                guidance.append(
                    f"No --workgroup given and 'primary' has no result output "
                    f"location; auto-selected workgroup {name!r} "
                    f"({'managed results' if managed and not loc else loc})."
                )
                return ok_result(
                    {
                        "workgroup": name,
                        "auto_resolved": True,
                        "output_location": loc,
                        "managed_results": managed,
                    },
                    guidance=guidance,
                )
        token = (listing.get("value") or {}).get("NextToken")
        if not token:
            break

    # Nothing better than 'primary' was found; return it and let the query step
    # emit the precise "no output location" error if applicable.
    guidance.append(
        "No workgroup with a result output location was found; using 'primary'. "
        "Pass --output-location='s3://<bucket>/<prefix>/' to run the query."
    )
    return ok_result(
        {
            "workgroup": "primary",
            "auto_resolved": False,
            "output_location": None,
            "managed_results": False,
        },
        guidance=guidance,
    )


_ALARM_TYPES = ["MetricAlarm", "CompositeAlarm"]


def _alarms_from(value: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten DescribeAlarms metric + composite alarms into one list."""
    return list(value.get("MetricAlarms") or []) + list(
        value.get("CompositeAlarms") or []
    )


def _describe_prefix(
    call_op: CallFn, prefix: str
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """DescribeAlarms by name prefix, following NextToken (bounded)."""
    alarms: list[dict[str, Any]] = []
    token: str | None = None
    for _ in range(_MAX_ALARM_PAGES):
        params: dict[str, Any] = {
            "AlarmNamePrefix": prefix,
            "AlarmTypes": _ALARM_TYPES,
        }
        if token:
            params["NextToken"] = token
        res = call_op("DescribeAlarms", params)
        if not res["ok"]:
            return alarms, res
        value = res.get("value") or {}
        alarms.extend(_alarms_from(value))
        token = value.get("NextToken")
        if not token:
            break
    return alarms, None


def resolve_alarm(call_op: CallFn, name: str) -> dict[str, Any]:
    """Resolve a CloudWatch alarm by exact name, then prefix, then fuzzy match.

    ``call_op(operation, params)`` is the read-only call bridge bound to a
    CloudWatch client. Resolution order:

    1. **Exact** -- ``DescribeAlarms AlarmNames=[name]``. A hit short-circuits
       with no fallback, so an exact name is never silently rerouted.
    2. **Prefix** -- ``DescribeAlarms AlarmNamePrefix=name`` (bounded paging). A
       single candidate is used; a case-insensitive exact match among several is
       preferred before considering it ambiguous.
    3. **Fuzzy** -- a bounded, deterministic ``difflib`` ranking over the prefix
       candidates only (no extra scan). A single close match above the cutoff is
       used; ties and misses return NOT_FOUND with the candidate list.

    The value is ``{"alarm": <alarm dict>, "kind": "metric_alarm"|
    "composite_alarm", "match": "exact"|"prefix"|"fuzzy", "guidance": [...]}``.
    """
    # 1. Exact.
    exact = call_op(
        "DescribeAlarms", {"AlarmNames": [name], "AlarmTypes": _ALARM_TYPES}
    )
    if not exact["ok"]:
        return exact
    metric = (exact.get("value") or {}).get("MetricAlarms") or []
    composite = (exact.get("value") or {}).get("CompositeAlarms") or []
    if metric or composite:
        alarm = (metric or composite)[0]
        return ok_result(
            {
                "alarm": alarm,
                "kind": "metric_alarm" if metric else "composite_alarm",
                "match": "exact",
                "guidance": [],
            }
        )

    # 2. Prefix.
    candidates, err = _describe_prefix(call_op, name)
    if err is not None:
        return err
    if not candidates:
        return err_result(
            NOT_FOUND,
            f"No alarm named or prefixed by {name!r}.",
            key=name,
            guidance=[
                "Alarm names are exact and case-sensitive; list candidates with "
                "'aws cloudwatch describe-alarms --alarm-name-prefix <prefix>'.",
                "Alarms are regional: pass --region.",
            ],
        )
    by_name = {a.get("AlarmName"): a for a in candidates if a.get("AlarmName")}
    if len(candidates) == 1:
        return _prefix_hit(candidates[0], name)
    # Several share the prefix: prefer a unique case-insensitive exact match.
    ci_matches = [
        a
        for a in candidates
        if (a.get("AlarmName") or "").casefold() == name.casefold()
    ]
    if len(ci_matches) == 1:
        return _prefix_hit(ci_matches[0], name, note="case-insensitive exact")

    # 3. Fuzzy over the already-fetched candidate names (bounded, deterministic).
    names_sorted = sorted(by_name)
    close = difflib.get_close_matches(
        name, names_sorted, n=2, cutoff=_FUZZY_CUTOFF
    )
    if len(close) == 1 or (len(close) >= 2 and close[0] != close[1]):
        chosen = by_name[close[0]]
        return _prefix_hit(chosen, name, match="fuzzy")
    # Ambiguous: report the candidates rather than guessing.
    listed = ", ".join(names_sorted[:10])
    return err_result(
        NOT_FOUND,
        f"{name!r} is ambiguous: {len(candidates)} alarms share this prefix "
        "and none is an unambiguous match.",
        key=name,
        guidance=[
            f"Matching alarms: {listed}"
            + (" ..." if len(names_sorted) > 10 else ""),
            "Pass --alarm-name with one of these exact names.",
        ],
    )


def _prefix_hit(
    alarm: dict[str, Any],
    requested: str,
    *,
    match: str = "prefix",
    note: str = "",
) -> dict[str, Any]:
    kind = "composite_alarm" if alarm.get("AlarmRule") else "metric_alarm"
    resolved = alarm.get("AlarmName")
    detail = f" ({note})" if note else ""
    guidance = (
        []
        if match == "exact"
        else [
            f"No alarm named exactly {requested!r}; resolved to {resolved!r} "
            f"by {match} match{detail}."
        ]
    )
    return ok_result(
        {"alarm": alarm, "kind": kind, "match": match, "guidance": guidance}
    )
