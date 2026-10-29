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
"""Unit tests for the compositereadonly name resolvers (workgroup + alarm)."""

from awscli.customizations.compositereadonly import resolve
from awscli.customizations.compositereadonly.helpers import (
    NOT_FOUND,
    PERMISSION_DENIED,
    assert_read_only,
    err_result,
    ok_result,
)


class Scripted:
    """A scripted read-only call bridge; records calls, returns queued dicts."""

    def __init__(self):
        self.calls = []
        self._responses = {}

    def queue(self, operation, *results):
        self._responses.setdefault(operation, []).extend(results)
        return self

    def __call__(self, operation, params):
        assert_read_only(operation)
        self.calls.append((operation, params))
        queued = self._responses.get(operation)
        if queued:
            return queued.pop(0)
        return ok_result({})

    def ops(self):
        return [op for op, _ in self.calls]


def _wg(name="primary", output=None, managed=False, state="ENABLED"):
    cfg = {}
    if output is not None:
        cfg["ResultConfiguration"] = {"OutputLocation": output}
    if managed:
        cfg["ManagedQueryResultsConfiguration"] = {"Enabled": True}
    return ok_result(
        {"WorkGroup": {"Name": name, "State": state, "Configuration": cfg}}
    )


# ----------------------------------------------- resolve_workgroup_with_output


def test_explicit_workgroup_is_honoured_without_any_call():
    call = Scripted()
    r = resolve.resolve_workgroup_with_output(call, "analytics")
    assert r["ok"] is True
    assert r["value"]["workgroup"] == "analytics"
    assert r["value"]["auto_resolved"] is False
    # An explicit choice never triggers a read.
    assert call.calls == []


def test_primary_with_output_location_is_used():
    call = Scripted()
    call.queue("GetWorkGroup", _wg(output="s3://primary/"))
    r = resolve.resolve_workgroup_with_output(call, None)
    assert r["value"]["workgroup"] == "primary"
    assert r["value"]["output_location"] == "s3://primary/"
    assert r["value"]["auto_resolved"] is False
    assert call.ops() == ["GetWorkGroup"]


def test_primary_with_managed_results_is_used():
    call = Scripted()
    call.queue("GetWorkGroup", _wg(managed=True))
    r = resolve.resolve_workgroup_with_output(call, None)
    assert r["value"]["managed_results"] is True
    assert r["value"]["workgroup"] == "primary"


def test_auto_selects_enabled_workgroup_with_output():
    call = Scripted()
    # primary has neither output nor managed results.
    call.queue("GetWorkGroup", _wg(output=None))
    call.queue(
        "ListWorkGroups",
        ok_result(
            {
                "WorkGroups": [
                    {"Name": "disabled-wg", "State": "DISABLED"},
                    {"Name": "noname-skip"},  # has a name, no output
                    {"Name": "analytics", "State": "ENABLED"},
                ]
            }
        ),
    )
    # First enabled candidate with no name key filtered; 'noname-skip' read but
    # has no output, 'analytics' has one and wins.
    call.queue("GetWorkGroup", _wg(name="noname-skip", output=None))
    call.queue("GetWorkGroup", _wg(name="analytics", output="s3://a/"))
    r = resolve.resolve_workgroup_with_output(call, None)
    assert r["value"]["workgroup"] == "analytics"
    assert r["value"]["auto_resolved"] is True
    assert r["value"]["output_location"] == "s3://a/"
    assert any("auto-selected" in g for g in r["guidance"])


def test_auto_selects_managed_workgroup_reports_managed():
    call = Scripted()
    call.queue("GetWorkGroup", _wg(output=None))
    call.queue(
        "ListWorkGroups",
        ok_result({"WorkGroups": [{"Name": "mgd", "State": "ENABLED"}]}),
    )
    call.queue("GetWorkGroup", _wg(name="mgd", managed=True))
    r = resolve.resolve_workgroup_with_output(call, None)
    assert r["value"]["workgroup"] == "mgd"
    assert r["value"]["managed_results"] is True
    # Guidance names managed results when there is no output location.
    assert any("managed results" in g for g in r["guidance"])


def test_skips_entries_without_name_and_failed_getworkgroup():
    call = Scripted()
    call.queue("GetWorkGroup", _wg(output=None))
    call.queue(
        "ListWorkGroups",
        ok_result(
            {
                "WorkGroups": [
                    {"State": "ENABLED"},  # no Name -> skipped, no read
                    {"Name": "broken", "State": "ENABLED"},
                    {"Name": "good", "State": "ENABLED"},
                ]
            }
        ),
    )
    call.queue(
        "GetWorkGroup", err_result(PERMISSION_DENIED, "denied")
    )  # 'broken' read fails -> continue
    call.queue("GetWorkGroup", _wg(name="good", output="s3://g/"))
    r = resolve.resolve_workgroup_with_output(call, None)
    assert r["value"]["workgroup"] == "good"


def test_listworkgroups_failure_falls_back_to_primary():
    call = Scripted()
    call.queue("GetWorkGroup", _wg(output=None))
    call.queue("ListWorkGroups", err_result(PERMISSION_DENIED, "denied"))
    r = resolve.resolve_workgroup_with_output(call, None)
    assert r["value"]["workgroup"] == "primary"
    assert r["value"]["auto_resolved"] is False
    assert any("falling back to 'primary'" in g for g in r["guidance"])


def test_pagination_followed_then_nothing_found():
    call = Scripted()
    call.queue("GetWorkGroup", _wg(output=None))  # primary
    # Page 1 has one candidate with no output; a NextToken leads to an empty
    # page 2; resolver then exhausts and falls back to 'primary'.
    call.queue(
        "ListWorkGroups",
        ok_result(
            {
                "WorkGroups": [{"Name": "x", "State": "ENABLED"}],
                "NextToken": "tok2",
            }
        ),
    )
    call.queue("GetWorkGroup", _wg(name="x", output=None))
    call.queue("ListWorkGroups", ok_result({"WorkGroups": []}))
    r = resolve.resolve_workgroup_with_output(call, None)
    assert r["value"]["workgroup"] == "primary"
    assert r["value"]["auto_resolved"] is False
    assert any("No workgroup with a result output" in g for g in r["guidance"])
    # Pagination token on page 1 was followed into a second ListWorkGroups.
    assert call.ops().count("ListWorkGroups") == 2


def test_workgroup_listing_stops_at_page_cap():
    # Every page carries a NextToken and only unusable workgroups, so the loop
    # runs the full _MAX_WORKGROUP_PAGES and then falls back to 'primary'
    # instead of paging forever.
    call = Scripted()
    call.queue("GetWorkGroup", _wg(output=None))  # primary
    for _ in range(resolve._MAX_WORKGROUP_PAGES + 2):
        # No "Name" key -> each candidate is skipped (no GetWorkGroup read).
        call.queue(
            "ListWorkGroups",
            ok_result(
                {"WorkGroups": [{"State": "ENABLED"}], "NextToken": "more"}
            ),
        )
    r = resolve.resolve_workgroup_with_output(call, None)
    assert r["value"]["workgroup"] == "primary"
    assert r["value"]["auto_resolved"] is False
    # The cap bounded the enumeration to exactly _MAX_WORKGROUP_PAGES reads.
    assert call.ops().count("ListWorkGroups") == resolve._MAX_WORKGROUP_PAGES


def test_alarm_prefix_paging_stops_at_page_cap():
    call = Scripted()
    call.queue("DescribeAlarms", _describe())  # exact miss
    # Every prefix page returns a candidate and a NextToken; _describe_prefix
    # must stop after _MAX_ALARM_PAGES reads rather than chase the last token,
    # whatever the eventual match decision is.
    for i in range(resolve._MAX_ALARM_PAGES + 2):
        call.queue(
            "DescribeAlarms",
            _describe(metric=[_metric_alarm(f"svc-{i}")], token="more"),
        )
    resolve.resolve_alarm(call, "svc")
    # 1 exact + exactly _MAX_ALARM_PAGES prefix reads (the cap held); the
    # surplus queued pages were never requested.
    assert call.ops().count("DescribeAlarms") == (1 + resolve._MAX_ALARM_PAGES)


def test_primary_unreadable_still_enumerates():
    call = Scripted()
    # primary read fails -> skip the has-output short-circuit, go to listing.
    call.queue("GetWorkGroup", err_result(NOT_FOUND, "gone"))
    call.queue(
        "ListWorkGroups",
        ok_result({"WorkGroups": [{"Name": "a", "State": "ENABLED"}]}),
    )
    call.queue("GetWorkGroup", _wg(name="a", output="s3://a/"))
    r = resolve.resolve_workgroup_with_output(call, None)
    assert r["value"]["workgroup"] == "a"
    assert r["value"]["auto_resolved"] is True


# --------------------------------------------------------------- resolve_alarm


def _metric_alarm(name, rule=False):
    a = {"AlarmName": name}
    if rule:
        a["AlarmRule"] = "ALARM(x)"
    return a


def _describe(metric=None, composite=None, token=None):
    body = {
        "MetricAlarms": list(metric or []),
        "CompositeAlarms": list(composite or []),
    }
    if token:
        body["NextToken"] = token
    return ok_result(body)


def test_alarm_exact_metric_hit_short_circuits():
    call = Scripted()
    call.queue("DescribeAlarms", _describe(metric=[_metric_alarm("cpu-high")]))
    r = resolve.resolve_alarm(call, "cpu-high")
    assert r["ok"] is True
    assert r["value"]["match"] == "exact"
    assert r["value"]["kind"] == "metric_alarm"
    # Exact hit means no prefix fallback was attempted.
    assert call.ops() == ["DescribeAlarms"]


def test_alarm_exact_composite_hit():
    call = Scripted()
    call.queue(
        "DescribeAlarms",
        _describe(composite=[_metric_alarm("agg", rule=True)]),
    )
    r = resolve.resolve_alarm(call, "agg")
    assert r["value"]["kind"] == "composite_alarm"
    assert r["value"]["match"] == "exact"


def test_alarm_exact_describe_error_is_returned():
    call = Scripted()
    call.queue("DescribeAlarms", err_result(PERMISSION_DENIED, "denied"))
    r = resolve.resolve_alarm(call, "cpu")
    assert r["ok"] is False
    assert r["code"] == PERMISSION_DENIED


def test_alarm_prefix_single_candidate():
    call = Scripted()
    call.queue("DescribeAlarms", _describe())  # exact miss
    call.queue("DescribeAlarms", _describe(metric=[_metric_alarm("cpu-high")]))
    r = resolve.resolve_alarm(call, "cpu")
    assert r["ok"] is True
    assert r["value"]["match"] == "prefix"
    assert any("prefix match" in g for g in r["value"]["guidance"])


def test_alarm_prefix_describe_error_propagates():
    call = Scripted()
    call.queue("DescribeAlarms", _describe())  # exact miss
    call.queue("DescribeAlarms", err_result(PERMISSION_DENIED, "denied"))
    r = resolve.resolve_alarm(call, "cpu")
    assert r["ok"] is False
    assert r["code"] == PERMISSION_DENIED


def test_alarm_no_candidates_is_not_found():
    call = Scripted()
    call.queue("DescribeAlarms", _describe())  # exact miss
    call.queue("DescribeAlarms", _describe())  # prefix miss
    r = resolve.resolve_alarm(call, "ghost")
    assert r["ok"] is False
    assert r["code"] == NOT_FOUND
    assert r["key"] == "ghost"


def test_alarm_prefix_paging_is_followed():
    call = Scripted()
    call.queue("DescribeAlarms", _describe())  # exact miss
    call.queue(
        "DescribeAlarms",
        _describe(metric=[_metric_alarm("cpu-a")], token="more"),
    )
    call.queue("DescribeAlarms", _describe(metric=[_metric_alarm("cpu-b")]))
    r = resolve.resolve_alarm(call, "cpu-a")
    # Two prefix candidates fetched across pages; exact-ci picks 'cpu-a'.
    assert r["ok"] is True
    assert r["value"]["alarm"]["AlarmName"] == "cpu-a"


def test_alarm_case_insensitive_exact_among_several():
    call = Scripted()
    call.queue("DescribeAlarms", _describe())  # exact miss (case differs)
    call.queue(
        "DescribeAlarms",
        _describe(
            metric=[_metric_alarm("CPU-High"), _metric_alarm("cpu-low")]
        ),
    )
    r = resolve.resolve_alarm(call, "cpu-high")
    assert r["ok"] is True
    assert r["value"]["alarm"]["AlarmName"] == "CPU-High"
    assert any("case-insensitive" in g for g in r["value"]["guidance"])


def test_alarm_fuzzy_single_close_match():
    call = Scripted()
    call.queue("DescribeAlarms", _describe())  # exact miss
    call.queue(
        "DescribeAlarms",
        _describe(
            metric=[
                _metric_alarm("cpu-utilization-high"),
                _metric_alarm("totally-different-name"),
            ]
        ),
    )
    # 'cpu-utilisation-high' (British spelling) fuzzily matches the first only.
    r = resolve.resolve_alarm(call, "cpu-utilisation-high")
    assert r["ok"] is True
    assert r["value"]["match"] == "fuzzy"
    assert r["value"]["alarm"]["AlarmName"] == "cpu-utilization-high"


def test_alarm_ambiguous_prefix_is_not_found_with_candidates():
    call = Scripted()
    call.queue("DescribeAlarms", _describe())  # exact miss
    # Many similar names, none an unambiguous match.
    names = [f"svc-alarm-{i}" for i in range(12)]
    call.queue(
        "DescribeAlarms",
        _describe(metric=[_metric_alarm(n) for n in names]),
    )
    r = resolve.resolve_alarm(call, "svc")
    assert r["ok"] is False
    assert r["code"] == NOT_FOUND
    assert "ambiguous" in r["error"]
    # More than 10 candidates -> the guidance list is elided with "...".
    assert any("..." in g for g in r["guidance"])
