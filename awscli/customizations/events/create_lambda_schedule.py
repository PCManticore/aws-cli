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
"""``aws events create-lambda-schedule`` -- schedule a Lambda via EventBridge.

Replicates CDK's ``events.Rule`` + ``LambdaFunction`` target pattern: it creates
an EventBridge scheduled rule and wires it to invoke a Lambda function, in one
call instead of four hand-rolled ones.

Procedure (each AWS call goes through the write-side call bridge, and every
resource created is tracked so a later failure rolls the earlier ones back):

1. ``lambda:GetFunction`` -- verify the function exists and read its ARN.
2. ``events:PutRule`` -- create the scheduled rule and read its ``RuleArn``.
3. ``events:PutTargets`` -- add the Lambda function as a target of the rule
   (with the optional constant JSON ``--input``).
4. ``lambda:AddPermission`` -- add a resource-based policy statement letting
   ``events.amazonaws.com`` invoke the function, scoped to the rule's ARN.

Rollback: if ``PutTargets`` fails the rule is deleted; if ``AddPermission``
fails the targets are removed and then the rule is deleted. Both happen
automatically, LIFO, via the resource tracker.

Idempotency: running the command twice with the same arguments succeeds both
times without errors or duplicates. ``PutRule`` and ``PutTargets`` are upserts,
and an ``AddPermission`` ``ResourceConflictException`` (the statement already
exists) is treated as success. A ``DescribeRule`` probe up front decides whether
the run is a create or an update: the result's ``Idempotent`` field is ``true``
when the rule pre-existed, and a pre-existing rule (or permission) is never
registered for rollback, so a later failure only undoes resources this run
actually created.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from awscli.customizations.commands import BasicCommand
from awscli.customizations.compositecreate.cleanup import make_delete_cleanup
from awscli.customizations.compositecreate.helpers import (
    NOT_FOUND,
    SERVICE_ERROR,
    ResourceTracker,
    call,
    check_non_empty_str,
    create_client,
    emit_result,
    err_result,
    invalid_input,
    ok_result,
    step,
)
from awscli.customizations.compositecreate.progress import ProgressReporter

logger = logging.getLogger(__name__)

# Valid EventBridge rule states.
_VALID_STATES = frozenset({"ENABLED", "DISABLED"})
# A short, stable id for the single Lambda target of the rule.
_TARGET_ID = "target-1"
# This command creates the rule, its target and the invoke permission in four
# AWS calls (GetFunction, PutRule, PutTargets, AddPermission).
_TOTAL_STEPS = 4


def _derive_rule_name(function_name: str) -> str:
    """Derive a rule name from the function name when one is not given.

    ``function_name`` may be a bare name or a full Lambda ARN
    (``arn:aws:lambda:<region>:<acct>:function:<name>[:<qualifier>]``); in the
    ARN case only the function's short name is used. The derived rule name is
    ``<short-name>-schedule``.
    """
    name = function_name
    if name.startswith("arn:"):
        parts = name.split(":function:")
        if len(parts) == 2:
            # Strip any trailing :<qualifier> (alias or version).
            name = parts[1].split(":")[0]
    return f"{name}-schedule"


def _validate_params(
    function_name: Any,
    schedule: Any,
    rule_name: Any,
    input_json: Any,
    description: Any,
    state: Any,
) -> dict[str, Any] | None:
    """Validate CLI input before any AWS call; return an error result or None.

    Checks the two required strings, the optional strings, that ``--input`` (if
    given) is valid JSON, and that ``--state`` is one of ENABLED/DISABLED.
    Returns an :func:`invalid_input` result (exit code 252) on the first
    problem, or ``None`` when everything is valid.
    """
    err = check_non_empty_str("function-name", function_name, required=True)
    if err is not None:
        return invalid_input(
            err,
            ["Pass --function-name <name-or-arn> of an existing Lambda."],
        )
    err = check_non_empty_str("schedule", schedule, required=True)
    if err is not None:
        return invalid_input(
            err,
            [
                "Pass --schedule with a rate() or cron() expression, e.g. "
                "'rate(5 minutes)' or 'cron(0 12 * * ? *)'."
            ],
        )
    err = check_non_empty_str("rule-name", rule_name, required=False)
    if err is not None:
        return invalid_input(
            err,
            ["Omit --rule-name to derive it, or pass a non-empty name."],
        )
    err = check_non_empty_str("description", description, required=False)
    if err is not None:
        return invalid_input(
            err,
            ["Omit --description or pass a non-empty string."],
        )
    if input_json is not None:
        err = check_non_empty_str("input", input_json, required=False)
        if err is not None:
            return invalid_input(
                err,
                ["Omit --input or pass a non-empty JSON string."],
            )
        try:
            json.loads(input_json)
        except (ValueError, TypeError) as exc:
            return invalid_input(
                f"input is not valid JSON: {exc}",
                [
                    "Pass --input as a JSON value, e.g. '{\"key\":\"value\"}' "
                    "or '\"a-string\"'."
                ],
            )
    if not isinstance(state, str) or state not in _VALID_STATES:
        return invalid_input(
            f"state must be one of ENABLED or DISABLED, got {state!r}.",
            ["Pass --state ENABLED or --state DISABLED."],
        )
    return None


def create_lambda_schedule(
    events_call,
    lambda_call,
    *,
    function_name: str,
    schedule: str,
    rule_name: str | None = None,
    input_json: str | None = None,
    description: str | None = None,
    state: str = "ENABLED",
    stderr=None,
) -> dict[str, Any]:
    """Create a scheduled EventBridge rule that invokes a Lambda function.

    ``events_call(operation, params)`` and ``lambda_call(operation, params)`` are
    the write-side call bridge bound to an EventBridge client and a Lambda client
    respectively; each returns the plain-dict result convention and never raises.
    Returns an :func:`ok_result` summarizing the created resources on success, or
    an error result (with any partially created resources already rolled back) on
    failure. Never raises.
    """
    try:
        bad = _validate_params(
            function_name,
            schedule,
            rule_name,
            input_json,
            description,
            state,
        )
        if bad is not None:
            return bad

        effective_rule_name = rule_name or _derive_rule_name(function_name)
        tracker = ResourceTracker()
        progress = ProgressReporter(total_steps=_TOTAL_STEPS, stderr=stderr)

        # 1. Verify the function exists and read its ARN.
        progress.report(1, "Verifying Lambda function")
        progress.detail("lambda:GetFunction", FunctionName=function_name)
        get_fn = lambda_call("GetFunction", {"FunctionName": function_name})
        if not get_fn["ok"]:
            progress.failure(1, "Verify Lambda function", get_fn.get("error"))
            if get_fn.get("code") == NOT_FOUND:
                return err_result(
                    NOT_FOUND,
                    f"Lambda function {function_name!r} was not found: "
                    f"{get_fn.get('error')}",
                    aws_code=get_fn.get("aws_code"),
                    key=function_name,
                    guidance=[
                        "Create the function first, or pass the name/ARN of an "
                        "existing one (aws lambda list-functions)."
                    ],
                )
            return get_fn
        function_arn = (
            (get_fn.get("value") or {}).get("Configuration") or {}
        ).get("FunctionArn")
        if not function_arn:
            return err_result(
                SERVICE_ERROR,
                "GetFunction returned no Configuration.FunctionArn.",
                key=function_name,
                guidance=[
                    "Retry; if it persists, inspect the function with "
                    "aws lambda get-function."
                ],
            )
        progress.success(1, "Verified Lambda function", resource_id=function_arn)

        # Idempotency probe: look up the rule before creating anything. PutRule/
        # PutTargets are upserts and AddPermission's CONFLICT is treated as
        # success, so we proceed with the create steps either way -- this read
        # only decides whether we report a create or an update and whether the
        # rule is registered for rollback (a pre-existing rule we did not create
        # must never be deleted on failure). A probe error (NOT_FOUND or any
        # other) is non-fatal: we fall back to treating the rule as new, which
        # keeps full rollback tracking for a rule we may genuinely create.
        progress.detail("events:DescribeRule", Name=effective_rule_name)
        describe_res = events_call("DescribeRule", {"Name": effective_rule_name})
        rule_preexisted = bool(describe_res.get("ok"))
        if rule_preexisted:
            existing_rule_arn = (describe_res.get("value") or {}).get("Arn")
            progress.success(
                2, "Rule already exists, updating", resource_id=existing_rule_arn
            )

        # 2. Create the scheduled rule.
        progress.report(2, "Creating EventBridge rule")
        put_rule_params: dict[str, Any] = {
            "Name": effective_rule_name,
            "ScheduleExpression": schedule,
            "State": state,
        }
        if description is not None:
            put_rule_params["Description"] = description
        progress.detail("events:PutRule", Name=effective_rule_name)
        rule_res = step(
            tracker,
            "PutRule",
            events_call,
            "PutRule",
            put_rule_params,
        )
        if not rule_res["ok"]:
            progress.failure(2, "Create EventBridge rule", rule_res.get("error"))
            return rule_res
        rule_arn = (rule_res.get("value") or {}).get("RuleArn")
        if not rule_arn:
            tracker.cleanup()
            return err_result(
                SERVICE_ERROR,
                "PutRule returned no RuleArn.",
                key=effective_rule_name,
                guidance=["Retry the command."],
            )
        # Track the rule so a later failure deletes it (LIFO rollback) -- but
        # only when we created it. A pre-existing rule is left untouched on
        # rollback, since we must not delete a resource we did not create.
        if not rule_preexisted:
            tracker.track(
                "events-rule",
                effective_rule_name,
                make_delete_cleanup(
                    events_call, "DeleteRule", {"Name": effective_rule_name}
                ),
            )
        progress.success(2, "Created EventBridge rule", resource_id=rule_arn)

        # 3. Add the Lambda function as a target of the rule.
        progress.report(3, "Adding Lambda target")
        target: dict[str, Any] = {"Id": _TARGET_ID, "Arn": function_arn}
        if input_json is not None:
            target["Input"] = input_json
        progress.detail(
            "events:PutTargets", Rule=effective_rule_name, TargetArn=function_arn
        )
        targets_res = step(
            tracker,
            "PutTargets",
            events_call,
            "PutTargets",
            {"Rule": effective_rule_name, "Targets": [target]},
        )
        if not targets_res["ok"]:
            # step() already rolled back the rule.
            progress.failure(3, "Add Lambda target", targets_res.get("error"))
            return targets_res
        # PutTargets can report a per-entry failure while the call itself
        # succeeds; treat that as a failed step and roll the rule back.
        failed_count = (targets_res.get("value") or {}).get(
            "FailedEntryCount", 0
        )
        if failed_count:
            entries = (targets_res.get("value") or {}).get(
                "FailedEntries"
            ) or []
            reason = entries[0].get("ErrorMessage") if entries else "unknown"
            tracker.cleanup()
            progress.failure(3, "Add Lambda target", reason)
            return err_result(
                SERVICE_ERROR,
                f"PutTargets reported {failed_count} failed entry(ies): "
                f"{reason}",
                key=effective_rule_name,
                guidance=[
                    "Check the target ARN and the rule's limits, then retry."
                ],
            )
        tracker.track(
            "events-target",
            f"{effective_rule_name}/{_TARGET_ID}",
            make_delete_cleanup(
                events_call,
                "RemoveTargets",
                {"Rule": effective_rule_name, "Ids": [_TARGET_ID]},
            ),
        )
        progress.success(3, "Added Lambda target")

        # 4. Allow EventBridge to invoke the function, scoped to the rule ARN.
        progress.report(4, "Granting invoke permission")
        statement_id = f"events-{effective_rule_name}-invoke"
        progress.detail(
            "lambda:AddPermission",
            FunctionName=function_name,
            StatementId=statement_id,
        )
        perm_res = step(
            tracker,
            "AddPermission",
            lambda_call,
            "AddPermission",
            {
                "FunctionName": function_name,
                "StatementId": statement_id,
                "Action": "lambda:InvokeFunction",
                "Principal": "events.amazonaws.com",
                "SourceArn": rule_arn,
            },
        )
        if not perm_res["ok"]:
            # CONFLICT means the statement already exists — that's fine, the
            # function is already invokable by this rule. We did not create it,
            # so it is NOT tracked for rollback.
            if perm_res.get("code") == "CONFLICT":
                progress.success(
                    4, "Invoke permission already exists (reused)"
                )
            else:
                # step() already removed the target and deleted the rule (LIFO).
                progress.failure(
                    4, "Grant invoke permission", perm_res.get("error")
                )
                return perm_res
        else:
            # Only a permission we actually created is tracked, so rollback
            # removes it on a later failure without touching pre-existing ones.
            tracker.track(
                "lambda-permission",
                statement_id,
                make_delete_cleanup(
                    lambda_call,
                    "RemovePermission",
                    {
                        "FunctionName": function_name,
                        "StatementId": statement_id,
                    },
                ),
            )
            progress.success(4, "Granted invoke permission")

        # Success: summarize the created resources.
        return ok_result(
            {
                "RuleName": effective_rule_name,
                "RuleArn": rule_arn,
                "FunctionArn": function_arn,
                "Schedule": schedule,
                "State": state,
                # True when the rule pre-existed (this run was an update/reuse);
                # False when the rule was newly created by this run.
                "Idempotent": rule_preexisted,
                "ResourcesCreated": tracker.summary()["created_resources"],
            },
            key=effective_rule_name,
        )
    except Exception as exc:  # noqa: BLE001 - the command contract is "never raises"
        return err_result(
            SERVICE_ERROR,
            f"create-lambda-schedule internal error: {exc!r}",
            guidance=[
                "Unexpected response shape; retry once, then wire the rule by "
                "hand with events:PutRule / events:PutTargets / "
                "lambda:AddPermission."
            ],
        )


class CreateLambdaScheduleCommand(BasicCommand):
    """``aws events create-lambda-schedule`` command definition."""

    NAME = "create-lambda-schedule"
    DESCRIPTION = (
        "Create an EventBridge scheduled rule wired to invoke a Lambda "
        "function (CDK's events.Rule + LambdaFunction target pattern). Verifies "
        "the function, creates the rule, adds the function as a target, and "
        "grants events.amazonaws.com permission to invoke it. If a step fails, "
        "the resources already created are rolled back automatically."
    )
    EXAMPLES = (
        "To run a Lambda every five minutes::\n\n"
        "  aws events create-lambda-schedule --function-name my-fn "
        "--schedule 'rate(5 minutes)'\n\n"
        "To run it on a cron schedule with a constant JSON input and a custom "
        "rule name::\n\n"
        "  aws events create-lambda-schedule --function-name my-fn "
        "--schedule 'cron(0 12 * * ? *)' --rule-name noon-job "
        "--input '{\"source\":\"schedule\"}'\n"
    )
    ARG_TABLE = [
        {
            "name": "function-name",
            "required": True,
            "cli_type_name": "string",
            "help_text": "Lambda function name or ARN to invoke.",
        },
        {
            "name": "schedule",
            "required": True,
            "cli_type_name": "string",
            "help_text": (
                "Schedule expression, e.g. 'rate(5 minutes)' or "
                "'cron(0 12 * * ? *)'."
            ),
        },
        {
            "name": "rule-name",
            "cli_type_name": "string",
            "help_text": (
                "EventBridge rule name. Derived from the function name "
                "(<name>-schedule) when omitted."
            ),
        },
        {
            "name": "input",
            "cli_type_name": "string",
            "help_text": "Constant JSON input passed to the Lambda on each run.",
        },
        {
            "name": "description",
            "cli_type_name": "string",
            "help_text": "Description for the EventBridge rule.",
        },
        {
            "name": "state",
            "default": "ENABLED",
            "cli_type_name": "string",
            "help_text": "Rule state: ENABLED (default) or DISABLED.",
        },
    ]

    def __init__(
        self,
        session,
        events_client=None,
        lambda_client=None,
        stdout=None,
        stderr=None,
    ):
        """Store the session and injectable clients/streams (for testing)."""
        super().__init__(session)
        self._events_client = events_client
        self._lambda_client = lambda_client
        self._stdout = stdout
        self._stderr = stderr

    def _run_main(self, parsed_args, parsed_globals):
        """Create the clients, run the orchestration, and emit the result."""
        events_client = self._events_client or create_client(
            self._session, "events", parsed_globals
        )
        lambda_client = self._lambda_client or create_client(
            self._session, "lambda", parsed_globals
        )

        def events_call(operation, params):
            return call(events_client, operation, params)

        def lambda_call(operation, params):
            return call(lambda_client, operation, params)

        result = create_lambda_schedule(
            events_call,
            lambda_call,
            function_name=parsed_args.function_name,
            schedule=parsed_args.schedule,
            rule_name=parsed_args.rule_name,
            input_json=parsed_args.input,
            description=parsed_args.description,
            state=parsed_args.state,
            stderr=self._stderr,
        )
        return emit_result(
            self.NAME,
            result,
            parsed_globals,
            self._session,
            self._stdout,
            self._stderr,
        )


def register(event_handlers):
    """Register this command on the EventBridge command table."""
    event_handlers.register(
        "building-command-table.events", _inject_commands
    )


def _inject_commands(command_table, session, **kwargs):
    """Add ``create-lambda-schedule`` to the ``events`` command table."""
    command_table["create-lambda-schedule"] = CreateLambdaScheduleCommand(
        session
    )
