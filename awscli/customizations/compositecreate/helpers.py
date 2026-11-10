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
"""Write-side call bridge, result emitter and rollback tracking for create commands.

These commands make several write API calls in sequence and summarize them into
one result. The call bridge returns a result object from every call and never
raises; the CLI, by contrast, returns an exit code and prints model-shaped JSON.
A failure partway through a multi-step create must undo what it already created.
This module bridges all three concerns:

* :func:`ok_result` / :func:`err_result` build a plain ``dict`` with the
  control-flow keys the command logic branches on (``ok``, ``value``, ``code``,
  ...). Only the ``value`` payload (plus ``guidance`` / ``is_exhaustive`` as plain
  keys) is ever emitted, so the output stays normal CLI JSON that ``--query`` and
  ``--output table/text`` understand -- it is not wrapped in a result envelope.
* :func:`call` runs one botocore operation and normalizes a ``ClientError`` into
  the same dict convention. It ALLOWS mutating operations: this is the write-side
  bridge, so there is no read-only guard.
* :class:`ResourceTracker` records each resource a command creates so that, on a
  later failure, :meth:`ResourceTracker.cleanup` can undo them in reverse order,
  and :func:`step` wraps "run this call, and if it fails, roll everything back".
* :func:`emit_result` serializes the result and maps it to a CLI exit code.

Error codes are plain strings (``"NOT_FOUND"`` etc.) that the command logic
branches on directly.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import Any, NamedTuple, TypedDict

from botocore import xform_name
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
)
from botocore.exceptions import (
    ConnectionError as BotoConnectionError,
)

from awscli.customizations.exceptions import ParamValidationError
from awscli.customizations.utils import create_client_from_parsed_globals
from awscli.formatter import get_formatter

logger = logging.getLogger(__name__)

# --- CLI exit codes (see awscli/constants.py) --------------------------------
RC_SUCCESS = 0
RC_OPERATIONAL_ERROR = 255
# ParamValidationError is raised for bad input; the clidriver maps it to rc 252.

# --- Normalized error codes --------------------------------------------------
NOT_FOUND = "NOT_FOUND"
NOT_READY = "NOT_READY"
THROTTLED = "THROTTLED"
PERMISSION_DENIED = "PERMISSION_DENIED"
INVALID_INPUT = "INVALID_INPUT"
TIMEOUT = "TIMEOUT"
CONFLICT = "CONFLICT"
QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
SERVICE_ERROR = "SERVICE_ERROR"
NETWORK_ERROR = "NETWORK_ERROR"
VALIDATION_ERROR = "VALIDATION_ERROR"
CREDENTIALS_EXPIRED = "CREDENTIALS_EXPIRED"

# Botocore error-code -> normalized code for the services these commands touch;
# anything unmapped falls back by HTTP status.
_CLIENT_ERROR_MAP = {
    "AccessDenied": PERMISSION_DENIED,
    "AccessDeniedException": PERMISSION_DENIED,
    "UnauthorizedException": PERMISSION_DENIED,
    "UnauthorizedOperation": PERMISSION_DENIED,
    "NotAuthorized": PERMISSION_DENIED,
    "ResourceNotFoundException": NOT_FOUND,
    "ResourceNotFound": NOT_FOUND,
    "NoSuchEntity": NOT_FOUND,
    "WorkGroupNotFound": NOT_FOUND,
    "DashboardNotFoundError": NOT_FOUND,
    "ThrottlingException": THROTTLED,
    "Throttling": THROTTLED,
    "TooManyRequestsException": THROTTLED,
    "RequestLimitExceeded": THROTTLED,
    "LimitExceededException": QUOTA_EXCEEDED,
    "ServiceQuotaExceededException": QUOTA_EXCEEDED,
    "ServiceUnavailableException": SERVICE_ERROR,
    "InternalServerError": SERVICE_ERROR,
    "InternalFailure": SERVICE_ERROR,
    "ExpiredTokenException": CREDENTIALS_EXPIRED,
    "ExpiredToken": CREDENTIALS_EXPIRED,
    "RequestExpired": CREDENTIALS_EXPIRED,
    "InvalidParameterException": INVALID_INPUT,
    "ValidationException": VALIDATION_ERROR,
    "ValidationError": VALIDATION_ERROR,
    "MalformedQueryException": INVALID_INPUT,
    "InvalidRequestException": INVALID_INPUT,
    "ResourceInUseException": CONFLICT,
    "AlreadyExistsException": CONFLICT,
    "ResourceAlreadyExistsException": CONFLICT,
    "EntityAlreadyExists": CONFLICT,
}
_RETRIABLE = frozenset(
    {THROTTLED, QUOTA_EXCEEDED, NETWORK_ERROR, CREDENTIALS_EXPIRED}
)


class _ResultEnvelopeBase(TypedDict):
    """Required keys present on every result, success or failure."""

    ok: bool
    key: str | None


class ResultEnvelope(_ResultEnvelopeBase, total=False):
    """The plain-``dict`` control-flow envelope built by the result helpers.

    Produced by :func:`ok_result` and :func:`err_result` (and extended by
    :func:`invalid_input`). ``ok`` and ``key`` form the minimum contract and are
    always present; the remaining keys are populated depending on outcome -- the
    ``value``/``is_exhaustive``/``guidance`` payload on success, and the
    ``error``/``code``/``aws_code``/``retriable`` diagnostics on failure. This is
    a typing aid only: the helpers still return ordinary ``dict`` instances, so
    the type is structurally dict-compatible and existing call sites are
    unaffected.
    """

    # Success payload (also carried through on failures as data).
    value: Any
    is_exhaustive: bool | None
    guidance: list[str]
    # Failure diagnostics.
    error: str
    code: str
    aws_code: str | None
    retriable: bool
    # Set by :func:`invalid_input` to request ParamValidationError (rc 252).
    param_error: bool


def ok_result(
    value: Any,
    *,
    is_exhaustive: bool | None = None,
    guidance: list[str] | None = None,
    key: str | None = None,
) -> ResultEnvelope:
    """Build a success result dict."""
    return {
        "ok": True,
        "value": value,
        "is_exhaustive": is_exhaustive,
        "guidance": guidance or [],
        "key": key,
    }


def err_result(
    code: str,
    error: str,
    *,
    value: Any = None,
    guidance: list[str] | None = None,
    retriable: bool = False,
    aws_code: str | None = None,
    key: str | None = None,
    is_exhaustive: bool | None = None,
) -> ResultEnvelope:
    """Build a failure result dict (errors are data, never raised)."""
    return {
        "ok": False,
        "value": value,
        "error": error,
        "code": code,
        "aws_code": aws_code,
        "retriable": retriable or code in _RETRIABLE,
        "guidance": guidance or [],
        "key": key,
        "is_exhaustive": is_exhaustive,
    }


# --- Created-resource tracking and rollback ----------------------------------
# A write-side high-order command creates several resources in sequence. If a
# later step fails, the resources already created must be undone or the caller is
# left with orphans. ``ResourceTracker`` records each created resource with an
# optional undo callable; ``cleanup`` reverses them LIFO; ``step`` ties a single
# call to "roll back everything so far if this fails".


class CreatedResource(NamedTuple):
    """A resource created during a multi-step operation, with its undo hook.

    ``cleanup_fn`` is a zero-argument callable that reverses the creation (e.g.
    deletes the resource); it is ``None`` when the resource needs no cleanup.
    :class:`ResourceTracker` stores these in creation order and, on failure,
    invokes each ``cleanup_fn`` in reverse (LIFO) order.
    """

    resource_type: str
    resource_id: str
    cleanup_fn: Callable[[], Any] | None = None


class ResourceTracker:
    """Records resources created during a multi-step operation for rollback.

    A command creates resources in sequence and tracks each one here. If a later
    step fails, :meth:`cleanup` undoes them in reverse creation order (LIFO),
    which respects dependency order -- a resource created later often depends on
    an earlier one, so it must be removed first. Cleanup is best-effort: a
    failing ``cleanup_fn`` is logged and skipped, never raised, so one failure
    does not strand the rest.
    """

    def __init__(self) -> None:
        """Start with no tracked resources."""
        self._resources: list[CreatedResource] = []

    def track(
        self,
        resource_type: str,
        resource_id: str,
        cleanup_fn: Callable[[], Any] | None = None,
    ) -> CreatedResource:
        """Record a created resource and its optional cleanup callable.

        ``cleanup_fn`` is a zero-argument callable that undoes the creation; pass
        ``None`` for a resource that needs no cleanup. Returns the stored
        :class:`CreatedResource`.
        """
        resource = CreatedResource(resource_type, resource_id, cleanup_fn)
        self._resources.append(resource)
        return resource

    @property
    def resources(self) -> list[CreatedResource]:
        """The tracked resources, in creation order (oldest first)."""
        return list(self._resources)

    def cleanup(self) -> int:
        """Invoke each cleanup callable in reverse creation order (LIFO).

        Best-effort: an exception from any ``cleanup_fn`` is logged and swallowed
        so the remaining resources are still cleaned up. Resources with no
        ``cleanup_fn`` are skipped. Returns the number of cleanup callables that
        raised.
        """
        failures = 0
        for resource in reversed(self._resources):
            if resource.cleanup_fn is None:
                continue
            try:
                resource.cleanup_fn()
            except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
                failures += 1
                logger.warning(
                    "Cleanup failed for %s %r: %r",
                    resource.resource_type,
                    resource.resource_id,
                    exc,
                )
        return failures

    def summary(self) -> dict[str, Any]:
        """Return a serializable dict of created resources, in creation order."""
        return {
            "created_resources": [
                {
                    "resource_type": resource.resource_type,
                    "resource_id": resource.resource_id,
                }
                for resource in self._resources
            ],
            "count": len(self._resources),
        }


def step(
    tracker: ResourceTracker,
    description: str,
    call_fn: Callable[..., dict[str, Any]],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run one step of a multi-step create, rolling back on failure.

    Calls ``call_fn(*args, **kwargs)`` and inspects the result envelope it
    returns. On success (``ok`` is True) the result is returned unchanged. On an
    error result, :meth:`ResourceTracker.cleanup` undoes every resource created
    so far and the error result is returned. This is the core primitive: "do this
    step, and if it fails, roll back everything done so far". ``description``
    labels the step for logging.
    """
    result = call_fn(*args, **kwargs)
    if result.get("ok"):
        return result
    logger.info(
        "Step %r failed (%s); rolling back %d tracked resource(s).",
        description,
        result.get("code"),
        len(tracker.resources),
    )
    tracker.cleanup()
    return result


# --- Input validation predicates ---------------------------------------------
# Each predicate returns a human-readable error description when the check fails,
# or ``None`` when it passes. Callers that fail-fast wrap the description in
# :func:`invalid_input`; callers that accumulate append the non-``None`` results
# to a problem list. Sharing one definition of "non-empty string", "positive
# int", etc. keeps the high-order commands from re-deriving the same predicates
# (and the bool-is-a-subclass-of-int trap) by hand in every validator.


def check_non_empty_str(
    name: str, val: Any, *, required: bool = True
) -> str | None:
    """Reject a blank or non-string value.

    ``None`` passes when ``required`` is False (the argument is optional);
    otherwise it is reported as required. A non-string, an empty string, or a
    whitespace-only string is always rejected.
    """
    if val is None:
        return None if not required else f"{name} is required."
    if not isinstance(val, str):
        return f"{name} must be a string, got {val!r}."
    if val == "":
        return f"{name} is an empty string."
    if not val.strip():
        return f"{name} is empty."
    return None


def check_positive_int(
    name: str, val: Any, *, lo: int = 1, hi: int | None = None
) -> str | None:
    """Reject a non-integer or an integer outside ``[lo, hi]``.

    ``bool`` is rejected explicitly: it is a subclass of ``int``, so ``True``
    would otherwise sneak through as ``1``. ``lo=0`` accepts non-negative
    values; ``hi`` bounds the value from above when provided.
    """
    if isinstance(val, bool) or not isinstance(val, int):
        return f"{name} must be an integer, got {val!r}."
    if hi is not None and not (lo <= val <= hi):
        return f"{name} must be an integer between {lo} and {hi}, got {val!r}."
    if val < lo:
        if lo == 0:
            descriptor = "a non-negative integer"
        elif lo == 1:
            descriptor = "a positive integer"
        else:
            descriptor = f"an integer >= {lo}"
        return f"{name} must be {descriptor}, got {val!r}."
    return None


def check_positive_number(name: str, val: Any) -> str | None:
    """Reject anything that is not an ``int``/``float`` strictly greater than 0.

    ``bool`` is rejected explicitly, for the same reason as
    :func:`check_positive_int`.
    """
    if isinstance(val, bool) or not isinstance(val, (int, float)) or val <= 0:
        return f"{name} must be a positive number, got {val!r}."
    return None


def check_max_size(
    name: str,
    val: str,
    limit: int,
    *,
    unit: str = "bytes",
    measure: str = "utf-8",
) -> str | None:
    """Reject a string whose size exceeds ``limit``.

    ``measure='chars'`` measures with :func:`len`; any other value is treated as
    a text encoding and the size is the encoded byte length (UTF-8 by default).
    ``unit`` only labels the error message.
    """
    size = len(val) if measure == "chars" else len(val.encode(measure))
    if size > limit:
        return f"{name} is {size} {unit}; the limit is {limit} {unit}."
    return None


def invalid_input(
    error: str, guidance: list[str] | None = None, *, value: Any = None
) -> dict[str, Any]:
    """Shorthand for a pre-dispatch INVALID_INPUT failure result.

    Flagged ``param_error`` so :func:`emit_result` raises ``ParamValidationError``
    (exit code 252), matching how the CLI reports bad input. Server-discovered
    invalids use :func:`err_result` directly and stay operational (exit code 255).
    """
    result = err_result(INVALID_INPUT, error, guidance=guidance, value=value)
    result["param_error"] = True
    return result


def create_client(session, service_name, parsed_globals):
    """Create a botocore client honouring --region/--endpoint-url/--verify-ssl.

    Reuses the standard CLI helper so credentials and endpoint resolution match
    every other command.
    """
    return create_client_from_parsed_globals(
        session, service_name, parsed_globals
    )


def _map_client_error(exc: ClientError) -> tuple[str, str | None]:
    aws_code = exc.response.get("Error", {}).get("Code")
    if aws_code in _CLIENT_ERROR_MAP:
        return _CLIENT_ERROR_MAP[aws_code], aws_code
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if status == 404:
        return NOT_FOUND, aws_code
    if status in (401, 403):
        return PERMISSION_DENIED, aws_code
    if status == 409:
        return CONFLICT, aws_code
    if status == 429:
        return THROTTLED, aws_code
    if isinstance(status, int) and status >= 500:
        return SERVICE_ERROR, aws_code
    return SERVICE_ERROR, aws_code


def call(client, operation: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run ONE botocore operation, returning the plain-dict result convention.

    ``operation`` is the modeled PascalCase name (e.g. ``"CreateBucket"``); it is
    translated to the botocore method name. Mutating operations are ALLOWED --
    this is the write-side bridge, so there is no read-only guard. Errors are
    returned as data, never raised, so the caller can branch on ``result["code"]``
    and roll back resources created by earlier steps.
    """
    method_name = xform_name(operation)
    method = getattr(client, method_name)
    try:
        response = method(**params)
    except ClientError as exc:
        code, aws_code = _map_client_error(exc)
        return err_result(
            code, str(exc), aws_code=aws_code, retriable=code in _RETRIABLE
        )
    except NoCredentialsError as exc:
        return err_result(CREDENTIALS_EXPIRED, str(exc), retriable=True)
    except (EndpointConnectionError, BotoConnectionError) as exc:
        return err_result(NETWORK_ERROR, str(exc), retriable=True)
    except BotoCoreError as exc:
        return err_result(SERVICE_ERROR, str(exc))
    return ok_result(response)


def _render_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Project the user-facing JSON from a result dict.

    Emits the primary ``value`` payload with ``guidance``/``is_exhaustive``
    folded in as plain keys -- not a result envelope. On a failure the
    error/code are included so the caller can branch on them.
    """
    value = result.get("value")
    if isinstance(value, dict):
        payload = dict(value)
    elif value is None:
        payload = {}
    else:
        payload = {"result": value}
    if result.get("is_exhaustive") is not None:
        payload.setdefault("is_exhaustive", result["is_exhaustive"])
    if result.get("guidance"):
        payload.setdefault("guidance", result["guidance"])
    if not result.get("ok"):
        payload.setdefault("error", result.get("error"))
        payload.setdefault("code", result.get("code"))
        if result.get("retriable"):
            payload.setdefault("retriable", True)
    return payload


def emit_result(
    name,
    result: dict[str, Any],
    parsed_globals,
    session,
    stdout=None,
    stderr=None,
) -> int:
    """Serialize ``result`` as CLI JSON and return the exit code.

    Success -> payload to stdout, rc 0. Operational failure -> payload (if any)
    to stdout so ``--query`` still works, a concise ``code: error`` line to
    stderr, rc 255.
    """
    if stdout is None:
        stdout = sys.stdout
    if stderr is None:
        stderr = sys.stderr
    # Pre-dispatch bad input is reported the CLI-native way: ParamValidationError
    # (exit code 252), no JSON payload.
    if not result.get("ok") and result.get("param_error"):
        raise ParamValidationError(result.get("error") or "Invalid input")
    output = getattr(parsed_globals, "output", None)
    if output is None:
        output = session.get_config_variable("output")
    formatter = get_formatter(output, parsed_globals)
    payload = _render_payload(result)
    formatter(name, payload, stream=stdout)
    stdout.write("\n")
    if result.get("ok"):
        return RC_SUCCESS
    stderr.write(
        f"{result.get('code', SERVICE_ERROR)}: {result.get('error', '')}\n"
    )
    return RC_OPERATIONAL_ERROR


def raise_invalid(message: str) -> None:
    """Raise a ParamValidationError (rc 252) for pre-dispatch bad input."""
    raise ParamValidationError(message)
