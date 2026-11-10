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
"""Generic cleanup/rollback helpers for write-side creation commands.

When a multi-step create fails partway through, the resources already created
must be undone. :class:`~awscli.customizations.compositecreate.helpers.ResourceTracker`
holds the per-resource cleanup callables; this module builds the common ones --
a delete and a detach -- as best-effort, error-swallowing callables, and reports
what a tracker's rollback undid or failed to undo.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def make_delete_cleanup(call_op, operation, params):
    """Build a best-effort cleanup that deletes a created resource.

    Returns a zero-argument callable suitable as :meth:`ResourceTracker.track`'s
    ``cleanup_fn``. It invokes ``call_op(operation, params)`` -- typically a
    delete such as ``DeleteBucket`` -- and ignores the outcome: cleanup is
    best-effort, so any error is logged and swallowed, never raised.
    """
    def _cleanup():
        try:
            logger.debug("Cleanup delete: %s %r", operation, params)
            return call_op(operation, params)
        except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
            logger.warning("Cleanup delete %s failed: %r", operation, exc)
            return None

    return _cleanup


def make_detach_cleanup(call_op, operation, params):
    """Build a best-effort cleanup that detaches a created association.

    Same pattern as :func:`make_delete_cleanup`, but for a detach/disassociate
    operation (e.g. ``DetachRolePolicy``). Returns a zero-argument callable that
    invokes ``call_op(operation, params)`` and ignores any error.
    """
    def _cleanup():
        try:
            logger.debug("Cleanup detach: %s %r", operation, params)
            return call_op(operation, params)
        except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
            logger.warning("Cleanup detach %s failed: %r", operation, exc)
            return None

    return _cleanup


def cleanup_resources(tracker, stderr=None):
    """Roll back a tracker's resources and report the outcome to stderr.

    Runs :meth:`ResourceTracker.cleanup` (which invokes each resource's
    ``cleanup_fn`` in reverse creation order, best-effort) and writes a short
    summary to stderr: the resources being rolled back and how many cleanup calls
    failed. ``stderr`` is injectable for testing; it defaults to
    :data:`sys.stderr`. Returns the number of cleanup failures.
    """
    if stderr is None:
        stderr = sys.stderr
    resources = tracker.resources
    if not resources:
        return 0
    stderr.write(f"Rolling back {len(resources)} created resource(s)...\n")
    # Report in the same reverse (LIFO) order cleanup runs in.
    for resource in reversed(resources):
        stderr.write(
            f"  cleaning up {resource.resource_type} {resource.resource_id}\n"
        )
    failures = tracker.cleanup()
    if failures:
        stderr.write(
            f"Cleanup finished with {failures} failure(s); some resources may "
            "remain. Check the account and remove them manually.\n"
        )
    else:
        stderr.write("Cleanup complete.\n")
    return failures
