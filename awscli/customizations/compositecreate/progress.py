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
"""Per-step progress reporting for write-side creation commands.

A high-order create command runs several steps; without feedback the CLI appears
to hang while resources are provisioned. :class:`ProgressReporter` writes a
one-line-per-step trace to stderr -- stdout stays reserved for the final JSON
result -- so the caller can see how far a multi-step create got and where it
stopped on failure.
"""

from __future__ import annotations

import sys


class ProgressReporter:
    """Writes per-step progress for a multi-step operation to stderr.

    Progress goes to stderr so stdout carries only the command's JSON result.
    ``total_steps`` is the expected number of steps, used only to label each line
    ``[step_num/total]``. ``stderr`` is injectable for testing; it defaults to
    :data:`sys.stderr`.

    When ``verbose`` is True (the default), :meth:`detail` writes a compact
    ``  → service:Operation key=value`` line under the current step, showing
    which API is being called and the key parameters. This is agent-friendly:
    enough to diagnose failures without flooding the context window.
    """

    def __init__(self, total_steps, stderr=None, verbose=True):
        """Store the step total and the stream progress is written to."""
        self._total = total_steps
        self._stderr = sys.stderr if stderr is None else stderr
        self._verbose = verbose

    def report(self, step_num, description):
        """Announce the start of a step: ``[step_num/total] description...``."""
        self._stderr.write(f"[{step_num}/{self._total}] {description}...\n")

    def detail(self, operation, **key_params):
        """Log which API call is being made with its key parameters.

        Writes ``  → Operation param=value ...`` on one line. Only the most
        identifying parameters should be passed (e.g. the resource name/id),
        not the full request body. Values longer than 60 chars are truncated.
        Silently skipped when ``verbose`` is False.
        """
        if not self._verbose:
            return
        parts = []
        for k, v in key_params.items():
            s = str(v)
            if len(s) > 60:
                s = s[:57] + "..."
            parts.append(f"{k}={s}")
        params_str = " ".join(parts)
        if params_str:
            self._stderr.write(f"  → {operation} {params_str}\n")
        else:
            self._stderr.write(f"  → {operation}\n")

    def success(self, step_num, description, resource_id=None):
        """Mark a step succeeded: ``[step_num/total] ✓ description``.

        If ``resource_id`` is given, appends ``→ id`` to show what was created.
        """
        if resource_id:
            self._stderr.write(
                f"[{step_num}/{self._total}] ✓ {description} → {resource_id}\n"
            )
        else:
            self._stderr.write(
                f"[{step_num}/{self._total}] ✓ {description}\n"
            )

    def failure(self, step_num, description, error):
        """Mark a step failed: ``[step_num/total] ✗ description: error``."""
        self._stderr.write(
            f"[{step_num}/{self._total}] ✗ {description}: {error}\n"
        )
