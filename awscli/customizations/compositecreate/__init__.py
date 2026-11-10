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
"""Shared mechanism for write-side, multi-step creating CLI commands.

Some operational goals need several write API calls in sequence to reach a
usable end state: "create this resource and everything it depends on", "stand up
this stack and wire it together". A single API never does it, and a failure
partway through leaves orphaned, half-built resources behind. This package
provides the common machinery so each such command stays a thin, service-scoped
verb that creates safely and rolls back cleanly on failure:

* :mod:`helpers` -- a botocore ``call`` wrapper that returns errors as data (and,
  unlike the read-only sibling, allows mutating operations), plain-dict result
  factories, a :class:`~.helpers.ResourceTracker` that records created resources
  for LIFO rollback, a :func:`~.helpers.step` primitive that rolls back on a
  failed step, and a JSON emitter that maps a result to a CLI exit code.
* :mod:`progress` -- :class:`~.progress.ProgressReporter`, a per-step stderr
  trace so a multi-step create's progress and failure point are visible.
* :mod:`cleanup` -- generic best-effort cleanup builders (delete, detach) and a
  reporter that rolls a tracker back and prints what it undid.

Nothing here is service-specific; the commands live under their owning service's
customization namespace and build on these primitives.
"""
