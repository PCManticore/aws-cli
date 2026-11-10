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
"""``aws apigateway create-lambda-api``: build a Lambda-backed REST API in one call.

Replicates CDK's ``LambdaRestApi`` L2 construct. A single verb verifies the
target Lambda function, creates a REST API with a greedy ``{proxy+}`` resource
(and a root ``/`` method) wired to the function through an ``AWS_PROXY``
integration, grants API Gateway permission to invoke the function, then deploys
the API to a stage and returns its invoke URL.

The command drives the strict create-order chain below; every step runs through
the shared write-side call bridge (errors are returned as data, never raised),
and every created resource is tracked so a failure partway through rolls back
LIFO -- deleting the REST API cascades to its resources, methods, integrations,
deployment and stage, and the Lambda permission is removed separately:

1. ``lambda:GetFunction``        -- verify the function, read its ARN
2. ``apigateway:CreateRestApi``  -- create the API (restApiId, rootResourceId)
3. ``apigateway:CreateResource`` -- create ``{proxy+}`` under the root
4. ``apigateway:PutMethod``      -- ``ANY`` on ``{proxy+}`` (no auth)
5. ``apigateway:PutIntegration`` -- ``AWS_PROXY`` on ``{proxy+}``
6. ``apigateway:PutMethod``      -- ``ANY`` on the root ``/``
7. ``apigateway:PutIntegration`` -- ``AWS_PROXY`` on the root
8. ``lambda:AddPermission``      -- allow apigateway to invoke the function
9. ``apigateway:CreateDeployment``
10. ``apigateway:CreateStage``   -- deploy to a stage, return the invoke URL
"""

from __future__ import annotations

import logging

from awscli.customizations.commands import BasicCommand
from awscli.customizations.compositecreate.cleanup import make_delete_cleanup
from awscli.customizations.compositecreate.helpers import (
    call,
    check_non_empty_str,
    create_client,
    emit_result,
    invalid_input,
    ok_result,
    step,
    ResourceTracker,
)
from awscli.customizations.compositecreate.progress import ProgressReporter

logger = logging.getLogger(__name__)

# Number of create-order steps, used only to label progress lines.
_TOTAL_STEPS = 10

# API Gateway's Lambda invoke URI and the service-linked invoke principal.
# The partition (``aws``, ``aws-cn``, ``aws-us-gov``, …) is derived at
# runtime from the function ARN so this works in every AWS partition.
_INVOKE_URI_FMT = (
    "arn:{partition}:apigateway:{region}:lambda:path/2015-03-31/functions/"
    "{function_arn}/invocations"
)
_APIGATEWAY_PRINCIPAL = "apigateway.amazonaws.com"
# Lambda AWS_PROXY integrations always call Lambda over POST, regardless of the
# HTTP method the client used against the API.
_INTEGRATION_HTTP_METHOD = "POST"


def _short_function_name(function_name):
    """Return the bare function name from a name or a full function ARN.

    ``arn:aws:lambda:<region>:<acct>:function:<name>[:<qualifier>]`` yields
    ``<name>``; a plain name is returned unchanged. Used to derive a default
    REST API name.
    """
    if function_name.startswith("arn:") and ":function:" in function_name:
        tail = function_name.split(":function:", 1)[1]
        return tail.split(":", 1)[0]
    return function_name


def _account_from_arn(function_arn):
    """Return the account id embedded in a Lambda function ARN, or ``None``.

    The ARN layout is ``arn:<partition>:lambda:<region>:<account>:function:…``;
    the account is the 5th colon-delimited field. Returns ``None`` when the ARN
    is too short to carry one.
    """
    parts = function_arn.split(":")
    if len(parts) >= 5 and parts[4]:
        return parts[4]
    return None


def _partition_from_arn(arn):
    """Return the partition from an ARN (``aws``, ``aws-cn``, ``aws-us-gov``).

    Falls back to ``"aws"`` when the ARN is too short.
    """
    parts = arn.split(":")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return "aws"


def _dns_suffix_for_partition(partition):
    """Return the DNS suffix for the given AWS partition.

    ``aws`` → ``amazonaws.com``, ``aws-cn`` → ``amazonaws.com.cn``,
    everything else → ``amazonaws.com``.
    """
    if partition == "aws-cn":
        return "amazonaws.com.cn"
    return "amazonaws.com"


def create_lambda_api(
    apigw_call,
    lambda_call,
    function_name,
    region,
    api_name=None,
    stage_name="prod",
    description=None,
    stderr=None,
):
    """Create a Lambda-backed REST API end to end, rolling back on failure.

    ``apigw_call`` and ``lambda_call`` are the shared write-side call bridges
    bound to the apigateway and lambda clients respectively. Runs the 10-step
    create chain in strict dependency order; on any failure the resources
    created so far are undone LIFO and the error result is returned. On success
    returns an :func:`ok_result` whose value carries the REST API id, root
    resource id, stage name, invoke URL, function ARN and the created-resource
    list. Never raises -- failures are data.

    The chain is idempotent: a REST API, ``{proxy+}`` resource or stage of the
    same name that already exists is reused rather than duplicated (PutMethod,
    PutIntegration and AddPermission are themselves safe to repeat), so running
    the command twice with the same arguments succeeds both times. Only
    resources this invocation actually creates are tracked for rollback, so a
    failure never deletes a pre-existing API, permission or stage.
    """
    tracker = ResourceTracker()
    progress = ProgressReporter(total_steps=_TOTAL_STEPS, stderr=stderr)

    # Step 1: verify the Lambda function exists and read its ARN.
    progress.report(1, "Verifying Lambda function")
    progress.detail("lambda:GetFunction", FunctionName=function_name)
    result = step(
        tracker,
        "Verify Lambda function",
        lambda_call,
        "GetFunction",
        {"FunctionName": function_name},
    )
    if not result["ok"]:
        return result
    configuration = result["value"].get("Configuration") or {}
    function_arn = configuration.get("FunctionArn")
    if not function_arn:
        return invalid_input(
            "GetFunction did not return a FunctionArn for "
            f"{function_name!r}; cannot wire the integration."
        )
    account_id = _account_from_arn(function_arn)
    partition = _partition_from_arn(function_arn)
    dns_suffix = _dns_suffix_for_partition(partition)
    progress.success(1, "Lambda function verified", resource_id=function_arn)

    # Derive a sensible default API name from the function name.
    if not api_name:
        api_name = "{0}-api".format(_short_function_name(function_name))

    # Step 2: find or create the REST API.
    # CreateRestApi is NOT idempotent -- it mints a brand-new API on every call
    # -- so probe GetRestApis for an API with the same name first and reuse it
    # when present. A pre-existing API is NOT tracked for rollback: a later
    # failure must not delete an API this invocation did not create.
    progress.report(2, "Creating REST API")
    create_api_params = {"name": api_name}
    if description:
        create_api_params["description"] = description
    rest_api_id = None
    root_resource_id = None
    pre_existing = False
    progress.detail("apigateway:GetRestApis")
    existing = apigw_call("GetRestApis", {})
    if existing["ok"]:
        for item in existing["value"].get("items") or []:
            if item.get("name") == api_name:
                rest_api_id = item["id"]
                root_resource_id = item.get("rootResourceId")
                pre_existing = True
                break
    if pre_existing:
        progress.success(2, "REST API already exists", resource_id=rest_api_id)
    else:
        progress.detail("apigateway:CreateRestApi", name=api_name)
        result = step(
            tracker,
            "Create REST API",
            apigw_call,
            "CreateRestApi",
            create_api_params,
        )
        if not result["ok"]:
            return result
        rest_api_id = result["value"]["id"]
        # Track the API first so its DeleteRestApi cleanup runs LAST (LIFO),
        # cascading to every resource, method, integration, deployment and stage.
        tracker.track(
            "rest-api",
            rest_api_id,
            make_delete_cleanup(
                apigw_call, "DeleteRestApi", {"restApiId": rest_api_id}
            ),
        )
        root_resource_id = result["value"].get("rootResourceId")
        progress.success(2, "Created REST API", resource_id=rest_api_id)

    # CreateRestApi normally returns rootResourceId; fall back to GetResources
    # if an older endpoint omits it so the chain can still proceed.
    if not root_resource_id:
        progress.detail("apigateway:GetResources", restApiId=rest_api_id)
        lookup = step(
            tracker,
            "Look up root resource",
            apigw_call,
            "GetResources",
            {"restApiId": rest_api_id},
        )
        if not lookup["ok"]:
            return lookup
        for item in lookup["value"].get("items") or []:
            if item.get("path") == "/":
                root_resource_id = item.get("id")
                break
    if not root_resource_id:
        tracker.cleanup()
        return invalid_input(
            "Could not determine the root resource id for REST API "
            f"{rest_api_id!r}."
        )

    # Step 3: find or create the greedy {proxy+} resource under the root.
    # Only a reused API can already carry a {proxy+} resource, so the
    # GetResources probe is skipped on a freshly created API.
    progress.report(3, "Creating {proxy+} resource")
    proxy_resource_id = None
    proxy_pre_existing = False
    if pre_existing:
        progress.detail("apigateway:GetResources", restApiId=rest_api_id)
        proxy_lookup = apigw_call("GetResources", {"restApiId": rest_api_id})
        if proxy_lookup["ok"]:
            for item in proxy_lookup["value"].get("items") or []:
                if item.get("pathPart") == "{proxy+}":
                    proxy_resource_id = item["id"]
                    proxy_pre_existing = True
                    break
    if proxy_pre_existing:
        progress.success(
            3, "{proxy+} resource already exists", resource_id=proxy_resource_id
        )
    else:
        progress.detail(
            "apigateway:CreateResource",
            restApiId=rest_api_id,
            pathPart="{proxy+}",
        )
        result = step(
            tracker,
            "Create {proxy+} resource",
            apigw_call,
            "CreateResource",
            {
                "restApiId": rest_api_id,
                "parentId": root_resource_id,
                "pathPart": "{proxy+}",
            },
        )
        if not result["ok"]:
            return result
        proxy_resource_id = result["value"]["id"]
        # Cascade-deleted with the API, so no standalone cleanup is needed.
        tracker.track("resource", "{{proxy+}} ({0})".format(proxy_resource_id))
        progress.success(
            3, "Created {proxy+} resource", resource_id=proxy_resource_id
        )

    invoke_uri = _INVOKE_URI_FMT.format(
        partition=partition, region=region, function_arn=function_arn
    )

    # Steps 4-7: wire ANY -> AWS_PROXY on both the proxy and the root resource.
    # PutMethod throws CONFLICT if the method already exists, so we call the
    # bridge directly (not step()) and treat CONFLICT as success for idempotency.
    # PutIntegration is safe to repeat but can also throw CONFLICT.
    for step_num, label, resource_id in (
        (4, "{proxy+}", proxy_resource_id),
        (6, "root", root_resource_id),
    ):
        progress.report(step_num, "Adding ANY method on {0}".format(label))
        progress.detail(
            "apigateway:PutMethod", resourceId=resource_id, httpMethod="ANY"
        )
        result = apigw_call(
            "PutMethod",
            {
                "restApiId": rest_api_id,
                "resourceId": resource_id,
                "httpMethod": "ANY",
                "authorizationType": "NONE",
            },
        )
        if not result["ok"]:
            if result.get("code") == "CONFLICT":
                progress.success(
                    step_num,
                    "ANY method already exists on {0}".format(label),
                )
            else:
                tracker.cleanup()
                return result
        else:
            tracker.track("method", "ANY {0}".format(label))
            progress.success(step_num, "Added ANY method on {0}".format(label))

        integration_step = step_num + 1
        progress.report(
            integration_step, "Adding integration on {0}".format(label)
        )
        progress.detail(
            "apigateway:PutIntegration",
            resourceId=resource_id,
            type="AWS_PROXY",
        )
        result = apigw_call(
            "PutIntegration",
            {
                "restApiId": rest_api_id,
                "resourceId": resource_id,
                "httpMethod": "ANY",
                "type": "AWS_PROXY",
                "integrationHttpMethod": _INTEGRATION_HTTP_METHOD,
                "uri": invoke_uri,
            },
        )
        if not result["ok"]:
            if result.get("code") == "CONFLICT":
                progress.success(
                    integration_step,
                    "Integration already exists on {0}".format(label),
                )
            else:
                tracker.cleanup()
                return result
        else:
            tracker.track("integration", "AWS_PROXY {0}".format(label))
        progress.success(
            integration_step, "Added integration on {0}".format(label)
        )

    # Step 8: grant API Gateway permission to invoke the function.
    progress.report(8, "Granting invoke permission")
    statement_id = "apigateway-{0}-invoke".format(rest_api_id)
    source_arn = "arn:{partition}:execute-api:{region}:{account}:{api_id}/*".format(
        partition=partition,
        region=region, account=account_id or "*", api_id=rest_api_id
    )
    progress.detail(
        "lambda:AddPermission",
        FunctionName=function_name,
        StatementId=statement_id,
    )
    # AddPermission is called directly rather than through step(): a CONFLICT
    # means the statement already exists (an expected idempotent outcome) and
    # must NOT trigger a rollback of resources created earlier. A pre-existing
    # permission is also left untracked so a later failure does not revoke
    # access this invocation did not grant.
    result = lambda_call(
        "AddPermission",
        {
            "FunctionName": function_name,
            "StatementId": statement_id,
            "Action": "lambda:InvokeFunction",
            "Principal": _APIGATEWAY_PRINCIPAL,
            "SourceArn": source_arn,
        },
    )
    permission_pre_existing = False
    if not result["ok"]:
        if result.get("code") == "CONFLICT":
            permission_pre_existing = True
            progress.success(8, "Invoke permission already exists")
        else:
            tracker.cleanup()
            return result
    else:
        progress.success(8, "Granted invoke permission")
    if not permission_pre_existing:
        tracker.track(
            "lambda-permission",
            statement_id,
            make_delete_cleanup(
                lambda_call,
                "RemovePermission",
                {"FunctionName": function_name, "StatementId": statement_id},
            ),
        )

    # Step 9: create a deployment.
    progress.report(9, "Creating deployment")
    progress.detail("apigateway:CreateDeployment", restApiId=rest_api_id)
    result = step(
        tracker,
        "Create deployment",
        apigw_call,
        "CreateDeployment",
        {"restApiId": rest_api_id},
    )
    if not result["ok"]:
        return result
    deployment_id = result["value"]["id"]
    tracker.track("deployment", deployment_id)
    progress.success(9, "Created deployment", resource_id=deployment_id)

    # Step 10: create the stage, or repoint an existing one at the new
    # deployment. CreateStage fails when the stage already exists, so probe with
    # GetStage first; when present, UpdateStage swaps in the deployment created
    # above (both CreateDeployment and this update are safe to repeat). A
    # pre-existing stage is not tracked for rollback.
    progress.report(10, "Creating stage {0}".format(stage_name))
    progress.detail(
        "apigateway:GetStage",
        restApiId=rest_api_id,
        stageName=stage_name,
    )
    stage_check = apigw_call(
        "GetStage", {"restApiId": rest_api_id, "stageName": stage_name}
    )
    stage_pre_existing = stage_check["ok"]
    if stage_pre_existing:
        progress.detail(
            "apigateway:UpdateStage",
            restApiId=rest_api_id,
            stageName=stage_name,
        )
        result = step(
            tracker,
            "Update stage deployment",
            apigw_call,
            "UpdateStage",
            {
                "restApiId": rest_api_id,
                "stageName": stage_name,
                "patchOperations": [
                    {
                        "op": "replace",
                        "path": "/deploymentId",
                        "value": deployment_id,
                    }
                ],
            },
        )
        if not result["ok"]:
            return result
        progress.success(10, "Updated existing stage", resource_id=stage_name)
    else:
        progress.detail(
            "apigateway:CreateStage",
            restApiId=rest_api_id,
            stageName=stage_name,
        )
        result = step(
            tracker,
            "Create stage",
            apigw_call,
            "CreateStage",
            {
                "restApiId": rest_api_id,
                "stageName": stage_name,
                "deploymentId": deployment_id,
            },
        )
        if not result["ok"]:
            return result
        tracker.track("stage", stage_name)
        progress.success(10, "Created stage", resource_id=stage_name)

    invoke_url = "https://{api_id}.execute-api.{region}.{dns_suffix}/{stage}".format(
        api_id=rest_api_id, region=region, dns_suffix=dns_suffix,
        stage=stage_name
    )

    return ok_result(
        {
            "RestApiId": rest_api_id,
            "RootResourceId": root_resource_id,
            "StageName": stage_name,
            "InvokeUrl": invoke_url,
            "FunctionArn": function_arn,
            "Authorization": "NONE — endpoint is publicly accessible",
            "Idempotent": pre_existing,
            "ResourcesCreated": tracker.summary()["created_resources"],
        }
    )


class CreateLambdaApiCommand(BasicCommand):
    NAME = "create-lambda-api"
    DESCRIPTION = (
        "Create a REST API fully wired to a Lambda function in one call "
        "(CDK LambdaRestApi equivalent): a greedy {proxy+} resource and a "
        "root method with AWS_PROXY integrations, the invoke permission, a "
        "deployment and a stage. Rolls back every created resource on failure.\n\n"
        "**Note:** The created endpoint is publicly accessible with no "
        "authorization (authorizationType=NONE), matching CDK LambdaRestApi "
        "defaults. Add an authorizer (IAM, Cognito, or Lambda) after creation "
        "if the endpoint should be protected."
    )
    EXAMPLES = (
        "To create and deploy a Lambda-backed REST API::\n\n"
        "  aws apigateway create-lambda-api --function-name my-function\n\n"
        "To name the API and stage explicitly::\n\n"
        "  aws apigateway create-lambda-api --function-name my-function "
        "--api-name my-api --stage-name dev --description 'My service API'\n"
    )
    ARG_TABLE = [
        {
            "name": "function-name",
            "required": True,
            "cli_type_name": "string",
            "help_text": "Name or ARN of the Lambda function to wire the API to.",
        },
        {
            "name": "api-name",
            "cli_type_name": "string",
            "help_text": (
                "Name for the REST API. Default: the function name with an "
                "'-api' suffix."
            ),
        },
        {
            "name": "stage-name",
            "default": "prod",
            "cli_type_name": "string",
            "help_text": "Stage to deploy the API to (default: prod).",
        },
        {
            "name": "description",
            "cli_type_name": "string",
            "help_text": "Optional description for the REST API.",
        },
    ]

    def __init__(
        self,
        session,
        client=None,
        lambda_client=None,
        stdout=None,
        stderr=None,
    ):
        super().__init__(session)
        self._provided_client = client
        self._provided_lambda_client = lambda_client
        self._stdout = stdout
        self._stderr = stderr

    def _run_main(self, parsed_args, parsed_globals):
        apigw_client = self._provided_client or create_client(
            self._session, "apigateway", parsed_globals
        )
        lambda_client = self._provided_lambda_client or create_client(
            self._session, "lambda", parsed_globals
        )

        def apigw_call(operation, params):
            return call(apigw_client, operation, params)

        def lambda_call(operation, params):
            return call(lambda_client, operation, params)

        # Validate pre-dispatch input before any API call.
        problems = []
        name_err = check_non_empty_str(
            "function-name", parsed_args.function_name, required=True
        )
        if name_err is not None:
            problems.append(name_err)
        stage_err = check_non_empty_str(
            "stage-name", parsed_args.stage_name, required=True
        )
        if stage_err is not None:
            problems.append(stage_err)
        if problems:
            result = invalid_input("; ".join(problems), guidance=problems)
        else:
            region = apigw_client.meta.region_name
            result = create_lambda_api(
                apigw_call,
                lambda_call,
                parsed_args.function_name,
                region,
                api_name=parsed_args.api_name,
                stage_name=parsed_args.stage_name,
                description=parsed_args.description,
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
    """Inject the ``create-lambda-api`` verb into the apigateway command table."""
    event_handlers.register(
        "building-command-table.apigateway", _inject_commands
    )


def _inject_commands(command_table, session, **kwargs):
    """Register the command on the apigateway command table."""
    command_table["create-lambda-api"] = CreateLambdaApiCommand(session)
