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
"""``aws ecs create-fargate-web-service`` -- stand up an ALB-fronted Fargate service.

This command replicates CDK's ``ApplicationLoadBalancedFargateService`` L3
construct as a single verb: it provisions a complete, internet-facing ECS
Fargate web service -- IAM roles, a CloudWatch log group, a task definition, two
security groups, an Application Load Balancer, a target group, a listener and
the service itself -- in one strict dependency-ordered sequence.

A single API never does this, and a failure partway through leaves orphaned,
half-built resources behind. The command routes every write through the shared
:mod:`awscli.customizations.compositecreate.helpers` call bridge (errors are
returned as data, never raised), records each created resource in a
:class:`~awscli.customizations.compositecreate.helpers.ResourceTracker`, and
rolls everything back LIFO if any step fails. Progress for the multi-step create
is written to stderr so stdout carries only the final JSON result.

Resources are created in this order (dependencies first)::

    DescribeClusters (verify)        -> cluster must be ACTIVE
    CreateRole (execution)           -> executionRoleArn
    AttachRolePolicy                 -> AmazonECSTaskExecutionRolePolicy
    CreateRole (task)                -> taskRoleArn
    CreateLogGroup                   -> logGroupName
    RegisterTaskDefinition           -> taskDefinitionArn
    CreateSecurityGroup (ALB)        -> albSgId
    AuthorizeSecurityGroupIngress    -> 0.0.0.0/0:80 on the ALB SG
    CreateSecurityGroup (service)    -> serviceSgId
    AuthorizeSecurityGroupIngress    -> ALB SG -> service SG on the container port
    CreateLoadBalancer               -> lbArn, dnsName
    CreateTargetGroup (ip)           -> tgArn
    CreateListener (:80 -> TG)       -> listenerArn
    CreateService (FARGATE)          -> serviceArn

Rollback (LIFO) is handled automatically by the tracker: the service is scaled
to zero and deleted first, then the listener, target group, load balancer,
security groups, task definition, log group, task role, the policy detach, and
finally the execution role.
"""

from __future__ import annotations

import json
import logging
import re

from awscli.customizations.commands import BasicCommand
from awscli.customizations.compositecreate.cleanup import (
    make_delete_cleanup,
    make_detach_cleanup,
)
from awscli.customizations.compositecreate.helpers import (
    CONFLICT,
    NOT_FOUND,
    ResourceTracker,
    call,
    check_non_empty_str,
    check_positive_int,
    create_client,
    emit_result,
    err_result,
    invalid_input,
    ok_result,
    step,
)
from awscli.customizations.compositecreate.progress import ProgressReporter

logger = logging.getLogger(__name__)

# The services whose clients this command needs, created lazily on first use.
_SERVICES = ("ecs", "iam", "ec2", "elbv2", "logs")

# The managed policy that grants a Fargate task's execution role permission to
# pull the container image and write logs. The ``{partition}`` placeholder is
# filled at runtime so this works in every AWS partition.
_EXECUTION_ROLE_POLICY_ARN_FMT = (
    "arn:{partition}:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
)

# The trust policy shared by both the execution and task roles: ECS tasks assume
# them on the service's behalf.
_ECS_TASKS_TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "ecs-tasks.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}

# Elastic Load Balancing names are capped at 32 chars and limited to
# alphanumerics and hyphens, with no leading/trailing hyphen.
_ELB_NAME_MAX = 32

# Total number of reported steps (matches the API call sequence above).
_TOTAL_STEPS = 14

# EC2 signals a name collision on CreateSecurityGroup with this error code. The
# generic error map in helpers does not fold it into CONFLICT (it arrives as a
# plain HTTP 400), so the create path matches it explicitly to reuse an existing
# group left behind by a prior partial run.
_DUPLICATE_SG_AWS_CODE = "InvalidGroup.Duplicate"


def _is_already_exists(result, *aws_codes):
    """Return True when a failed call means the resource already exists.

    Treats the normalized ``CONFLICT`` code (which covers IAM
    ``EntityAlreadyExists`` and CloudWatch Logs
    ``ResourceAlreadyExistsException``) as already-exists, plus any botocore
    error code passed explicitly for services whose collision error the generic
    map does not classify as CONFLICT (e.g. EC2 ``InvalidGroup.Duplicate``).
    """
    if result.get("code") == CONFLICT:
        return True
    return result.get("aws_code") in aws_codes


def _derive_service_name(image):
    """Derive an ECS service name from a container image reference.

    Takes the repository component of ``image`` -- the part after the last
    ``/`` -- and strips any ``:tag`` or ``@digest`` suffix, then sanitizes it to
    the characters ECS accepts. For example
    ``123.dkr.ecr.us-east-1.amazonaws.com/my-app:latest`` yields ``my-app`` and
    a bare ``nginx`` yields ``nginx``. Falls back to ``fargate-web-service`` when
    nothing usable remains.
    """
    repo = image.rsplit("/", 1)[-1]
    # Drop a @sha256:... digest first, then a :tag.
    repo = repo.split("@", 1)[0]
    repo = repo.split(":", 1)[0]
    cleaned = re.sub(r"[^a-zA-Z0-9-]", "-", repo).strip("-")
    return cleaned or "fargate-web-service"


def _elb_name(service_name, suffix):
    """Build an ELB/target-group name of at most 32 valid characters.

    Sanitizes ``service_name`` to alphanumerics and hyphens, truncates it to
    leave room for ``suffix``, and trims any hyphen that would otherwise land at
    the boundary so the result never begins or ends with one.
    """
    base = re.sub(r"[^a-zA-Z0-9-]", "-", service_name).strip("-") or "svc"
    max_base = _ELB_NAME_MAX - len(suffix)
    base = base[:max_base].rstrip("-") or "svc"
    return "{0}{1}".format(base, suffix)


def _validate(parsed_args):
    """Validate CLI input before any API call, returning problems as data.

    Returns an :func:`invalid_input` result (which maps to ParamValidationError,
    exit code 252) describing every problem found, or ``None`` when the input is
    valid. Also returns the parsed, non-empty subnet-id list so the caller does
    not re-split it.
    """
    problems = []
    for name, value in (
        ("--cluster", parsed_args.cluster),
        ("--image", parsed_args.image),
        ("--vpc-id", parsed_args.vpc_id),
        ("--subnet-ids", parsed_args.subnet_ids),
        ("--cpu", parsed_args.cpu),
        ("--memory", parsed_args.memory),
    ):
        problem = check_non_empty_str(name, value)
        if problem is not None:
            problems.append(problem)

    subnet_ids = []
    if isinstance(parsed_args.subnet_ids, str):
        subnet_ids = [s.strip() for s in parsed_args.subnet_ids.split(",")]
        subnet_ids = [s for s in subnet_ids if s]
        if not subnet_ids:
            problems.append("--subnet-ids must list at least one subnet id.")

    port_problem = check_positive_int(
        "--container-port", parsed_args.container_port, lo=1, hi=65535
    )
    if port_problem is not None:
        problems.append(port_problem)

    count_problem = check_positive_int(
        "--desired-count", parsed_args.desired_count, lo=0
    )
    if count_problem is not None:
        problems.append(count_problem)

    if parsed_args.assign_public_ip not in ("ENABLED", "DISABLED"):
        problems.append(
            "--assign-public-ip must be ENABLED or DISABLED, got {0!r}.".format(
                parsed_args.assign_public_ip
            )
        )

    if problems:
        return invalid_input("; ".join(problems)), subnet_ids
    return None, subnet_ids


def _lookup_security_group_id(ec2_call, group_name, vpc_id):
    """Return the id of the security group named ``group_name`` in ``vpc_id``.

    Used by the create path to resolve a security group that already exists
    (EC2 ``InvalidGroup.Duplicate``) from a prior partial run so it can be
    reused instead of created. Returns ``None`` when the lookup call fails or
    finds no matching group.
    """
    res = ec2_call(
        "DescribeSecurityGroups",
        {
            "Filters": [
                {"Name": "group-name", "Values": [group_name]},
                {"Name": "vpc-id", "Values": [vpc_id]},
            ]
        },
    )
    if not res["ok"]:
        return None
    groups = res["value"].get("SecurityGroups", [])
    return groups[0]["GroupId"] if groups else None


def _describe_existing_service(call_ops, cluster, service_name, progress):
    """Describe an existing ACTIVE ECS service by name, or return ``None``.

    Idempotency fast path. The ECS service is the last resource the create flow
    provisions and it references the whole stack -- task definition, target
    group, load balancer, security groups and (via the task definition) the IAM
    roles -- so if a service with this name already exists and is ACTIVE, a prior
    run with the same arguments already built everything. This gathers that
    stack's current state with read-only Describe calls across ECS and ELBv2 and
    returns it as an ok result in the same shape as a fresh create (with
    ``Idempotent: True`` added), so the command is safe to run twice with the
    same arguments without building a duplicate service. Returns:

    * ``None`` -- no matching ACTIVE service exists (or the initial lookup itself
      failed); the caller proceeds with the normal create flow.
    * an ok result -- a matching ACTIVE service was found and described.
    * an error result -- the service was found but a follow-up Describe failed;
      the caller surfaces the error rather than risking a duplicate create.

    Progress is reported on ``progress`` so the fast path shows what it found.
    """
    ecs = call_ops["ecs"]
    elbv2 = call_ops["elbv2"]
    # The create path derives the log group name from the service name the same
    # way, so recompute it rather than reading it back from the task definition.
    log_group_name = "/ecs/{0}".format(service_name)

    # Step 1: look the service up in the cluster. A failed lookup is treated as
    # "not found" so a transient read error falls back to the create path rather
    # than blocking the command.
    label = "Checking for existing service {0}".format(service_name)
    progress.report(1, label)
    progress.detail(
        "ecs:DescribeServices", cluster=cluster, service=service_name
    )
    check = ecs(
        "DescribeServices",
        {"cluster": cluster, "services": [service_name]},
    )
    if not check["ok"]:
        progress.success(1, "No existing service found; creating a new one")
        return None
    active = [
        s
        for s in check["value"].get("services", [])
        if s.get("status") == "ACTIVE"
    ]
    if not active:
        progress.success(1, "No existing service found; creating a new one")
        return None
    service = active[0]
    service_arn = service.get("serviceArn", "")
    task_definition_arn = service.get("taskDefinition", "")
    progress.success(1, label, service_arn)

    # The service SG is the one attached to the tasks; the ALB SG lives on the
    # load balancer and is read in step 4.
    awsvpc = service.get("networkConfiguration", {}).get(
        "awsvpcConfiguration", {}
    )
    service_sgs = awsvpc.get("securityGroups", [])
    service_sg_id = service_sgs[0] if service_sgs else None

    # Target group arn comes straight off the service's load balancer wiring.
    load_balancers = service.get("loadBalancers", [])
    target_group_arn = (
        load_balancers[0].get("targetGroupArn") if load_balancers else None
    )

    # Step 2: task definition -> execution/task role ARNs.
    progress.report(2, "Describing task definition")
    execution_role_arn = None
    task_role_arn = None
    if task_definition_arn:
        progress.detail(
            "ecs:DescribeTaskDefinition", taskDefinition=task_definition_arn
        )
        td_res = ecs(
            "DescribeTaskDefinition",
            {"taskDefinition": task_definition_arn},
        )
        if not td_res["ok"]:
            progress.failure(
                2, "Describing task definition", td_res.get("error")
            )
            return td_res
        task_def = td_res["value"].get("taskDefinition", {})
        execution_role_arn = task_def.get("executionRoleArn")
        task_role_arn = task_def.get("taskRoleArn")
    progress.success(2, "Described task definition")

    # Step 3: target group -> load balancer ARN.
    progress.report(3, "Describing target group")
    load_balancer_arn = None
    if target_group_arn:
        progress.detail(
            "elbv2:DescribeTargetGroups", TargetGroupArn=target_group_arn
        )
        tg_res = elbv2(
            "DescribeTargetGroups",
            {"TargetGroupArns": [target_group_arn]},
        )
        if not tg_res["ok"]:
            progress.failure(3, "Describing target group", tg_res.get("error"))
            return tg_res
        target_groups = tg_res["value"].get("TargetGroups", [])
        if target_groups:
            lb_arns = target_groups[0].get("LoadBalancerArns", [])
            load_balancer_arn = lb_arns[0] if lb_arns else None
    progress.success(3, "Described target group", load_balancer_arn)

    # Step 4: load balancer -> DNS name + ALB security group. The listener is
    # read best-effort so the output shape matches a fresh create exactly.
    progress.report(4, "Describing load balancer")
    dns_name = ""
    alb_sg_id = None
    listener_arn = None
    if load_balancer_arn:
        progress.detail(
            "elbv2:DescribeLoadBalancers", LoadBalancerArn=load_balancer_arn
        )
        lb_res = elbv2(
            "DescribeLoadBalancers",
            {"LoadBalancerArns": [load_balancer_arn]},
        )
        if not lb_res["ok"]:
            progress.failure(
                4, "Describing load balancer", lb_res.get("error")
            )
            return lb_res
        lbs = lb_res["value"].get("LoadBalancers", [])
        if lbs:
            dns_name = lbs[0].get("DNSName", "")
            alb_sgs = lbs[0].get("SecurityGroups", [])
            alb_sg_id = alb_sgs[0] if alb_sgs else None
        listener_res = elbv2(
            "DescribeListeners",
            {"LoadBalancerArn": load_balancer_arn},
        )
        if listener_res["ok"]:
            listeners = listener_res["value"].get("Listeners", [])
            if listeners:
                listener_arn = listeners[0].get("ListenerArn")
    progress.success(4, "Described load balancer", dns_name or None)

    url = "http://{0}".format(dns_name) if dns_name else ""
    return ok_result(
        {
            "ServiceArn": service_arn,
            "ServiceName": service_name,
            "TaskDefinitionArn": task_definition_arn,
            "LoadBalancerDnsName": dns_name,
            "LoadBalancerArn": load_balancer_arn,
            "TargetGroupArn": target_group_arn,
            "ListenerArn": listener_arn,
            "AlbSecurityGroupId": alb_sg_id,
            "ServiceSecurityGroupId": service_sg_id,
            "ExecutionRoleArn": execution_role_arn,
            "TaskRoleArn": task_role_arn,
            "LogGroupName": log_group_name,
            "Url": url,
            # Nothing was created on this run; keep the key shape identical to a
            # fresh create (ResourceTracker.summary()).
            "ResourcesCreated": {"created_resources": [], "count": 0},
            "Idempotent": True,
        }
    )


def create_fargate_web_service(
    call_ops, parsed_args, subnet_ids, region=None, stderr=None
):
    """Provision an ALB-fronted Fargate service, rolling back on any failure.

    ``call_ops`` maps a service name (``"ecs"``, ``"iam"``, ``"ec2"``,
    ``"elbv2"``, ``"logs"``) to a ``(operation, params) -> result`` call bridge.
    ``subnet_ids`` is the validated, non-empty list of subnet ids. Creates the
    resources in strict dependency order, tracking each for LIFO rollback, and
    returns an :func:`ok_result` summarizing the service on success or the first
    failing step's error result (after rollback) on failure.
    """
    ecs = call_ops["ecs"]
    iam = call_ops["iam"]
    ec2 = call_ops["ec2"]
    elbv2 = call_ops["elbv2"]
    logs = call_ops["logs"]

    cluster = parsed_args.cluster
    image = parsed_args.image
    container_port = parsed_args.container_port
    vpc_id = parsed_args.vpc_id
    cpu = parsed_args.cpu
    memory = parsed_args.memory
    desired_count = parsed_args.desired_count
    assign_public_ip = parsed_args.assign_public_ip

    service_name = parsed_args.service_name or _derive_service_name(image)

    # Idempotency fast path: the ECS service is the last resource created and
    # references the whole stack (task definition, target group, load balancer,
    # security groups, roles). If a service with this name already exists and is
    # ACTIVE, a prior run with the same arguments already built everything, so
    # describe and return that stack instead of creating a duplicate.
    existing = _describe_existing_service(
        call_ops, cluster, service_name,
        ProgressReporter(total_steps=4, stderr=stderr),
    )
    if existing is not None:
        return existing

    container_name = service_name
    log_group_name = "/ecs/{0}".format(service_name)
    trust_policy = json.dumps(_ECS_TASKS_TRUST_POLICY)

    tracker = ResourceTracker()
    progress = ProgressReporter(total_steps=_TOTAL_STEPS, stderr=stderr)

    # Step 1: verify the target cluster exists and is ACTIVE.
    progress.report(1, "Verifying cluster {0}".format(cluster))
    progress.detail("ecs:DescribeClusters", cluster=cluster)
    result = step(
        tracker, "Describe cluster", ecs, "DescribeClusters",
        {"clusters": [cluster]},
    )
    if not result["ok"]:
        progress.failure(1, "Verifying cluster", result.get("error"))
        return result
    clusters = result["value"].get("clusters", [])
    active = [c for c in clusters if c.get("status") == "ACTIVE"]
    if not active:
        error = (
            "Cluster {0!r} was not found or is not ACTIVE. Create it first or "
            "pass an existing cluster.".format(cluster)
        )
        progress.failure(1, "Verifying cluster", error)
        return err_result(NOT_FOUND, error)
    progress.success(1, "Cluster {0} is ACTIVE".format(cluster))

    # Step 2: execution role (image pull + log writes).
    execution_role_name = "{0}-execution-role".format(service_name)[:64]
    progress.report(2, "Creating task execution role")
    progress.detail("iam:CreateRole", RoleName=execution_role_name)
    # CreateRole is not idempotent, so tolerate a prior partial run: on an
    # already-exists conflict, fetch the existing role's ARN with GetRole and
    # continue. A reused role is intentionally NOT tracked for rollback -- this
    # run did not create it, so a later failure must not delete it. Not routed
    # through step() because step() rolls everything back on any non-ok result,
    # including the conflict we want to recover from.
    result = iam(
        "CreateRole",
        {
            "RoleName": execution_role_name,
            "AssumeRolePolicyDocument": trust_policy,
            "Description": "ECS task execution role for {0}.".format(
                service_name
            ),
        },
    )
    if result["ok"]:
        execution_role_arn = result["value"]["Role"]["Arn"]
        tracker.track(
            "iam-role", execution_role_name,
            make_delete_cleanup(
                iam, "DeleteRole", {"RoleName": execution_role_name}
            ),
        )
        progress.success(
            2, "Created execution role {0}".format(execution_role_name),
            resource_id=execution_role_arn,
        )
    elif _is_already_exists(result):
        get_res = iam("GetRole", {"RoleName": execution_role_name})
        if not get_res["ok"]:
            progress.failure(
                2, "Creating task execution role", get_res.get("error")
            )
            tracker.cleanup()
            return get_res
        execution_role_arn = get_res["value"]["Role"]["Arn"]
        progress.success(
            2,
            "Reusing existing execution role {0}".format(execution_role_name),
            resource_id=execution_role_arn,
        )
    else:
        progress.failure(2, "Creating task execution role", result.get("error"))
        tracker.cleanup()
        return result
    # Derive the partition from the role ARN so policy ARNs work in GovCloud,
    # China and ISO partitions, not just the commercial ``aws`` partition.
    partition = execution_role_arn.split(":")[1] if len(execution_role_arn.split(":")) >= 2 else "aws"
    execution_policy_arn = _EXECUTION_ROLE_POLICY_ARN_FMT.format(partition=partition)

    # Step 3: attach the managed execution-role policy.
    progress.report(3, "Attaching AmazonECSTaskExecutionRolePolicy")
    progress.detail(
        "iam:AttachRolePolicy",
        RoleName=execution_role_name,
        PolicyArn=execution_policy_arn,
    )
    result = step(
        tracker, "Attach execution policy", iam, "AttachRolePolicy",
        {
            "RoleName": execution_role_name,
            "PolicyArn": execution_policy_arn,
        },
    )
    if not result["ok"]:
        progress.failure(3, "Attaching execution policy", result.get("error"))
        return result
    tracker.track(
        "iam-policy-attachment",
        "{0}:{1}".format(execution_role_name, execution_policy_arn),
        make_detach_cleanup(
            iam, "DetachRolePolicy",
            {
                "RoleName": execution_role_name,
                "PolicyArn": execution_policy_arn,
            },
        ),
    )
    progress.success(3, "Attached execution policy")

    # Step 4: task role (the application's own permissions).
    task_role_name = "{0}-task-role".format(service_name)[:64]
    progress.report(4, "Creating task role")
    progress.detail("iam:CreateRole", RoleName=task_role_name)
    result = iam(
        "CreateRole",
        {
            "RoleName": task_role_name,
            "AssumeRolePolicyDocument": trust_policy,
            "Description": "ECS task role for {0}.".format(service_name),
        },
    )
    if result["ok"]:
        task_role_arn = result["value"]["Role"]["Arn"]
        tracker.track(
            "iam-role", task_role_name,
            make_delete_cleanup(
                iam, "DeleteRole", {"RoleName": task_role_name}
            ),
        )
        progress.success(
            4, "Created task role {0}".format(task_role_name),
            resource_id=task_role_arn,
        )
    elif _is_already_exists(result):
        get_res = iam("GetRole", {"RoleName": task_role_name})
        if not get_res["ok"]:
            progress.failure(4, "Creating task role", get_res.get("error"))
            tracker.cleanup()
            return get_res
        task_role_arn = get_res["value"]["Role"]["Arn"]
        progress.success(
            4, "Reusing existing task role {0}".format(task_role_name),
            resource_id=task_role_arn,
        )
    else:
        progress.failure(4, "Creating task role", result.get("error"))
        tracker.cleanup()
        return result

    # Step 5: CloudWatch log group for the container's awslogs driver.
    progress.report(5, "Creating log group {0}".format(log_group_name))
    progress.detail("logs:CreateLogGroup", logGroupName=log_group_name)
    result = logs("CreateLogGroup", {"logGroupName": log_group_name})
    if result["ok"]:
        tracker.track(
            "logs-log-group", log_group_name,
            make_delete_cleanup(
                logs, "DeleteLogGroup", {"logGroupName": log_group_name}
            ),
        )
        progress.success(5, "Created log group", resource_id=log_group_name)
    elif _is_already_exists(result):
        # Log group already exists (ResourceAlreadyExistsException) from a prior
        # partial run; reuse it, untracked so rollback won't delete it.
        progress.success(
            5, "Reusing existing log group", resource_id=log_group_name
        )
    else:
        progress.failure(5, "Creating log group", result.get("error"))
        tracker.cleanup()
        return result

    # Step 6: register the Fargate task definition.
    progress.report(6, "Registering task definition")
    task_def_params = {
        "family": service_name,
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["FARGATE"],
        "cpu": cpu,
        "memory": memory,
        "executionRoleArn": execution_role_arn,
        "taskRoleArn": task_role_arn,
        "containerDefinitions": [
            {
                "name": container_name,
                "image": image,
                "portMappings": [
                    {"containerPort": container_port, "protocol": "tcp"}
                ],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": log_group_name,
                        "awslogs-region": region or "",
                        "awslogs-stream-prefix": "ecs",
                    },
                },
            }
        ],
    }
    progress.detail("ecs:RegisterTaskDefinition", family=service_name)
    result = step(
        tracker, "Register task definition", ecs, "RegisterTaskDefinition",
        task_def_params,
    )
    if not result["ok"]:
        progress.failure(6, "Registering task definition", result.get("error"))
        return result
    task_definition_arn = result["value"]["taskDefinition"][
        "taskDefinitionArn"
    ]
    tracker.track(
        "ecs-task-definition", task_definition_arn,
        make_delete_cleanup(
            ecs, "DeregisterTaskDefinition",
            {"taskDefinition": task_definition_arn},
        ),
    )
    progress.success(6, "Registered task definition", resource_id=task_definition_arn)

    # Step 7: ALB security group.
    alb_sg_name = "{0}-alb-sg".format(service_name)
    progress.report(7, "Creating ALB security group")
    progress.detail(
        "ec2:CreateSecurityGroup",
        GroupName=alb_sg_name,
        VpcId=vpc_id,
    )
    result = ec2(
        "CreateSecurityGroup",
        {
            "GroupName": alb_sg_name,
            "Description": "ALB security group for {0}.".format(service_name),
            "VpcId": vpc_id,
        },
    )
    if result["ok"]:
        alb_sg_id = result["value"]["GroupId"]
        tracker.track(
            "ec2-security-group", alb_sg_id,
            make_delete_cleanup(
                ec2, "DeleteSecurityGroup", {"GroupId": alb_sg_id}
            ),
        )
        progress.success(7, "Created ALB security group", resource_id=alb_sg_id)
    elif _is_already_exists(result, _DUPLICATE_SG_AWS_CODE):
        # A group with this name already exists in the VPC from a prior partial
        # run; look up its id and reuse it, untracked so rollback won't delete it.
        alb_sg_id = _lookup_security_group_id(ec2, alb_sg_name, vpc_id)
        if alb_sg_id is None:
            progress.failure(
                7, "Creating ALB security group", result.get("error")
            )
            tracker.cleanup()
            return result
        progress.success(
            7, "Reusing existing ALB security group", resource_id=alb_sg_id
        )
    else:
        progress.failure(7, "Creating ALB security group", result.get("error"))
        tracker.cleanup()
        return result

    # Step 8: allow public HTTP in to the ALB. Ingress rules are removed with
    # the security group, so this step needs no separate rollback entry.
    progress.report(8, "Authorizing public HTTP to the ALB")
    progress.detail(
        "ec2:AuthorizeSecurityGroupIngress", GroupId=alb_sg_id, FromPort=80
    )
    result = step(
        tracker, "Authorize ALB ingress", ec2,
        "AuthorizeSecurityGroupIngress",
        {
            "GroupId": alb_sg_id,
            "IpPermissions": [
                {
                    "IpProtocol": "tcp",
                    "FromPort": 80,
                    "ToPort": 80,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                }
            ],
        },
    )
    if not result["ok"]:
        progress.failure(8, "Authorizing ALB ingress", result.get("error"))
        return result
    progress.success(8, "Authorized public HTTP to the ALB")

    # Step 9: service security group.
    service_sg_name = "{0}-service-sg".format(service_name)
    progress.report(9, "Creating service security group")
    progress.detail(
        "ec2:CreateSecurityGroup",
        GroupName=service_sg_name,
        VpcId=vpc_id,
    )
    result = ec2(
        "CreateSecurityGroup",
        {
            "GroupName": service_sg_name,
            "Description": "Service security group for {0}.".format(
                service_name
            ),
            "VpcId": vpc_id,
        },
    )
    if result["ok"]:
        service_sg_id = result["value"]["GroupId"]
        tracker.track(
            "ec2-security-group", service_sg_id,
            make_delete_cleanup(
                ec2, "DeleteSecurityGroup", {"GroupId": service_sg_id}
            ),
        )
        progress.success(
            9, "Created service security group", resource_id=service_sg_id
        )
    elif _is_already_exists(result, _DUPLICATE_SG_AWS_CODE):
        service_sg_id = _lookup_security_group_id(ec2, service_sg_name, vpc_id)
        if service_sg_id is None:
            progress.failure(
                9, "Creating service security group", result.get("error")
            )
            tracker.cleanup()
            return result
        progress.success(
            9, "Reusing existing service security group",
            resource_id=service_sg_id,
        )
    else:
        progress.failure(
            9, "Creating service security group", result.get("error")
        )
        tracker.cleanup()
        return result

    # Step 10: allow the ALB SG in to the service SG on the container port.
    progress.report(10, "Authorizing ALB to reach the service")
    progress.detail(
        "ec2:AuthorizeSecurityGroupIngress",
        GroupId=service_sg_id,
        FromPort=container_port,
    )
    result = step(
        tracker, "Authorize service ingress", ec2,
        "AuthorizeSecurityGroupIngress",
        {
            "GroupId": service_sg_id,
            "IpPermissions": [
                {
                    "IpProtocol": "tcp",
                    "FromPort": container_port,
                    "ToPort": container_port,
                    "UserIdGroupPairs": [{"GroupId": alb_sg_id}],
                }
            ],
        },
    )
    if not result["ok"]:
        progress.failure(10, "Authorizing service ingress", result.get("error"))
        return result
    progress.success(10, "Authorized ALB to reach the service")

    # Step 11: internet-facing Application Load Balancer.
    progress.report(11, "Creating load balancer")
    progress.detail(
        "elbv2:CreateLoadBalancer", Name=_elb_name(service_name, "-alb")
    )
    result = step(
        tracker, "Create load balancer", elbv2, "CreateLoadBalancer",
        {
            "Name": _elb_name(service_name, "-alb"),
            "Subnets": subnet_ids,
            "SecurityGroups": [alb_sg_id],
            "Scheme": "internet-facing",
            "Type": "application",
            "IpAddressType": "ipv4",
        },
    )
    if not result["ok"]:
        progress.failure(11, "Creating load balancer", result.get("error"))
        return result
    load_balancer = result["value"]["LoadBalancers"][0]
    load_balancer_arn = load_balancer["LoadBalancerArn"]
    dns_name = load_balancer.get("DNSName", "")
    tracker.track(
        "elbv2-load-balancer", load_balancer_arn,
        make_delete_cleanup(
            elbv2, "DeleteLoadBalancer",
            {"LoadBalancerArn": load_balancer_arn},
        ),
    )
    progress.success(
        11, "Created load balancer {0}".format(dns_name),
        resource_id=load_balancer_arn,
    )

    # Step 12: IP-type target group (required for Fargate awsvpc).
    progress.report(12, "Creating target group")
    progress.detail(
        "elbv2:CreateTargetGroup", Name=_elb_name(service_name, "-tg")
    )
    result = step(
        tracker, "Create target group", elbv2, "CreateTargetGroup",
        {
            "Name": _elb_name(service_name, "-tg"),
            "Protocol": "HTTP",
            "Port": container_port,
            "VpcId": vpc_id,
            "TargetType": "ip",
            "HealthCheckEnabled": True,
            "HealthCheckProtocol": "HTTP",
            "HealthCheckPath": "/",
        },
    )
    if not result["ok"]:
        progress.failure(12, "Creating target group", result.get("error"))
        return result
    target_group_arn = result["value"]["TargetGroups"][0]["TargetGroupArn"]
    tracker.track(
        "elbv2-target-group", target_group_arn,
        make_delete_cleanup(
            elbv2, "DeleteTargetGroup",
            {"TargetGroupArn": target_group_arn},
        ),
    )
    progress.success(12, "Created target group", resource_id=target_group_arn)

    # Step 13: listener on :80 forwarding to the target group.
    progress.report(13, "Creating listener")
    progress.detail(
        "elbv2:CreateListener", LoadBalancerArn=load_balancer_arn, Port=80
    )
    result = step(
        tracker, "Create listener", elbv2, "CreateListener",
        {
            "LoadBalancerArn": load_balancer_arn,
            "Protocol": "HTTP",
            "Port": 80,
            "DefaultActions": [
                {"Type": "forward", "TargetGroupArn": target_group_arn}
            ],
        },
    )
    if not result["ok"]:
        progress.failure(13, "Creating listener", result.get("error"))
        return result
    listener_arn = result["value"]["Listeners"][0]["ListenerArn"]
    tracker.track(
        "elbv2-listener", listener_arn,
        make_delete_cleanup(
            elbv2, "DeleteListener", {"ListenerArn": listener_arn}
        ),
    )
    progress.success(13, "Created listener", resource_id=listener_arn)

    # Step 14: the Fargate service, wired to the target group.
    progress.report(14, "Creating Fargate service {0}".format(service_name))
    progress.detail(
        "ecs:CreateService", serviceName=service_name, cluster=cluster
    )
    result = step(
        tracker, "Create service", ecs, "CreateService",
        {
            "cluster": cluster,
            "serviceName": service_name,
            "taskDefinition": task_definition_arn,
            "desiredCount": desired_count,
            "launchType": "FARGATE",
            "networkConfiguration": {
                "awsvpcConfiguration": {
                    "subnets": subnet_ids,
                    "securityGroups": [service_sg_id],
                    "assignPublicIp": assign_public_ip,
                }
            },
            "loadBalancers": [
                {
                    "targetGroupArn": target_group_arn,
                    "containerName": container_name,
                    "containerPort": container_port,
                }
            ],
        },
    )
    if not result["ok"]:
        progress.failure(14, "Creating Fargate service", result.get("error"))
        return result
    service_arn = result["value"]["service"]["serviceArn"]
    tracker.track(
        "ecs-service", service_arn,
        _make_service_cleanup(ecs, cluster, service_arn),
    )
    progress.success(
        14, "Created Fargate service {0}".format(service_name),
        resource_id=service_arn,
    )

    url = "http://{0}".format(dns_name) if dns_name else ""
    return ok_result(
        {
            "ServiceArn": service_arn,
            "ServiceName": service_name,
            "TaskDefinitionArn": task_definition_arn,
            "LoadBalancerDnsName": dns_name,
            "LoadBalancerArn": load_balancer_arn,
            "TargetGroupArn": target_group_arn,
            "ListenerArn": listener_arn,
            "AlbSecurityGroupId": alb_sg_id,
            "ServiceSecurityGroupId": service_sg_id,
            "ExecutionRoleArn": execution_role_arn,
            "TaskRoleArn": task_role_arn,
            "LogGroupName": log_group_name,
            "Url": url,
            "ResourcesCreated": tracker.summary(),
            "Idempotent": False,
        }
    )


def _make_service_cleanup(ecs_call, cluster, service_arn):
    """Build a best-effort cleanup that scales a service to zero and deletes it.

    An ECS service with running tasks cannot be deleted outright, so rollback
    first drives ``desiredCount`` to 0 via ``UpdateService`` and then issues a
    forced ``DeleteService``. After the delete request, the cleanup polls
    ``DescribeServices`` until the service reaches ``INACTIVE`` (or a generous
    timeout), so that Fargate task ENIs are released and downstream security
    group deletes have a chance to succeed. All calls go through the ``call``
    bridge, which returns errors as data, so this callable never raises.
    """

    def _cleanup():
        ecs_call(
            "UpdateService",
            {"cluster": cluster, "service": service_arn, "desiredCount": 0},
        )
        ecs_call(
            "DeleteService",
            {"cluster": cluster, "service": service_arn, "force": True},
        )
        # Wait for the service to reach INACTIVE so Fargate ENIs are released
        # and security-group deletes downstream don't hit DependencyViolation.
        import time
        for _ in range(30):  # ~60 seconds max (30 × 2s)
            time.sleep(2)
            result = ecs_call(
                "DescribeServices",
                {"cluster": cluster, "services": [service_arn]},
            )
            if not result.get("ok"):
                break  # can't poll, best-effort
            services = result.get("value", {}).get("services", [])
            if not services or services[0].get("status") == "INACTIVE":
                break

    return _cleanup


class CreateFargateWebServiceCommand(BasicCommand):
    NAME = "create-fargate-web-service"
    DESCRIPTION = (
        "Create an internet-facing ECS Fargate web service fronted by an "
        "Application Load Balancer in one command, replicating CDK's "
        "ApplicationLoadBalancedFargateService construct. Provisions IAM roles, "
        "a log group, a task definition, security groups, an ALB, a target "
        "group, a listener and the service, and rolls everything back if any "
        "step fails."
    )
    ARG_TABLE = [
        {
            "name": "cluster",
            "required": True,
            "cli_type_name": "string",
            "help_text": "ECS cluster name or ARN. The cluster must exist and "
            "be ACTIVE.",
        },
        {
            "name": "image",
            "required": True,
            "cli_type_name": "string",
            "help_text": "Container image URI to run.",
        },
        {
            "name": "container-port",
            "cli_type_name": "integer",
            "default": 80,
            "help_text": "Container port the service listens on. Default: 80.",
        },
        {
            "name": "vpc-id",
            "required": True,
            "cli_type_name": "string",
            "help_text": "VPC in which to create the load balancer, security "
            "groups and target group.",
        },
        {
            "name": "subnet-ids",
            "required": True,
            "cli_type_name": "string",
            "help_text": "Comma-separated subnet ids for the service and the "
            "load balancer.",
        },
        {
            "name": "cpu",
            "cli_type_name": "string",
            "default": "256",
            "help_text": "Fargate CPU units. Default: 256.",
        },
        {
            "name": "memory",
            "cli_type_name": "string",
            "default": "512",
            "help_text": "Fargate memory in MiB. Default: 512.",
        },
        {
            "name": "desired-count",
            "cli_type_name": "integer",
            "default": 1,
            "help_text": "Number of tasks to run. Default: 1.",
        },
        {
            "name": "service-name",
            "cli_type_name": "string",
            "help_text": "ECS service name. Defaults to a name derived from the "
            "image repository.",
        },
        {
            "name": "assign-public-ip",
            "cli_type_name": "string",
            "default": "ENABLED",
            "help_text": "Whether tasks receive a public IP: ENABLED or "
            "DISABLED. Default: ENABLED.",
        },
    ]

    def __init__(self, session, clients=None, stdout=None, stderr=None):
        super().__init__(session)
        self._provided_clients = clients or {}
        self._stdout = stdout
        self._stderr = stderr

    def _run_main(self, parsed_args, parsed_globals):
        # Validate input up front; bad input fails as ParamValidationError
        # (exit code 252) before any API call is made.
        problem, subnet_ids = _validate(parsed_args)
        if problem is not None:
            return emit_result(
                self.NAME, problem, parsed_globals, self._session,
                self._stdout, self._stderr,
            )

        # Create one client per service lazily, honouring
        # --region/--endpoint-url/--verify-ssl via the shared helper, and build
        # a per-service (operation, params) -> result call bridge.
        cache = dict(self._provided_clients)

        def make_call_op(service):
            def _op(operation, params):
                client = cache.get(service)
                if client is None:
                    client = cache[service] = create_client(
                        self._session, service, parsed_globals
                    )
                return call(client, operation, params)

            return _op

        call_ops = {service: make_call_op(service) for service in _SERVICES}

        region = getattr(parsed_globals, "region", None)
        if region is None:
            region = self._session.get_config_variable("region")

        result = create_fargate_web_service(
            call_ops,
            parsed_args,
            subnet_ids,
            region=region,
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
    """Inject the ``create-fargate-web-service`` verb into the ecs command table."""
    event_handlers.register("building-command-table.ecs", _inject_commands)


def _inject_commands(command_table, session, **kwargs):
    command_table["create-fargate-web-service"] = (
        CreateFargateWebServiceCommand(session)
    )
