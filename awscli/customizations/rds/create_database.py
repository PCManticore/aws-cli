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
"""High-order ``aws rds create-database`` command.

Replicates the end state of CDK's ``rds.DatabaseInstance`` L2 construct: a single
verb that stands up an RDS database together with everything it needs to be
usable -- a dedicated security group with an ingress rule, a DB subnet group
built from the VPC's private subnets, generated master credentials stored in
Secrets Manager, and the DB instance itself -- then backfills the connection
details into the secret so applications can consume it directly.

A single API never does this, and a failure partway through would otherwise
leave orphaned resources behind. The command therefore runs each write through
the :mod:`~awscli.customizations.compositecreate` ``step`` primitive, tracking
every created resource so a later failure rolls them back in reverse (LIFO)
order.

Creating a DB instance is a long-running operation (often 5-15 minutes). This
command deliberately does NOT wait for the instance to become available: it
returns as soon as ``CreateDBInstance`` is accepted, reporting the identifier and
whatever endpoint information is already known.
"""

from __future__ import annotations

import json
import logging
import time

from awscli.customizations.commands import BasicCommand
from awscli.customizations.compositecreate.cleanup import make_delete_cleanup
from awscli.customizations.compositecreate.helpers import (
    call,
    create_client,
    emit_result,
    err_result,
    invalid_input,
    ok_result,
    step,
    CONFLICT,
    NOT_FOUND,
    ResourceTracker,
)
from awscli.customizations.compositecreate.progress import ProgressReporter

logger = logging.getLogger(__name__)

# Database engines this command supports, mapped to their default listener port.
# CDK derives the same defaults from each engine's EngineVersion family.
_ENGINE_DEFAULT_PORTS = {
    'mysql': 3306,
    'mariadb': 3306,
    'postgres': 5432,
    'oracle-ee': 1521,
    'sqlserver-ee': 1433,
}

# SQL Server rejects the CreateDBInstance ``DBName`` parameter; every other
# supported engine accepts it. Used to decide whether to pass ``DBName`` through.
_ENGINES_WITHOUT_DB_NAME = frozenset({'sqlserver-ee'})

# Characters excluded from the generated master password so the value is safe to
# embed in connection strings, shell invocations and RDS itself.
_PASSWORD_EXCLUDE_CHARACTERS = '/@"\\ '

# Number of progress steps reported to stderr; see ``create_database`` below.
_TOTAL_STEPS = 7


def _default_db_name(engine):
    """Derive a sensible default database name from the engine identifier.

    The engine string may contain characters (such as the ``-`` in
    ``oracle-ee``) that are not valid in an RDS database name, so they are
    stripped. Returns a lowercase alphanumeric name, e.g. ``oracleeedb``.
    """
    base = ''.join(ch for ch in engine.lower() if ch.isalnum())
    return '{}db'.format(base or 'app')


def _derive_instance_identifier(db_name, engine):
    """Build a valid, unique ``DBInstanceIdentifier``.

    Prefers ``db_name`` when supplied, otherwise generates one from the engine
    and a timestamp. The result is sanitized to the RDS identifier rules:
    lowercase, alphanumeric-or-hyphen, starting with a letter. A timestamp
    suffix keeps generated identifiers unique across invocations.
    """
    if db_name:
        candidate = db_name.lower()
    else:
        candidate = '{}-{}'.format(engine.lower(), int(time.time()))
    sanitized = ''.join(
        ch if (ch.isalnum() or ch == '-') else '-' for ch in candidate
    )
    sanitized = sanitized.strip('-') or 'database'
    if not sanitized[0].isalpha():
        sanitized = 'db-' + sanitized
    return sanitized


def _validate_args(parsed_args):
    """Validate CLI input before any API call, returning a result or ``None``.

    Returns an ``invalid_input`` failure result (which maps to the CLI's native
    ParamValidationError / exit code 252) on the first problem found, or ``None``
    when every argument is acceptable. Validating up front avoids creating
    resources for a request that can never succeed.
    """
    engine = parsed_args.engine
    if engine not in _ENGINE_DEFAULT_PORTS:
        supported = ', '.join(sorted(_ENGINE_DEFAULT_PORTS))
        return invalid_input(
            '--engine {!r} is not supported; choose one of: {}.'.format(
                engine, supported
            )
        )
    if not parsed_args.instance_class or not parsed_args.instance_class.strip():
        return invalid_input('--instance-class is required.')
    if not parsed_args.vpc_id or not parsed_args.vpc_id.strip():
        return invalid_input('--vpc-id is required.')
    storage = parsed_args.allocated_storage
    if not isinstance(storage, int) or isinstance(storage, bool) or storage < 1:
        return invalid_input(
            '--allocated-storage must be a positive integer, got {!r}.'.format(
                storage
            )
        )
    port = parsed_args.port
    if port is not None:
        if (
            not isinstance(port, int)
            or isinstance(port, bool)
            or not (1 <= port <= 65535)
        ):
            return invalid_input(
                '--port must be an integer between 1 and 65535, got '
                '{!r}.'.format(port)
            )
    if not parsed_args.master_username or not parsed_args.master_username.strip():
        return invalid_input('--master-username must be a non-empty string.')
    return None


def _discover_private_subnets(call_op, vpc_id):
    """Find candidate private subnets in ``vpc_id`` for the DB subnet group.

    Calls ``DescribeSubnets`` filtered to the VPC and prefers subnets that do
    NOT auto-assign public IPs (``MapPublicIpOnLaunch`` false), which is the
    heuristic CDK-style tooling uses to distinguish private subnets without
    walking route tables. If none look private, every subnet in the VPC is
    returned so the command can still proceed. Returns a result whose ``value``
    is the list of subnet-id strings.
    """
    result = call_op(
        'DescribeSubnets', {'Filters': [{'Name': 'vpc-id', 'Values': [vpc_id]}]}
    )
    if not result['ok']:
        return result
    subnets = result['value'].get('Subnets', [])
    private = [
        s['SubnetId'] for s in subnets if not s.get('MapPublicIpOnLaunch', False)
    ]
    chosen = private or [s['SubnetId'] for s in subnets]
    return ok_result(chosen)


def _lookup_vpc_cidr(call_op, vpc_id):
    """Return the primary IPv4 CIDR block of ``vpc_id``.

    Used as the default ingress CIDR when the caller does not supply
    ``--ingress-cidr``. Returns a result whose ``value`` is the CIDR string.
    """
    result = call_op('DescribeVpcs', {'VpcIds': [vpc_id]})
    if not result['ok']:
        return result
    vpcs = result['value'].get('Vpcs', [])
    if not vpcs:
        return ok_result(None)
    return ok_result(vpcs[0].get('CidrBlock'))


# Raw, service-specific AWS error codes that mean "this resource already exists".
# The normalized result envelope maps only some already-exists errors to
# ``CONFLICT``: EC2's ``InvalidGroup.Duplicate`` and RDS's ``*AlreadyExists`` are
# returned as HTTP 400 and therefore fall through to ``SERVICE_ERROR``. We match
# the raw ``aws_code`` as well so idempotency works across all of them.
_ALREADY_EXISTS_AWS_CODES = frozenset(
    {
        'InvalidGroup.Duplicate',  # ec2:CreateSecurityGroup
        'DBSubnetGroupAlreadyExists',  # rds:CreateDBSubnetGroup
        'DBInstanceAlreadyExists',  # rds:CreateDBInstance
        'ResourceExistsException',  # secretsmanager:CreateSecret
    }
)


def _is_already_exists(result):
    """Return True when a failed result means the resource already exists.

    Matches both the normalized ``CONFLICT`` code and the raw service-specific
    ``aws_code`` values in :data:`_ALREADY_EXISTS_AWS_CODES`, because several of
    those codes are returned as HTTP 400 and so never normalize to ``CONFLICT``.
    """
    if result.get('ok'):
        return False
    if result.get('code') == CONFLICT:
        return True
    return result.get('aws_code') in _ALREADY_EXISTS_AWS_CODES


def _create_or_reuse(
    tracker,
    call_fn,
    create_op,
    create_params,
    *,
    resource_type,
    extract_id,
    make_cleanup,
    describe_existing,
):
    """Create a resource, or reuse the pre-existing one on an already-exists conflict.

    The idempotent analogue of :func:`step` for a resource that may already exist
    from an earlier (possibly partial) run:

    * On success, the created resource is tracked for LIFO rollback (via
      ``make_cleanup(resource_id)``), exactly as the create flow did before, and
      the result ``value`` is ``(resource_id, False)``.
    * On an already-exists conflict (:func:`_is_already_exists`),
      ``describe_existing()`` looks up the existing resource and it is REUSED
      WITHOUT being tracked -- a resource this run did not create must never be
      torn down by a later rollback. The result ``value`` is
      ``(existing_id, True)``.
    * On any other error, everything created so far is rolled back (matching
      :func:`step`) and the error result is returned.

    ``extract_id`` pulls the resource id/arn out of the create response and
    ``resource_type`` labels it in the tracker.
    """
    result = call_fn(create_op, create_params)
    if result['ok']:
        resource_id = extract_id(result['value'])
        tracker.track(resource_type, resource_id, make_cleanup(resource_id))
        return ok_result((resource_id, False))
    if _is_already_exists(result):
        existing = describe_existing()
        if not existing['ok']:
            tracker.cleanup()
            return existing
        return ok_result((existing['value'], True))
    logger.info(
        'Create %s failed (%s); rolling back %d tracked resource(s).',
        create_op, result.get('code'), len(tracker.resources),
    )
    tracker.cleanup()
    return result


def _describe_existing_security_group(call_ec2, group_name, vpc_id):
    """Look up an existing security group by name within a VPC.

    Used to reuse a security group that ``CreateSecurityGroup`` reported as a
    duplicate. Returns a result whose ``value`` is the GroupId, or a NOT_FOUND
    failure if the group cannot be found (an unexpected state after a duplicate
    error, e.g. a race).
    """
    result = call_ec2(
        'DescribeSecurityGroups',
        {
            'Filters': [
                {'Name': 'group-name', 'Values': [group_name]},
                {'Name': 'vpc-id', 'Values': [vpc_id]},
            ],
        },
    )
    if not result['ok']:
        return result
    groups = result['value'].get('SecurityGroups', [])
    if not groups:
        return err_result(
            NOT_FOUND,
            'Security group {!r} not found in VPC {!r}.'.format(
                group_name, vpc_id
            ),
        )
    return ok_result(groups[0]['GroupId'])


def _describe_existing_subnet_group(call_rds, subnet_group_name):
    """Confirm an existing DB subnet group by name.

    Used to reuse a DB subnet group that ``CreateDBSubnetGroup`` reported as
    already existing. Returns a result whose ``value`` is the subnet group name,
    or a NOT_FOUND failure if it cannot be found.
    """
    result = call_rds(
        'DescribeDBSubnetGroups', {'DBSubnetGroupName': subnet_group_name}
    )
    if not result['ok']:
        return result
    groups = result['value'].get('DBSubnetGroups', [])
    if not groups:
        return err_result(
            NOT_FOUND,
            'DB subnet group {!r} not found.'.format(subnet_group_name),
        )
    return ok_result(subnet_group_name)


def _describe_existing_secret(call_secrets, secret_name):
    """Look up an existing secret's ARN via ``GetSecretValue``.

    Used to reuse a secret that ``CreateSecret`` reported as already existing.
    The master username/password already live in the existing secret; the caller
    re-reads them with its own ``GetSecretValue`` step. Returns a result whose
    ``value`` is the secret ARN.
    """
    result = call_secrets('GetSecretValue', {'SecretId': secret_name})
    if not result['ok']:
        return result
    return ok_result(result['value']['ARN'])


def _find_secret_arn(call_secrets, secret_name):
    """Best-effort lookup of the master-credentials secret ARN by name.

    Returns the ARN string, or ``None`` when the secret cannot be found or the
    lookup fails. The DB instance existing matters more than the secret, so a
    missing secret never aborts the idempotent response.
    """
    result = call_secrets(
        'ListSecrets',
        {'Filters': [{'Key': 'name', 'Values': [secret_name]}]},
    )
    if not result['ok']:
        return None
    for entry in result['value'].get('SecretList', []):
        if entry.get('Name') == secret_name:
            return entry.get('ARN')
    return None


def _describe_existing_database(call_ops, db_identifier, progress):
    """Describe an already-existing DB instance, or return ``None`` if absent.

    The idempotency fast path: if a DB instance with ``db_identifier`` already
    exists (e.g. a previous successful run with the same arguments), gather its
    existing resources -- endpoint, engine, security groups, subnet group, and
    the master-credentials secret ARN (matched by the ``<id>-master-credentials``
    name pattern) -- and return an ``ok_result`` in the same shape as a fresh
    create, flagged ``Idempotent: True``.

    Returns ``None`` when the instance does not exist OR when ``DescribeDBInstances``
    fails for any reason (e.g. the caller lacks ``rds:DescribeDBInstances``): the
    fast path is a pure optimization, so an inconclusive describe falls through to
    the create path, whose own conflict handling still makes a re-run idempotent.

    ``call_ops`` is a mapping of service name to ``call`` bridge, with keys
    ``'rds'`` and ``'secrets'``.
    """
    call_rds = call_ops['rds']
    call_secrets = call_ops['secrets']
    progress.detail(
        'rds:DescribeDBInstances', DBInstanceIdentifier=db_identifier
    )
    result = call_rds(
        'DescribeDBInstances', {'DBInstanceIdentifier': db_identifier}
    )
    if not result['ok']:
        return None
    instances = result['value'].get('DBInstances', [])
    if not instances:
        return None
    db = instances[0]
    endpoint = db.get('Endpoint') or {}
    sg_ids = [
        g['VpcSecurityGroupId']
        for g in db.get('VpcSecurityGroups', [])
        if g.get('VpcSecurityGroupId')
    ]
    subnet_group_name = (db.get('DBSubnetGroup') or {}).get('DBSubnetGroupName')
    secret_name = '{}-master-credentials'.format(db_identifier)
    progress.detail('secretsmanager:ListSecrets', Name=secret_name)
    secret_arn = _find_secret_arn(call_secrets, secret_name)
    return ok_result(
        {
            'DBInstanceIdentifier': db.get(
                'DBInstanceIdentifier', db_identifier
            ),
            'Engine': db.get('Engine'),
            'Endpoint': {
                'Address': endpoint.get('Address', ''),
                'Port': endpoint.get('Port'),
            },
            'SecretArn': secret_arn,
            'SecurityGroupId': sg_ids[0] if sg_ids else None,
            'SecurityGroupIds': sg_ids,
            'SubnetGroupName': subnet_group_name,
            'Status': db.get('DBInstanceStatus', 'available'),
            'Note': 'Database {!r} already exists; returning existing '
            'resources.'.format(db_identifier),
            'ResourcesCreated': ResourceTracker().summary(),
            'Idempotent': True,
        }
    )


def create_database(call_ec2, call_rds, call_secrets, parsed_args, stderr=None):
    """Create an RDS database and its supporting resources, rolling back on error.

    Runs the full create sequence -- discover subnets, create the security group
    and ingress rule, create the DB subnet group, generate and store master
    credentials, create the DB instance, and backfill the connection details into
    the secret. Every write is tracked for LIFO rollback via
    :func:`~awscli.customizations.compositecreate.helpers.step`; the first failing
    step tears down everything created before it. ``call_ec2``/``call_rds``/
    ``call_secrets`` are per-service ``call`` bridges. On success returns an
    ``ok_result`` summarizing the created resources; the DB instance is NOT waited
    on and may still be provisioning when this returns.
    """
    tracker = ResourceTracker()
    progress = ProgressReporter(total_steps=_TOTAL_STEPS, stderr=stderr)

    engine = parsed_args.engine
    port = parsed_args.port or _ENGINE_DEFAULT_PORTS[engine]
    master_username = parsed_args.master_username
    db_name = parsed_args.db_name or _default_db_name(engine)
    instance_id = _derive_instance_identifier(parsed_args.db_name, engine)

    # Idempotency fast path: if a DB instance with this identifier already exists
    # (e.g. a previous successful run with the same arguments), describe it and
    # return its resources instead of creating anything. A non-deterministic
    # identifier (generated when --db-name is omitted) will not match, so this
    # only short-circuits when the caller supplied a stable --db-name.
    existing_result = _describe_existing_database(
        {'rds': call_rds, 'secrets': call_secrets}, instance_id, progress,
    )
    if existing_result is not None:
        return existing_result

    # Step 1: discover the private subnets (and the VPC CIDR, if we need it as
    # the default ingress range). Reads, but routed through step() so a later
    # failure still triggers rollback of anything created afterwards.
    progress.report(1, 'Discovering private subnets')
    progress.detail('ec2:DescribeSubnets', VpcId=parsed_args.vpc_id)
    subnets_result = step(
        tracker, 'Discover subnets', _discover_private_subnets,
        call_ec2, parsed_args.vpc_id,
    )
    if not subnets_result['ok']:
        progress.failure(1, 'Discover subnets', subnets_result['error'])
        return subnets_result
    subnet_ids = subnets_result['value']
    if len(subnet_ids) < 2:
        return invalid_input(
            'VPC {!r} has fewer than two usable subnets; an RDS DB subnet '
            'group needs subnets in at least two Availability Zones.'.format(
                parsed_args.vpc_id
            )
        )
    ingress_cidr = parsed_args.ingress_cidr
    if not ingress_cidr:
        progress.detail('ec2:DescribeVpcs', VpcIds=parsed_args.vpc_id)
        cidr_result = step(
            tracker, 'Lookup VPC CIDR', _lookup_vpc_cidr,
            call_ec2, parsed_args.vpc_id,
        )
        if not cidr_result['ok']:
            progress.failure(1, 'Discover subnets', cidr_result['error'])
            return cidr_result
        ingress_cidr = cidr_result['value']
        if not ingress_cidr:
            return invalid_input(
                'Could not determine a default ingress CIDR for VPC {!r}; pass '
                '--ingress-cidr explicitly.'.format(parsed_args.vpc_id)
            )
    progress.success(1, 'Discovered {} subnet(s)'.format(len(subnet_ids)))

    # Step 2: dedicated security group for the database.
    progress.report(2, 'Creating security group')
    sg_name = '{}-db-sg'.format(instance_id)
    progress.detail(
        'ec2:CreateSecurityGroup', GroupName=sg_name, VpcId=parsed_args.vpc_id
    )
    sg_result = _create_or_reuse(
        tracker, call_ec2, 'CreateSecurityGroup',
        {
            'GroupName': sg_name,
            'Description': 'Security group for RDS database {}'.format(
                instance_id
            ),
            'VpcId': parsed_args.vpc_id,
        },
        resource_type='SecurityGroup',
        extract_id=lambda v: v['GroupId'],
        make_cleanup=lambda rid: make_delete_cleanup(
            call_ec2, 'DeleteSecurityGroup', {'GroupId': rid}
        ),
        describe_existing=lambda: _describe_existing_security_group(
            call_ec2, sg_name, parsed_args.vpc_id
        ),
    )
    if not sg_result['ok']:
        progress.failure(2, 'Create security group', sg_result['error'])
        return sg_result
    sg_id, sg_reused = sg_result['value']
    progress.success(
        2,
        'Reusing existing security group'
        if sg_reused
        else 'Created security group',
        resource_id=sg_id,
    )

    # Step 3: allow inbound connections to the database port. The ingress rule
    # lives inside the security group, so it needs no separate cleanup.
    progress.report(3, 'Authorizing ingress on port {}'.format(port))
    progress.detail(
        'ec2:AuthorizeSecurityGroupIngress', GroupId=sg_id, FromPort=port
    )
    ingress_result = step(
        tracker, 'Authorize ingress', call_ec2,
        'AuthorizeSecurityGroupIngress',
        {
            'GroupId': sg_id,
            'IpPermissions': [
                {
                    'IpProtocol': 'tcp',
                    'FromPort': port,
                    'ToPort': port,
                    'IpRanges': [{'CidrIp': ingress_cidr}],
                }
            ],
        },
    )
    if not ingress_result['ok']:
        progress.failure(3, 'Authorize ingress', ingress_result['error'])
        return ingress_result
    progress.success(3, 'Authorized ingress from {}'.format(ingress_cidr))

    # Step 4: DB subnet group spanning the discovered private subnets.
    progress.report(4, 'Creating DB subnet group')
    subnet_group_name = '{}-subnet-group'.format(instance_id)
    progress.detail(
        'rds:CreateDBSubnetGroup', DBSubnetGroupName=subnet_group_name
    )
    subnet_group_result = _create_or_reuse(
        tracker, call_rds, 'CreateDBSubnetGroup',
        {
            'DBSubnetGroupName': subnet_group_name,
            'DBSubnetGroupDescription': 'Subnet group for RDS database '
            '{}'.format(instance_id),
            'SubnetIds': subnet_ids,
        },
        resource_type='DBSubnetGroup',
        extract_id=lambda v: v.get('DBSubnetGroup', {}).get(
            'DBSubnetGroupName', subnet_group_name
        ),
        make_cleanup=lambda rid: make_delete_cleanup(
            call_rds, 'DeleteDBSubnetGroup', {'DBSubnetGroupName': rid}
        ),
        describe_existing=lambda: _describe_existing_subnet_group(
            call_rds, subnet_group_name
        ),
    )
    if not subnet_group_result['ok']:
        progress.failure(
            4, 'Create DB subnet group', subnet_group_result['error']
        )
        return subnet_group_result
    subnet_group_name, subnet_group_reused = subnet_group_result['value']
    progress.success(
        4,
        'Reusing existing DB subnet group'
        if subnet_group_reused
        else 'Created DB subnet group',
        resource_id=subnet_group_name,
    )

    # Step 5: generate and store master credentials in Secrets Manager. The
    # password is generated server-side via GenerateSecretString; we then read it
    # back so it can be handed to CreateDBInstance.
    progress.report(5, 'Generating master credentials')
    secret_name = '{}-master-credentials'.format(instance_id)
    progress.detail('secretsmanager:CreateSecret', Name=secret_name)
    secret_result = _create_or_reuse(
        tracker, call_secrets, 'CreateSecret',
        {
            'Name': secret_name,
            'Description': 'Master credentials for RDS database '
            '{}'.format(instance_id),
            'GenerateSecretString': {
                'SecretStringTemplate': json.dumps(
                    {'username': master_username}
                ),
                'GenerateStringKey': 'password',
                'ExcludeCharacters': _PASSWORD_EXCLUDE_CHARACTERS,
                'PasswordLength': 30,
            },
        },
        resource_type='Secret',
        extract_id=lambda v: v['ARN'],
        make_cleanup=lambda rid: make_delete_cleanup(
            call_secrets, 'DeleteSecret',
            {'SecretId': rid, 'ForceDeleteWithoutRecovery': True},
        ),
        describe_existing=lambda: _describe_existing_secret(
            call_secrets, secret_name
        ),
    )
    if not secret_result['ok']:
        progress.failure(5, 'Create secret', secret_result['error'])
        return secret_result
    secret_arn, secret_reused = secret_result['value']
    # Retrieve the generated (or pre-existing) password to pass to
    # CreateDBInstance.
    progress.detail('secretsmanager:GetSecretValue', SecretId=secret_arn)
    get_secret_result = step(
        tracker, 'Read generated password', call_secrets, 'GetSecretValue',
        {'SecretId': secret_arn},
    )
    if not get_secret_result['ok']:
        progress.failure(5, 'Read generated password', get_secret_result['error'])
        return get_secret_result
    secret_dict = json.loads(get_secret_result['value']['SecretString'])
    password = secret_dict['password']
    progress.success(
        5,
        'Reusing existing master credentials'
        if secret_reused
        else 'Stored master credentials in {}'.format(secret_name),
        resource_id=secret_arn,
    )

    # Step 6: create the DB instance. This is long-running; we do NOT wait for it
    # to become available, returning as soon as the request is accepted.
    progress.report(6, 'Creating DB instance (not waiting for availability)')
    create_params = {
        'DBInstanceIdentifier': instance_id,
        'DBInstanceClass': parsed_args.instance_class,
        'Engine': engine,
        'MasterUsername': master_username,
        'MasterUserPassword': password,
        'AllocatedStorage': parsed_args.allocated_storage,
        'VpcSecurityGroupIds': [sg_id],
        'DBSubnetGroupName': subnet_group_name,
        'Port': port,
        'MultiAZ': bool(parsed_args.multi_az),
        'StorageEncrypted': bool(parsed_args.storage_encrypted),
    }
    # SQL Server rejects DBName; every other supported engine accepts it.
    if engine not in _ENGINES_WITHOUT_DB_NAME:
        create_params['DBName'] = db_name
    progress.detail(
        'rds:CreateDBInstance',
        DBInstanceIdentifier=instance_id,
        Engine=engine,
    )
    db_result = call_rds('CreateDBInstance', create_params)
    if not db_result['ok']:
        if _is_already_exists(db_result):
            # The instance already exists (a prior run, or a race the fast path
            # at the top missed). Treat as success and return its resources.
            progress.success(
                6, 'DB instance already exists (idempotent)',
                resource_id=instance_id,
            )
            existing = _describe_existing_database(
                {'rds': call_rds, 'secrets': call_secrets}, instance_id,
                progress,
            )
            if existing is not None:
                return existing
            # Could not describe it (e.g. no DescribeDBInstances permission);
            # return the resources already known so a re-run still succeeds.
            return ok_result(
                {
                    'DBInstanceIdentifier': instance_id,
                    'Engine': engine,
                    'Endpoint': {'Address': '', 'Port': port},
                    'SecretArn': secret_arn,
                    'SecurityGroupId': sg_id,
                    'SubnetGroupName': subnet_group_name,
                    'Status': 'available',
                    'Note': 'Database already exists; returning known '
                    'resources.',
                    'ResourcesCreated': tracker.summary(),
                    'Idempotent': True,
                }
            )
        progress.failure(6, 'Create DB instance', db_result['error'])
        tracker.cleanup()
        return db_result
    db_instance = db_result['value'].get('DBInstance', {})
    tracker.track(
        'DBInstance', instance_id,
        make_delete_cleanup(
            call_rds, 'DeleteDBInstance',
            {'DBInstanceIdentifier': instance_id, 'SkipFinalSnapshot': True},
        ),
    )
    # The endpoint is usually absent until the instance finishes provisioning.
    endpoint = db_instance.get('Endpoint') or {}
    endpoint_address = endpoint.get('Address', '')
    endpoint_port = endpoint.get('Port', port)
    progress.success(6, 'Requested DB instance', resource_id=instance_id)

    # Step 7: backfill connection details into the secret so applications can use
    # it directly. The host may still be empty if the endpoint is not yet known.
    progress.report(7, 'Backfilling connection details into secret')
    secret_dict.update(
        {
            'engine': engine,
            'host': endpoint_address or '',
            'port': endpoint_port,
            'dbname': db_name,
        }
    )
    progress.detail('secretsmanager:PutSecretValue', SecretId=secret_arn)
    backfill_result = step(
        tracker, 'Backfill secret', call_secrets, 'PutSecretValue',
        {
            'SecretId': secret_arn,
            'SecretString': json.dumps(secret_dict),
        },
    )
    if not backfill_result['ok']:
        progress.failure(7, 'Backfill secret', backfill_result['error'])
        return backfill_result
    progress.success(7, 'Secret updated with connection details')

    return ok_result(
        {
            'DBInstanceIdentifier': instance_id,
            'Engine': engine,
            'Endpoint': {'Address': endpoint_address, 'Port': endpoint_port},
            'SecretArn': secret_arn,
            'SecurityGroupId': sg_id,
            'SubnetGroupName': subnet_group_name,
            'Status': db_instance.get('DBInstanceStatus', 'creating'),
            'Note': 'Database is being created. It may take 5-15 minutes to '
            'become available.',
            'ResourcesCreated': tracker.summary(),
            'Idempotent': False,
        }
    )


class CreateDatabaseCommand(BasicCommand):
    """Thin ``aws rds create-database`` command delegating to ``create_database``.

    Follows the composite-create command pattern: the class only builds the
    per-service clients and hands control to the module-level
    :func:`create_database` orchestration function, then emits its result. All
    logic lives in the function so it stays independently testable.
    """

    NAME = 'create-database'
    DESCRIPTION = (
        'Create an RDS database with networking, generated credentials in '
        'Secrets Manager, and a dedicated security group, replicating CDK\'s '
        'rds.DatabaseInstance L2 construct. Does not wait for the instance to '
        'become available.'
    )
    ARG_TABLE = [
        {
            'name': 'engine',
            'required': True,
            'cli_type_name': 'string',
            'help_text': 'Database engine: mysql, postgres, mariadb, '
            'oracle-ee, or sqlserver-ee.',
        },
        {
            'name': 'instance-class',
            'required': True,
            'cli_type_name': 'string',
            'help_text': 'DB instance class, e.g. db.t3.micro.',
        },
        {
            'name': 'vpc-id',
            'required': True,
            'cli_type_name': 'string',
            'help_text': 'VPC ID; private subnets are auto-discovered within '
            'it.',
        },
        {
            'name': 'db-name',
            'cli_type_name': 'string',
            'help_text': 'Database name. Defaults to a value derived from the '
            'engine.',
        },
        {
            'name': 'master-username',
            'cli_type_name': 'string',
            'default': 'admin',
            'help_text': 'Master username. Defaults to "admin".',
        },
        {
            'name': 'allocated-storage',
            'cli_type_name': 'integer',
            'default': 20,
            'help_text': 'Allocated storage in GiB. Defaults to 20.',
        },
        {
            'name': 'port',
            'cli_type_name': 'integer',
            'help_text': 'Listener port. Defaults to the engine default '
            '(mysql/mariadb 3306, postgres 5432, oracle-ee 1521, '
            'sqlserver-ee 1433).',
        },
        {
            'name': 'multi-az',
            'cli_type_name': 'boolean',
            'default': False,
            'help_text': 'Deploy the instance across multiple Availability '
            'Zones. Defaults to false.',
        },
        {
            'name': 'storage-encrypted',
            'cli_type_name': 'boolean',
            'default': True,
            'help_text': 'Encrypt storage at rest. Defaults to true; use '
            '--no-storage-encrypted to disable.',
        },
        {
            'name': 'ingress-cidr',
            'cli_type_name': 'string',
            'help_text': 'CIDR range allowed inbound to the database security '
            'group. Defaults to the VPC CIDR.',
        },
    ]

    def __init__(self, session, ec2_client=None, rds_client=None,
                 secrets_client=None, stdout=None, stderr=None):
        """Store the session and any injected clients/streams (for testing)."""
        super().__init__(session)
        self._ec2_client = ec2_client
        self._rds_client = rds_client
        self._secrets_client = secrets_client
        self._stdout = stdout
        self._stderr = stderr

    def _run_main(self, parsed_args, parsed_globals):
        """Validate input, build clients, orchestrate the create, emit result."""
        invalid = _validate_args(parsed_args)
        if invalid is not None:
            return emit_result(
                self.NAME, invalid, parsed_globals,
                self._session, self._stdout, self._stderr,
            )

        ec2_client = self._ec2_client or create_client(
            self._session, 'ec2', parsed_globals,
        )
        rds_client = self._rds_client or create_client(
            self._session, 'rds', parsed_globals,
        )
        secrets_client = self._secrets_client or create_client(
            self._session, 'secretsmanager', parsed_globals,
        )
        call_ec2 = lambda op, params: call(ec2_client, op, params)
        call_rds = lambda op, params: call(rds_client, op, params)
        call_secrets = lambda op, params: call(secrets_client, op, params)

        result = create_database(
            call_ec2, call_rds, call_secrets, parsed_args, stderr=self._stderr,
        )
        return emit_result(
            self.NAME, result, parsed_globals,
            self._session, self._stdout, self._stderr,
        )


def register(event_handlers):
    """Register ``create-database`` on the rds command table.

    Self-contained registration: hooks ``building-command-table.rds`` and injects
    the command when the rds command table is built.
    """
    event_handlers.register('building-command-table.rds', _inject_command)


def _inject_command(command_table, session, **kwargs):
    """Add the ``create-database`` command to the rds command table."""
    command_table['create-database'] = CreateDatabaseCommand(session)
