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
"""High-order ``ec2 create-vpc-network`` command.

Replicates CDK's ``ec2.Vpc`` L2 construct as a single CLI verb: it creates a
complete, routable VPC -- the VPC itself, an internet gateway, public and
private subnets spread across Availability Zones, NAT gateways, and the route
tables wiring them together -- in one multi-step, write-side operation.

Every AWS call is routed through the ``compositecreate`` call bridge so errors
come back as data, and every resource created is recorded in a
:class:`~awscli.customizations.compositecreate.helpers.ResourceTracker` with a
cleanup callable, so a failure partway through rolls everything back in reverse
(LIFO) order rather than stranding a half-built VPC.

The command is self-contained: it registers itself on the
``building-command-table.ec2`` event via :func:`register`.
"""

from __future__ import annotations

import ipaddress
import logging
import time

from awscli.customizations.commands import BasicCommand
from awscli.customizations.compositecreate.cleanup import (
    make_delete_cleanup,
    make_detach_cleanup,
)
from awscli.customizations.compositecreate.helpers import (
    INVALID_INPUT,
    NOT_READY,
    TIMEOUT,
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

# Default route destination for both the public (IGW) and private (NAT) routes.
_DEFAULT_ROUTE = '0.0.0.0/0'
# NAT gateway state polling: 40 attempts * 15s = up to 10 minutes.
_NAT_POLL_ATTEMPTS = 40
_NAT_POLL_DELAY = 15


class CreateVpcNetworkCommand(BasicCommand):
    """Thin CLI verb that delegates to :func:`create_vpc_network`."""

    NAME = 'create-vpc-network'
    DESCRIPTION = (
        'Create a complete VPC network -- VPC, internet gateway, public and '
        'private subnets across Availability Zones, NAT gateways, and route '
        'tables -- in one command, rolling back every created resource if any '
        'step fails.'
    )
    ARG_TABLE = [
        {
            'name': 'cidr',
            'cli_type_name': 'string',
            'default': '10.0.0.0/16',
            'help_text': 'The IPv4 CIDR block for the VPC. Defaults to '
            '10.0.0.0/16.',
        },
        {
            'name': 'az-count',
            'cli_type_name': 'integer',
            'default': 2,
            'help_text': 'Number of Availability Zones to span (2 or 3). '
            'Defaults to 2.',
        },
        {
            'name': 'nat-gateways',
            'cli_type_name': 'integer',
            'default': 1,
            'help_text': 'Number of NAT gateways to create (0, 1, or equal to '
            '--az-count). Defaults to 1.',
        },
        {
            'name': 'vpc-name',
            'cli_type_name': 'string',
            'help_text': 'Optional Name tag applied to the VPC and used as a '
            'prefix on the Name tags of the resources created.',
        },
        {
            'name': 'enable-dns-hostnames',
            'action': 'store_true',
            'default': True,
            'dest': 'enable_dns_hostnames',
            'group_name': 'enable_dns_hostnames',
            'help_text': 'Enable DNS hostnames in the VPC (default).',
        },
        {
            'name': 'no-enable-dns-hostnames',
            'action': 'store_false',
            'dest': 'enable_dns_hostnames',
            'group_name': 'enable_dns_hostnames',
            'help_text': 'Disable DNS hostnames in the VPC.',
        },
        {
            'name': 'enable-dns-support',
            'action': 'store_true',
            'default': True,
            'dest': 'enable_dns_support',
            'group_name': 'enable_dns_support',
            'help_text': 'Enable DNS support in the VPC (default).',
        },
        {
            'name': 'no-enable-dns-support',
            'action': 'store_false',
            'dest': 'enable_dns_support',
            'group_name': 'enable_dns_support',
            'help_text': 'Disable DNS support in the VPC.',
        },
    ]

    def __init__(self, session, client=None, stdout=None, stderr=None):
        """Store the session and optional injected client/streams (testing)."""
        super().__init__(session)
        self._provided_client = client
        self._stdout = stdout
        self._stderr = stderr

    def _run_main(self, parsed_args, parsed_globals):
        """Create the ec2 client and delegate to the orchestration function."""
        client = self._provided_client or create_client(
            self._session, 'ec2', parsed_globals,
        )
        call_op = lambda op, params: call(client, op, params)

        result = create_vpc_network(call_op, parsed_args, stderr=self._stderr)

        return emit_result(
            self.NAME, result, parsed_globals,
            self._session, self._stdout, self._stderr,
        )


def _coerce_int(val):
    """Return ``val`` as an int when it is a numeric string, else unchanged.

    Integer arguments normally arrive already typed, but a defensive coercion
    keeps the validators (which reject non-ints) from rejecting a stringified
    integer. A non-numeric string is returned unchanged so validation reports it.
    """
    if isinstance(val, str):
        try:
            return int(val)
        except ValueError:
            return val
    return val


def _name(base, vpc_name):
    """Return ``base`` prefixed with the VPC name when one was supplied."""
    if vpc_name:
        return '%s-%s' % (vpc_name, base)
    return base


def _az_label(az_name):
    """Return the short AZ suffix used in resource names (e.g. ``1a``)."""
    return az_name[-2:]


def _tag_spec(resource_type, name):
    """Build a ``TagSpecifications`` entry setting the Name tag, or ``None``."""
    if not name:
        return None
    return [
        {
            'ResourceType': resource_type,
            'Tags': [{'Key': 'Name', 'Value': name}],
        }
    ]


def _with_tags(params, resource_type, name):
    """Add a Name ``TagSpecifications`` to ``params`` in place when applicable."""
    spec = _tag_spec(resource_type, name)
    if spec is not None:
        params['TagSpecifications'] = spec
    return params


def _plan_subnets(cidr, az_count):
    """Compute the public and private subnet CIDRs for the VPC.

    The VPC CIDR is split into two equal halves -- the first for public subnets,
    the second for private -- and ``az_count`` equal subnets are carved out of
    each half. The subnet size reproduces the documented layout for a /16 VPC
    (/20 subnets for 2 AZs, /19 for 3 AZs) and scales with the VPC prefix length.
    Returns ``(public_cidrs, private_cidrs)`` as lists of CIDR strings. Raises
    :class:`ValueError` when the block is too small to subdivide.
    """
    network = ipaddress.ip_network(cidr, strict=False)
    half_prefix = network.prefixlen + 1
    halves = list(network.subnets(new_prefix=half_prefix))
    public_half, private_half = halves[0], halves[1]
    # Extra bits below the VPC prefix for each subnet: 4 (-> /20 for a /16) when
    # spanning 2 AZs, 3 (-> /19 for a /16) for 3 AZs. This matches the documented
    # example layout exactly and scales with the VPC prefix.
    extra_bits = 4 if az_count == 2 else 3
    subnet_prefix = network.prefixlen + extra_bits
    if subnet_prefix > 28:
        raise ValueError(
            'VPC CIDR %s is too small to allocate /%d subnets for %d AZs.'
            % (cidr, subnet_prefix, az_count)
        )
    public = [
        str(net)
        for net in list(public_half.subnets(new_prefix=subnet_prefix))[:az_count]
    ]
    private = [
        str(net)
        for net in list(private_half.subnets(new_prefix=subnet_prefix))[
            :az_count
        ]
    ]
    if len(public) < az_count or len(private) < az_count:
        raise ValueError(
            'VPC CIDR %s cannot fit %d subnets in each half.' % (cidr, az_count)
        )
    return public, private


def _validate_args(cidr, az_count, nat_gateways, vpc_name):
    """Validate the parsed arguments, returning an error result or ``None``.

    Returns a pre-dispatch :func:`invalid_input` result (exit code 252) on the
    first problem found, or ``None`` when every argument is acceptable.
    """
    problem = check_non_empty_str('--cidr', cidr)
    if problem:
        return invalid_input(problem)
    try:
        ipaddress.ip_network(cidr, strict=False)
    except ValueError as exc:
        return invalid_input(
            '--cidr is not a valid IPv4 CIDR block: %s' % exc
        )
    problem = check_positive_int('--az-count', az_count, lo=2, hi=3)
    if problem:
        return invalid_input(problem)
    problem = check_positive_int('--nat-gateways', nat_gateways, lo=0)
    if problem:
        return invalid_input(problem)
    if nat_gateways not in (0, 1, az_count):
        return invalid_input(
            '--nat-gateways must be 0, 1, or equal to --az-count (%d), got %d.'
            % (az_count, nat_gateways)
        )
    problem = check_non_empty_str('--vpc-name', vpc_name, required=False)
    if problem:
        return invalid_input(problem)
    return None


def _wait_for_nat_available(call_op, nat_gateway_id, progress, step_num):
    """Poll ``DescribeNatGateways`` until the NAT gateway is available.

    NAT gateway creation is asynchronous: a route cannot target the gateway
    until it reaches the ``available`` state. Returns an ok result once it is
    available, or an error result on an API failure, a terminal non-available
    state, or a timeout. Reports progress on the shared reporter.
    """
    label = 'Waiting for NAT gateway %s' % nat_gateway_id
    progress.report(step_num, label)
    for _ in range(_NAT_POLL_ATTEMPTS):
        res = call_op(
            'DescribeNatGateways', {'NatGatewayIds': [nat_gateway_id]}
        )
        if not res['ok']:
            progress.failure(step_num, label, res.get('error'))
            return res
        gateways = res['value'].get('NatGateways', [])
        state = gateways[0].get('State') if gateways else None
        if state == 'available':
            progress.success(step_num, label)
            return ok_result(
                {'NatGatewayId': nat_gateway_id, 'State': state}
            )
        if state in ('failed', 'deleting', 'deleted'):
            res = err_result(
                NOT_READY,
                'NAT gateway %s entered terminal state %r while waiting for it '
                'to become available.' % (nat_gateway_id, state),
            )
            progress.failure(step_num, label, res['error'])
            return res
        time.sleep(_NAT_POLL_DELAY)
    res = err_result(
        TIMEOUT,
        'Timed out waiting for NAT gateway %s to become available.'
        % nat_gateway_id,
    )
    progress.failure(step_num, label, res['error'])
    return res


def _make_nat_delete_cleanup(call_op, nat_gateway_id):
    """Build a best-effort cleanup that deletes a NAT gateway and waits.

    NAT gateway deletion is asynchronous and the Elastic IP it uses cannot be
    released until the gateway has finished deleting. This cleanup deletes the
    gateway and then polls until it reports ``deleted`` (or disappears), so the
    LIFO rollback can release the EIP next. Best-effort: any error is logged and
    swallowed, never raised.
    """
    def _cleanup():
        try:
            logger.debug('Cleanup delete NAT gateway %s', nat_gateway_id)
            res = call_op(
                'DeleteNatGateway', {'NatGatewayId': nat_gateway_id}
            )
            for _ in range(_NAT_POLL_ATTEMPTS):
                desc = call_op(
                    'DescribeNatGateways',
                    {'NatGatewayIds': [nat_gateway_id]},
                )
                if not desc.get('ok'):
                    break
                gateways = desc['value'].get('NatGateways', [])
                state = gateways[0].get('State') if gateways else None
                if state in ('deleted', None):
                    break
                time.sleep(_NAT_POLL_DELAY)
            return res
        except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
            logger.warning(
                'Cleanup delete NAT gateway %s failed: %r',
                nat_gateway_id,
                exc,
            )
            return None

    return _cleanup


def _run_step(progress, step_num, label, tracker, call_op, operation, params,
              detail_op=None, detail_params=None):
    """Report, run one tracked API step, and report its outcome.

    Wraps :func:`step` (which rolls back the tracker on failure) with the
    per-step progress reporting. Returns the call's result envelope; the caller
    inspects ``result['ok']`` and, on success, records the created resource.

    When ``detail_op`` is given, a compact ``  → detail_op key=value`` line
    naming the API call and its key parameters (``detail_params``) is emitted
    between the step report and the call, for agent-friendly diagnostics.
    """
    progress.report(step_num, label)
    if detail_op is not None:
        progress.detail(detail_op, **(detail_params or {}))
    result = step(tracker, label, call_op, operation, params)
    if result['ok']:
        progress.success(step_num, label)
    else:
        progress.failure(step_num, label, result.get('error'))
    return result


def _tag_value(tags, key='Name'):
    """Return the value of tag ``key`` from a list of ``{Key, Value}`` tags."""
    for tag in tags or []:
        if tag.get('Key') == key:
            return tag.get('Value')
    return None


def _describe_existing_vpc(call_op, vpc_name, cidr, progress):
    """Describe an existing VPC matching ``vpc_name``+``cidr``, or return ``None``.

    Idempotency fast path. A previous ``create-vpc-network`` run with the same
    ``--vpc-name`` tags its VPC ``Name=<vpc_name>-vpc`` (see :func:`_name`), so
    this looks for a VPC with that exact Name tag and matching CIDR block. When
    one is found, its current resources are gathered with read-only Describe
    calls and returned as an ok result in the same shape as a fresh create (with
    ``Idempotent: True`` added), letting the command run twice with the same
    arguments without creating a duplicate network. Returns:

    * ``None`` -- no matching VPC exists (or the initial lookup itself failed);
      the caller proceeds with the normal create flow.
    * an ok result -- a matching VPC was found and described.
    * an error result -- a matching VPC was found but its resources could not be
      described; the caller surfaces the error rather than creating a duplicate.

    Progress is reported on ``progress`` so the fast path shows what it found.
    """
    # Match the Name tag the create path actually applies to the VPC, not the
    # bare --vpc-name, otherwise a second run would never find the first run's
    # VPC and idempotency would be defeated.
    vpc_tag_name = _name('vpc', vpc_name)
    label = 'Checking for existing VPC %r' % vpc_tag_name
    progress.report(1, label)
    progress.detail('ec2:DescribeVpcs', Name=vpc_tag_name, CidrBlock=cidr)
    check = call_op(
        'DescribeVpcs',
        {
            'Filters': [
                {'Name': 'tag:Name', 'Values': [vpc_tag_name]},
                {'Name': 'cidr-block', 'Values': [cidr]},
            ]
        },
    )
    if not check['ok'] or not check['value'].get('Vpcs'):
        progress.success(1, 'No existing VPC found; creating a new one')
        return None
    vpc = check['value']['Vpcs'][0]
    vpc_id = vpc['VpcId']
    progress.success(1, label, vpc_id)

    # Subnets: classify by MapPublicIpOnLaunch, which the create path sets on
    # public subnets only. Sort by AZ then CIDR for deterministic output order.
    progress.report(2, 'Describing subnets')
    progress.detail('ec2:DescribeSubnets', VpcId=vpc_id)
    subnets_res = call_op(
        'DescribeSubnets',
        {'Filters': [{'Name': 'vpc-id', 'Values': [vpc_id]}]},
    )
    if not subnets_res['ok']:
        progress.failure(2, 'Describing subnets', subnets_res.get('error'))
        return subnets_res
    public_subnets = []
    private_subnets = []
    for subnet in sorted(
        subnets_res['value'].get('Subnets', []),
        key=lambda s: (s.get('AvailabilityZone', ''), s.get('CidrBlock', '')),
    ):
        entry = {
            'SubnetId': subnet['SubnetId'],
            'AvailabilityZone': subnet.get('AvailabilityZone'),
            'CidrBlock': subnet.get('CidrBlock'),
        }
        if subnet.get('MapPublicIpOnLaunch'):
            public_subnets.append(entry)
        else:
            private_subnets.append(entry)
    progress.success(
        2, 'Describing subnets',
        '%d public, %d private' % (len(public_subnets), len(private_subnets)),
    )

    # Internet gateway attached to the VPC.
    progress.report(3, 'Describing internet gateway')
    progress.detail('ec2:DescribeInternetGateways', VpcId=vpc_id)
    igw_res = call_op(
        'DescribeInternetGateways',
        {'Filters': [{'Name': 'attachment.vpc-id', 'Values': [vpc_id]}]},
    )
    if not igw_res['ok']:
        progress.failure(
            3, 'Describing internet gateway', igw_res.get('error')
        )
        return igw_res
    igws = igw_res['value'].get('InternetGateways', [])
    igw_id = igws[0]['InternetGatewayId'] if igws else None
    progress.success(3, 'Describing internet gateway', igw_id)

    # Route tables: classify by the Name tag the create path applies -- the
    # public table is named "<vpc_name>-public", private ones
    # "<vpc_name>-private-<az>".
    progress.report(4, 'Describing route tables')
    progress.detail('ec2:DescribeRouteTables', VpcId=vpc_id)
    rtb_res = call_op(
        'DescribeRouteTables',
        {'Filters': [{'Name': 'vpc-id', 'Values': [vpc_id]}]},
    )
    if not rtb_res['ok']:
        progress.failure(4, 'Describing route tables', rtb_res.get('error'))
        return rtb_res
    public_rtb_id = None
    private_rtbs = []
    public_rtb_name = _name('public', vpc_name)
    private_rtb_prefix = _name('private', vpc_name)
    for rtb in rtb_res['value'].get('RouteTables', []):
        name = _tag_value(rtb.get('Tags'))
        if name == public_rtb_name:
            public_rtb_id = rtb['RouteTableId']
        elif name and name.startswith(private_rtb_prefix):
            private_rtbs.append((name, rtb['RouteTableId']))
    private_rtb_ids = [rtb_id for _, rtb_id in sorted(private_rtbs)]
    progress.success(4, 'Describing route tables')

    # NAT gateways in the VPC. NOTE: DescribeNatGateways uses the singular
    # ``Filter`` request key, unlike the other Describe calls (``Filters``).
    progress.report(5, 'Describing NAT gateways')
    progress.detail('ec2:DescribeNatGateways', VpcId=vpc_id)
    nat_res = call_op(
        'DescribeNatGateways',
        {'Filter': [{'Name': 'vpc-id', 'Values': [vpc_id]}]},
    )
    if not nat_res['ok']:
        progress.failure(5, 'Describing NAT gateways', nat_res.get('error'))
        return nat_res
    nat_gateways_out = []
    for nat in nat_res['value'].get('NatGateways', []):
        # Skip gateways that are gone or on their way out; they are not part of
        # the live network state.
        if nat.get('State') in ('deleting', 'deleted', 'failed'):
            continue
        addresses = nat.get('NatGatewayAddresses', [])
        public_ip = addresses[0].get('PublicIp') if addresses else None
        nat_gateways_out.append(
            {
                'NatGatewayId': nat['NatGatewayId'],
                'PublicIp': public_ip,
                'SubnetId': nat.get('SubnetId'),
            }
        )
    progress.success(
        5, 'Describing NAT gateways', '%d found' % len(nat_gateways_out)
    )

    return ok_result(
        {
            'VpcId': vpc_id,
            'InternetGatewayId': igw_id,
            'PublicSubnets': public_subnets,
            'PrivateSubnets': private_subnets,
            'NatGateways': nat_gateways_out,
            'RouteTables': {
                'Public': public_rtb_id,
                'Private': private_rtb_ids,
            },
            # Nothing was created on this run; keep the key shape identical to a
            # fresh create (ResourceTracker.summary()).
            'ResourcesCreated': {'created_resources': [], 'count': 0},
            'Idempotent': True,
        }
    )


def create_vpc_network(call_op, parsed_args, stderr=None):
    """Orchestrate creation of a complete VPC network, rolling back on failure.

    Runs the full sequence -- VPC, DNS attributes, internet gateway, subnets,
    public and private route tables, NAT gateways, and routes -- through the
    call bridge, tracking every created resource for LIFO rollback. Returns an
    ok result with the created resource summary on success, or the first failing
    step's error result (after rollback) on failure.
    """
    cidr = parsed_args.cidr
    az_count = _coerce_int(parsed_args.az_count)
    nat_gateways = _coerce_int(parsed_args.nat_gateways)
    vpc_name = parsed_args.vpc_name
    enable_dns_hostnames = parsed_args.enable_dns_hostnames
    enable_dns_support = parsed_args.enable_dns_support

    invalid = _validate_args(cidr, az_count, nat_gateways, vpc_name)
    if invalid is not None:
        return invalid

    # Idempotency fast path: when --vpc-name is given, a VPC with that Name tag
    # and the same CIDR may already exist from a previous run with identical
    # arguments. If so, describe its current resources and return them instead
    # of creating a duplicate network, so the command is safe to run twice.
    if vpc_name:
        existing = _describe_existing_vpc(
            call_op, vpc_name, cidr,
            ProgressReporter(total_steps=5, stderr=stderr),
        )
        if existing is not None:
            return existing

    try:
        public_cidrs, private_cidrs = _plan_subnets(cidr, az_count)
    except ValueError as exc:
        return invalid_input(str(exc))

    # Discover real AZ names for the region before creating anything; nothing is
    # tracked yet, so a failure here simply returns the error.
    az_res = call_op(
        'DescribeAvailabilityZones',
        {'Filters': [{'Name': 'state', 'Values': ['available']}]},
    )
    if not az_res['ok']:
        return az_res
    zones = [
        zone['ZoneName']
        for zone in az_res['value'].get('AvailabilityZones', [])
    ]
    if len(zones) < az_count:
        return err_result(
            INVALID_INPUT,
            'Region has %d available Availability Zone(s) but --az-count is %d.'
            % (len(zones), az_count),
        )
    azs = zones[:az_count]

    tracker = ResourceTracker()
    # Two DNS attribute modifications are always issued (one per attribute).
    total_steps = (
        1  # CreateVpc
        + 2  # DNS support + hostnames
        + 1  # CreateInternetGateway
        + 1  # AttachInternetGateway
        + 2 * az_count  # public + private subnets
        + 1  # public route table
        + 1  # public default route
        + az_count  # public route table associations
        + 3 * nat_gateways  # per NAT: EIP, NAT gateway, wait
        + az_count  # private route tables
        + (az_count if nat_gateways > 0 else 0)  # private default routes
        + az_count  # private route table associations
    )
    progress = ProgressReporter(total_steps=total_steps, stderr=stderr)
    step_num = 0

    # 1. Create the VPC.
    step_num += 1
    vpc_params = _with_tags(
        {'CidrBlock': cidr}, 'vpc', _name('vpc', vpc_name) if vpc_name else None
    )
    result = _run_step(
        progress, step_num, 'Creating VPC', tracker, call_op,
        'CreateVpc', vpc_params,
        detail_op='ec2:CreateVpc', detail_params={'CidrBlock': cidr},
    )
    if not result['ok']:
        return result
    vpc_id = result['value']['Vpc']['VpcId']
    tracker.track(
        'vpc', vpc_id,
        make_delete_cleanup(call_op, 'DeleteVpc', {'VpcId': vpc_id}),
    )

    # 2. Set DNS attributes (support first, then hostnames, which depends on it).
    step_num += 1
    result = _run_step(
        progress, step_num, 'Setting DNS support', tracker, call_op,
        'ModifyVpcAttribute',
        {'VpcId': vpc_id, 'EnableDnsSupport': {'Value': enable_dns_support}},
        detail_op='ec2:ModifyVpcAttribute',
        detail_params={'VpcId': vpc_id, 'EnableDnsSupport': enable_dns_support},
    )
    if not result['ok']:
        return result
    step_num += 1
    result = _run_step(
        progress, step_num, 'Setting DNS hostnames', tracker, call_op,
        'ModifyVpcAttribute',
        {'VpcId': vpc_id,
         'EnableDnsHostnames': {'Value': enable_dns_hostnames}},
        detail_op='ec2:ModifyVpcAttribute',
        detail_params={'VpcId': vpc_id,
                       'EnableDnsHostnames': enable_dns_hostnames},
    )
    if not result['ok']:
        return result

    # 3. Create the internet gateway.
    step_num += 1
    result = _run_step(
        progress, step_num, 'Creating internet gateway', tracker, call_op,
        'CreateInternetGateway',
        _with_tags({}, 'internet-gateway', _name('igw', vpc_name)),
        detail_op='ec2:CreateInternetGateway',
    )
    if not result['ok']:
        return result
    igw_id = result['value']['InternetGateway']['InternetGatewayId']
    tracker.track(
        'internet-gateway', igw_id,
        make_delete_cleanup(
            call_op, 'DeleteInternetGateway',
            {'InternetGatewayId': igw_id},
        ),
    )

    # 4. Attach the internet gateway to the VPC.
    step_num += 1
    result = _run_step(
        progress, step_num, 'Attaching internet gateway', tracker, call_op,
        'AttachInternetGateway',
        {'InternetGatewayId': igw_id, 'VpcId': vpc_id},
        detail_op='ec2:AttachInternetGateway',
        detail_params={'InternetGatewayId': igw_id, 'VpcId': vpc_id},
    )
    if not result['ok']:
        return result
    tracker.track(
        'igw-attachment', igw_id,
        make_detach_cleanup(
            call_op, 'DetachInternetGateway',
            {'InternetGatewayId': igw_id, 'VpcId': vpc_id},
        ),
    )

    # 5. Create the public and private subnets per AZ.
    public_subnets = []
    private_subnets = []
    for index, az_name in enumerate(azs):
        label = _az_label(az_name)

        step_num += 1
        result = _run_step(
            progress, step_num, 'Creating public subnet in %s' % az_name,
            tracker, call_op, 'CreateSubnet',
            _with_tags(
                {'VpcId': vpc_id, 'CidrBlock': public_cidrs[index],
                 'AvailabilityZone': az_name},
                'subnet', _name('public-%s' % label, vpc_name),
            ),
            detail_op='ec2:CreateSubnet',
            detail_params={'CidrBlock': public_cidrs[index],
                           'AvailabilityZone': az_name},
        )
        if not result['ok']:
            return result
        public_subnet_id = result['value']['Subnet']['SubnetId']
        tracker.track(
            'subnet', public_subnet_id,
            make_delete_cleanup(
                call_op, 'DeleteSubnet', {'SubnetId': public_subnet_id},
            ),
        )
        # Mark the public subnet for auto-assign public IP, matching CDK's
        # ec2.Vpc behaviour.  Without this, instances launched into "public"
        # subnets would get no public IP by default.
        progress.detail(
            'ec2:ModifySubnetAttribute',
            SubnetId=public_subnet_id,
            MapPublicIpOnLaunch=True,
        )
        call_op('ModifySubnetAttribute', {
            'SubnetId': public_subnet_id,
            'MapPublicIpOnLaunch': {'Value': True},
        })
        public_subnets.append(
            {'SubnetId': public_subnet_id, 'AvailabilityZone': az_name,
             'CidrBlock': public_cidrs[index]}
        )

        step_num += 1
        result = _run_step(
            progress, step_num, 'Creating private subnet in %s' % az_name,
            tracker, call_op, 'CreateSubnet',
            _with_tags(
                {'VpcId': vpc_id, 'CidrBlock': private_cidrs[index],
                 'AvailabilityZone': az_name},
                'subnet', _name('private-%s' % label, vpc_name),
            ),
            detail_op='ec2:CreateSubnet',
            detail_params={'CidrBlock': private_cidrs[index],
                           'AvailabilityZone': az_name},
        )
        if not result['ok']:
            return result
        private_subnet_id = result['value']['Subnet']['SubnetId']
        tracker.track(
            'subnet', private_subnet_id,
            make_delete_cleanup(
                call_op, 'DeleteSubnet', {'SubnetId': private_subnet_id},
            ),
        )
        private_subnets.append(
            {'SubnetId': private_subnet_id, 'AvailabilityZone': az_name,
             'CidrBlock': private_cidrs[index]}
        )

    # 6. Create the public route table.
    step_num += 1
    result = _run_step(
        progress, step_num, 'Creating public route table', tracker, call_op,
        'CreateRouteTable',
        _with_tags({'VpcId': vpc_id}, 'route-table',
                   _name('public', vpc_name)),
        detail_op='ec2:CreateRouteTable', detail_params={'VpcId': vpc_id},
    )
    if not result['ok']:
        return result
    public_rtb_id = result['value']['RouteTable']['RouteTableId']
    tracker.track(
        'route-table', public_rtb_id,
        make_delete_cleanup(
            call_op, 'DeleteRouteTable', {'RouteTableId': public_rtb_id},
        ),
    )

    # 7. Default route from the public route table to the internet gateway.
    step_num += 1
    result = _run_step(
        progress, step_num, 'Creating public default route', tracker, call_op,
        'CreateRoute',
        {'RouteTableId': public_rtb_id,
         'DestinationCidrBlock': _DEFAULT_ROUTE, 'GatewayId': igw_id},
        detail_op='ec2:CreateRoute',
        detail_params={'RouteTableId': public_rtb_id, 'GatewayId': igw_id},
    )
    if not result['ok']:
        return result
    tracker.track(
        'route', '%s:%s' % (public_rtb_id, _DEFAULT_ROUTE),
        make_delete_cleanup(
            call_op, 'DeleteRoute',
            {'RouteTableId': public_rtb_id,
             'DestinationCidrBlock': _DEFAULT_ROUTE},
        ),
    )

    # 8. Associate the public route table with every public subnet.
    for subnet in public_subnets:
        step_num += 1
        result = _run_step(
            progress, step_num,
            'Associating public route table with %s' % subnet['SubnetId'],
            tracker, call_op, 'AssociateRouteTable',
            {'RouteTableId': public_rtb_id, 'SubnetId': subnet['SubnetId']},
            detail_op='ec2:AssociateRouteTable',
            detail_params={'RouteTableId': public_rtb_id,
                           'SubnetId': subnet['SubnetId']},
        )
        if not result['ok']:
            return result
        association_id = result['value']['AssociationId']
        tracker.track(
            'rtb-association', association_id,
            make_detach_cleanup(
                call_op, 'DisassociateRouteTable',
                {'AssociationId': association_id},
            ),
        )

    # 9. Create NAT gateways (each needs an Elastic IP and a public subnet).
    nat_gateways_out = []
    for nat_index in range(nat_gateways):
        if nat_gateways == az_count:
            host_subnet = public_subnets[nat_index]
        else:
            host_subnet = public_subnets[0]
        label = _az_label(host_subnet['AvailabilityZone'])

        step_num += 1
        result = _run_step(
            progress, step_num, 'Allocating Elastic IP for NAT gateway',
            tracker, call_op, 'AllocateAddress',
            _with_tags({'Domain': 'vpc'}, 'elastic-ip',
                       _name('eip-%s' % label, vpc_name)),
            detail_op='ec2:AllocateAddress', detail_params={'Domain': 'vpc'},
        )
        if not result['ok']:
            return result
        allocation_id = result['value']['AllocationId']
        public_ip = result['value'].get('PublicIp')
        tracker.track(
            'elastic-ip', allocation_id,
            make_delete_cleanup(
                call_op, 'ReleaseAddress', {'AllocationId': allocation_id},
            ),
        )

        step_num += 1
        result = _run_step(
            progress, step_num,
            'Creating NAT gateway in %s' % host_subnet['SubnetId'],
            tracker, call_op, 'CreateNatGateway',
            _with_tags(
                {'SubnetId': host_subnet['SubnetId'],
                 'AllocationId': allocation_id},
                'natgateway', _name('nat-%s' % label, vpc_name),
            ),
            detail_op='ec2:CreateNatGateway',
            detail_params={'SubnetId': host_subnet['SubnetId'],
                           'AllocationId': allocation_id},
        )
        if not result['ok']:
            return result
        nat_gateway_id = result['value']['NatGateway']['NatGatewayId']
        tracker.track(
            'natgateway', nat_gateway_id,
            _make_nat_delete_cleanup(call_op, nat_gateway_id),
        )

        # Wait for the NAT gateway to become available before routing to it.
        step_num += 1
        result = step(
            tracker, 'Waiting for NAT gateway %s' % nat_gateway_id,
            _wait_for_nat_available, call_op, nat_gateway_id, progress,
            step_num,
        )
        if not result['ok']:
            return result
        nat_gateways_out.append(
            {'NatGatewayId': nat_gateway_id, 'PublicIp': public_ip,
             'SubnetId': host_subnet['SubnetId']}
        )

    # 10. Per AZ: a private route table, a default route to its NAT, and the
    #     association with the private subnet.
    private_rtb_ids = []
    for index, subnet in enumerate(private_subnets):
        label = _az_label(subnet['AvailabilityZone'])

        step_num += 1
        result = _run_step(
            progress, step_num,
            'Creating private route table for %s' % subnet['AvailabilityZone'],
            tracker, call_op, 'CreateRouteTable',
            _with_tags({'VpcId': vpc_id}, 'route-table',
                       _name('private-%s' % label, vpc_name)),
            detail_op='ec2:CreateRouteTable', detail_params={'VpcId': vpc_id},
        )
        if not result['ok']:
            return result
        private_rtb_id = result['value']['RouteTable']['RouteTableId']
        tracker.track(
            'route-table', private_rtb_id,
            make_delete_cleanup(
                call_op, 'DeleteRouteTable',
                {'RouteTableId': private_rtb_id},
            ),
        )
        private_rtb_ids.append(private_rtb_id)

        nat_gateway = _nat_for_index(nat_gateways_out, index)
        if nat_gateway is not None:
            step_num += 1
            result = _run_step(
                progress, step_num,
                'Creating private default route to NAT gateway',
                tracker, call_op, 'CreateRoute',
                {'RouteTableId': private_rtb_id,
                 'DestinationCidrBlock': _DEFAULT_ROUTE,
                 'NatGatewayId': nat_gateway['NatGatewayId']},
                detail_op='ec2:CreateRoute',
                detail_params={'RouteTableId': private_rtb_id,
                               'NatGatewayId': nat_gateway['NatGatewayId']},
            )
            if not result['ok']:
                return result
            tracker.track(
                'route', '%s:%s' % (private_rtb_id, _DEFAULT_ROUTE),
                make_delete_cleanup(
                    call_op, 'DeleteRoute',
                    {'RouteTableId': private_rtb_id,
                     'DestinationCidrBlock': _DEFAULT_ROUTE},
                ),
            )

        step_num += 1
        result = _run_step(
            progress, step_num,
            'Associating private route table with %s' % subnet['SubnetId'],
            tracker, call_op, 'AssociateRouteTable',
            {'RouteTableId': private_rtb_id, 'SubnetId': subnet['SubnetId']},
            detail_op='ec2:AssociateRouteTable',
            detail_params={'RouteTableId': private_rtb_id,
                           'SubnetId': subnet['SubnetId']},
        )
        if not result['ok']:
            return result
        association_id = result['value']['AssociationId']
        tracker.track(
            'rtb-association', association_id,
            make_detach_cleanup(
                call_op, 'DisassociateRouteTable',
                {'AssociationId': association_id},
            ),
        )

    return ok_result(
        {
            'VpcId': vpc_id,
            'InternetGatewayId': igw_id,
            'PublicSubnets': public_subnets,
            'PrivateSubnets': private_subnets,
            'NatGateways': nat_gateways_out,
            'RouteTables': {
                'Public': public_rtb_id,
                'Private': private_rtb_ids,
            },
            'ResourcesCreated': tracker.summary(),
            'Idempotent': False,
        }
    )


def _nat_for_index(nat_gateways_out, az_index):
    """Return the NAT gateway a given AZ's private subnet should route through.

    With no NAT gateways, returns ``None`` (the private subnets stay isolated).
    With a single NAT gateway, every private subnet shares it. With one per AZ,
    the private subnet routes through the NAT gateway in its own AZ.
    """
    if not nat_gateways_out:
        return None
    if len(nat_gateways_out) == 1:
        return nat_gateways_out[0]
    return nat_gateways_out[az_index]


def register(event_handlers):
    """Register the create-vpc-network command on the ec2 command table."""
    event_handlers.register(
        'building-command-table.ec2', _inject_command,
    )


def _inject_command(command_table, session, **kwargs):
    """Inject ``create-vpc-network`` into the ec2 command table."""
    command_table['create-vpc-network'] = CreateVpcNetworkCommand(session)
