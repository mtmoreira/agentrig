"""Generate locked Buck2 Python wheel targets from uv.lock."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import tomllib
from typing import cast
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class SupportedPlatform:
    """One host platform supported by AgentRig's Buck Python toolchain."""

    key: str
    os: str
    cpu: str
    wheel_pattern: re.Pattern[str]

    @property
    def os_setting(self) -> str:
        return f"prelude//os:{self.os}"

    @property
    def cpu_setting(self) -> str:
        return f"prelude//cpu:{self.cpu}"

    @property
    def os_constraint(self) -> str:
        return f"prelude//os/constraints:{self.os}"

    @property
    def cpu_constraint(self) -> str:
        return f"prelude//cpu/constraints:{self.cpu}"


SUPPORTED_PLATFORMS = (
    SupportedPlatform(
        key="linux_arm64",
        os="linux",
        cpu="arm64",
        wheel_pattern=re.compile(r"-manylinux[^/]*_aarch64\.whl$"),
    ),
    SupportedPlatform(
        key="linux_x86_64",
        os="linux",
        cpu="x86_64",
        wheel_pattern=re.compile(r"-manylinux[^/]*_x86_64\.whl$"),
    ),
    SupportedPlatform(
        key="macos_arm64",
        os="macos",
        cpu="arm64",
        wheel_pattern=re.compile(r"-macosx_[^/]*_arm64\.whl$"),
    ),
    SupportedPlatform(
        key="macos_x86_64",
        os="macos",
        cpu="x86_64",
        wheel_pattern=re.compile(r"-macosx_[^/]*_x86_64\.whl$"),
    ),
    SupportedPlatform(
        key="windows_arm64",
        os="windows",
        cpu="arm64",
        wheel_pattern=re.compile(r"-win_arm64\.whl$"),
    ),
    SupportedPlatform(
        key="windows_x86_64",
        os="windows",
        cpu="x86_64",
        wheel_pattern=re.compile(r"-win_amd64\.whl$"),
    ),
)

_UNIVERSAL_WHEEL = re.compile(r"-(?:py3|py2\.py3)-none-any\.whl$")
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9._+-]+\.whl$")
_SAFE_PACKAGE_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_PYTHON_FULL_VERSION_MARKER = re.compile(
    r"^python_full_version\s*(<|<=|==|!=|>=|>)\s*'"
    r"([0-9]+)\.([0-9]+)(?:\.([0-9]+))?'$"
)
_BUCK_PYTHON_VERSION = (3, 13, 0)
_SUPPORTED_HOST_MARKERS = {
    # The bridge renders one dependency graph shared by every supported Buck
    # host, so a host-only dependency is included when any supported host uses
    # it. Buck selects the correct wheel at the package target.
    "sys_platform == 'win32'": True,
}


class DependencyBridgeError(ValueError):
    """Raised when the lock cannot be represented safely in Buck2."""


@dataclass(frozen=True, slots=True)
class LockedWheel:
    """One immutable wheel artifact from uv.lock."""

    filename: str
    url: str
    sha256: str


@dataclass(frozen=True, slots=True)
class LockedPackage:
    """One locked registry package and its direct dependency names."""

    name: str
    version: str
    dependencies: tuple[str, ...]
    wheels: tuple[LockedWheel, ...]


@dataclass(frozen=True, slots=True)
class LockedDependencyGraph:
    """Optional dependency extras and the packages reachable from them."""

    extras: Mapping[str, tuple[str, ...]]
    packages: Mapping[str, LockedPackage]


def normalize_package_name(value: str) -> str:
    """Return the normalized distribution and Buck target name."""
    normalized = re.sub(r"[-_.]+", "-", value).lower()
    if not _SAFE_PACKAGE_NAME.fullmatch(normalized):
        raise DependencyBridgeError(
            f"package name {value!r} cannot form a safe Buck target"
        )
    return normalized


def load_dependency_graph(
    lock_path: Path,
    *,
    project_name: str = "agentrig",
) -> LockedDependencyGraph:
    """Load the union of all project optional-dependency closures."""
    raw_document = cast(object, tomllib.loads(lock_path.read_text()))
    document = _require_mapping(raw_document, "lock document")
    raw_packages = _require_sequence(document.get("package"), "package list")

    package_records: dict[str, Mapping[str, object]] = {}
    for index, raw_package in enumerate(raw_packages):
        package = _require_mapping(raw_package, f"package {index}")
        name = normalize_package_name(
            _require_string(package.get("name"), f"package {index} name")
        )
        if name in package_records:
            raise DependencyBridgeError(
                f"multiple locked versions of {name!r} are not supported"
            )
        package_records[name] = package

    normalized_project = normalize_package_name(project_name)
    try:
        project = package_records[normalized_project]
    except KeyError as error:
        raise DependencyBridgeError(
            f"project package {normalized_project!r} is missing"
        ) from error

    raw_extras = _require_mapping(
        project.get("optional-dependencies"),
        "project optional dependencies",
    )
    extras: dict[str, tuple[str, ...]] = {}
    for extra_name, raw_requirements in sorted(raw_extras.items()):
        requirements = _require_sequence(
            raw_requirements,
            f"optional dependency group {extra_name!r}",
        )
        selected_roots: list[str] = []
        for requirement in requirements:
            selected_root = _dependency_name(
                requirement,
                "optional dependency",
            )
            if selected_root is not None:
                selected_roots.append(selected_root)
        roots = tuple(selected_roots)
        if not roots:
            raise DependencyBridgeError(
                f"optional dependency group {extra_name!r} is empty"
            )
        extras[extra_name] = roots

    reachable = _resolve_reachable_packages(package_records, extras)
    packages = {
        name: _parse_registry_package(package_records[name])
        for name in sorted(reachable)
    }
    return LockedDependencyGraph(extras=extras, packages=packages)


def select_locked_wheels(
    package: LockedPackage,
) -> Mapping[str, LockedWheel]:
    """Select either one universal wheel or one wheel per supported host."""
    universal = tuple(
        wheel
        for wheel in package.wheels
        if _UNIVERSAL_WHEEL.search(wheel.filename)
    )
    if universal:
        if len(universal) != 1:
            raise DependencyBridgeError(
                f"{package.name} has multiple universal wheels"
            )
        return {"universal": universal[0]}

    selected: dict[str, LockedWheel] = {}
    for platform in SUPPORTED_PLATFORMS:
        matches = tuple(
            wheel
            for wheel in package.wheels
            if platform.wheel_pattern.search(wheel.filename)
            and (
                "cp313-cp313" in wheel.filename
                or "-py3-none-" in wheel.filename
            )
        )
        if len(matches) != 1:
            raise DependencyBridgeError(
                f"{package.name} requires exactly one CPython 3.13 wheel for "
                f"{platform.key}; found {len(matches)}"
            )
        selected[platform.key] = matches[0]
    return selected


def render_locked_dependencies(graph: LockedDependencyGraph) -> str:
    """Render deterministic Buck targets for one locked dependency graph."""
    lines = [
        '"""Generated by tools/generate_buck_python_deps.py; do not edit."""',
        "",
        'load("@prelude//:native.bzl", "native")',
        "",
        "",
        "def locked_python_dependencies():",
    ]
    for package in graph.packages.values():
        lines.extend(_render_package(package))
    for extra_name, roots in sorted(graph.extras.items()):
        lines.extend(_render_extra(extra_name, roots))
    return "\n".join(lines) + "\n"


def generate_locked_dependencies(lock_path: Path) -> str:
    """Load and render the project optional dependencies."""
    return render_locked_dependencies(load_dependency_graph(lock_path))


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the checked-in bridge or verify that it is current."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lockfile", type=Path, default=Path("uv.lock"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("third_party/python/locked_deps.bzl"),
    )
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)

    rendered = generate_locked_dependencies(arguments.lockfile)
    if arguments.check:
        if not arguments.output.is_file():
            print(f"missing generated dependency bridge: {arguments.output}")
            return 1
        if arguments.output.read_text() != rendered:
            print(
                "generated dependency bridge is stale; run "
                "uv run python tools/generate_buck_python_deps.py"
            )
            return 1
        print("Buck Python dependency bridge matches uv.lock")
        return 0

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(rendered)
    print(f"wrote {arguments.output}")
    return 0


def _resolve_reachable_packages(
    records: Mapping[str, Mapping[str, object]],
    extras: Mapping[str, tuple[str, ...]],
) -> frozenset[str]:
    pending = [root for roots in extras.values() for root in roots]
    reachable: set[str] = set()
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        try:
            package = records[name]
        except KeyError as error:
            raise DependencyBridgeError(
                f"locked dependency {name!r} is missing"
            ) from error
        reachable.add(name)
        pending.extend(_dependency_names(package))
    return frozenset(reachable)


def _parse_registry_package(
    package: Mapping[str, object],
) -> LockedPackage:
    name = normalize_package_name(
        _require_string(package.get("name"), "package name")
    )
    version = _require_string(package.get("version"), f"{name} version")
    source = _require_mapping(package.get("source"), f"{name} source")
    registry = source.get("registry")
    if not isinstance(registry, str) or registry != "https://pypi.org/simple":
        raise DependencyBridgeError(
            f"{name} must resolve from the locked PyPI registry"
        )
    raw_wheels = _require_sequence(package.get("wheels"), f"{name} wheels")
    wheels = tuple(
        _parse_wheel(raw_wheel, package_name=name)
        for raw_wheel in raw_wheels
    )
    if not wheels:
        raise DependencyBridgeError(f"{name} has no locked wheels")
    return LockedPackage(
        name=name,
        version=version,
        dependencies=_dependency_names(package),
        wheels=wheels,
    )


def _dependency_names(package: Mapping[str, object]) -> tuple[str, ...]:
    raw_dependencies = package.get("dependencies", [])
    dependencies = _require_sequence(raw_dependencies, "package dependencies")
    selected_names: list[str] = []
    for dependency in dependencies:
        selected_name = _dependency_name(dependency, "dependency")
        if selected_name is not None:
            selected_names.append(selected_name)
    names = tuple(selected_names)
    return tuple(sorted(set(names)))


def _dependency_name(value: object, label: str) -> str | None:
    dependency = _require_mapping(value, label)
    unsupported_fields = set(dependency) - {"marker", "name"}
    if unsupported_fields:
        raise DependencyBridgeError(
            f"{label} uses unsupported lock fields: "
            f"{', '.join(sorted(unsupported_fields))}"
        )
    marker = dependency.get("marker")
    if marker is not None:
        if not isinstance(marker, str):
            raise DependencyBridgeError(f"{label} marker must be text")
        if not _marker_applies_to_buck_python(marker):
            return None
    return normalize_package_name(
        _require_string(dependency.get("name"), f"{label} name")
    )


def _marker_applies_to_buck_python(marker: str) -> bool:
    """Evaluate the strict marker subset used by supported Buck hosts."""
    host_value = _SUPPORTED_HOST_MARKERS.get(marker)
    if host_value is not None:
        return host_value
    match = _PYTHON_FULL_VERSION_MARKER.fullmatch(marker)
    if match is None:
        raise DependencyBridgeError(
            "dependency marker is unsupported by the Buck Python bridge"
        )
    operator, major, minor, patch = match.groups()
    requested = (int(major), int(minor), int(patch or "0"))
    if operator == "<":
        return _BUCK_PYTHON_VERSION < requested
    if operator == "<=":
        return _BUCK_PYTHON_VERSION <= requested
    if operator == "==":
        return _BUCK_PYTHON_VERSION == requested
    if operator == "!=":
        return _BUCK_PYTHON_VERSION != requested
    if operator == ">=":
        return _BUCK_PYTHON_VERSION >= requested
    if operator == ">":
        return _BUCK_PYTHON_VERSION > requested
    raise AssertionError("validated marker operator is unsupported")


def _parse_wheel(
    raw_wheel: object,
    *,
    package_name: str,
) -> LockedWheel:
    wheel = _require_mapping(raw_wheel, f"{package_name} wheel")
    url = _require_string(wheel.get("url"), f"{package_name} wheel URL")
    if not url.startswith("https://files.pythonhosted.org/"):
        raise DependencyBridgeError(
            f"{package_name} wheel must use files.pythonhosted.org"
        )
    filename = Path(urlparse(url).path).name
    if not _SAFE_FILENAME.fullmatch(filename):
        raise DependencyBridgeError(
            f"{package_name} has unsafe wheel filename {filename!r}"
        )
    locked_hash = _require_string(
        wheel.get("hash"), f"{package_name} wheel hash"
    )
    prefix = "sha256:"
    if not locked_hash.startswith(prefix):
        raise DependencyBridgeError(
            f"{package_name} wheel must have a SHA-256 hash"
        )
    sha256 = locked_hash.removeprefix(prefix)
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise DependencyBridgeError(
            f"{package_name} wheel has an invalid SHA-256 hash"
        )
    return LockedWheel(filename=filename, url=url, sha256=sha256)


def _render_package(package: LockedPackage) -> list[str]:
    selected = select_locked_wheels(package)
    lines = ["", f"    # {package.name}=={package.version}"]
    labels = [
        "locked-python",
        f"package={package.name}",
        f"version={package.version}",
    ]
    if "universal" in selected:
        wheel = selected["universal"]
        lines.extend(
            _render_wheel_targets(
                package,
                wheel,
                suffix="universal",
                labels=labels,
            )
        )
        wheel_dependencies = [f":{package.name}__universal_wheel"]
    else:
        for platform in SUPPORTED_PLATFORMS:
            lines.extend(
                _render_wheel_targets(
                    package,
                    selected[platform.key],
                    suffix=platform.key,
                    labels=labels,
                    platform=platform,
                )
            )
        wheel_dependencies = []

    direct_dependencies = [f":{name}" for name in package.dependencies]
    lines.extend(
        _render_package_wrapper(
            package,
            direct_dependencies=direct_dependencies,
            universal_wheels=wheel_dependencies,
        )
    )
    return lines


def _render_wheel_targets(
    package: LockedPackage,
    wheel: LockedWheel,
    *,
    suffix: str,
    labels: Sequence[str],
    platform: SupportedPlatform | None = None,
) -> list[str]:
    download_name = f"{package.name}__{suffix}_download"
    wheel_name = f"{package.name}__{suffix}_wheel"
    compatibility: list[str] = []
    if platform is not None:
        compatibility = [platform.os_constraint, platform.cpu_constraint]

    lines = [
        "    native.remote_file(",
        f"        name = {_quote(download_name)},",
        f"        url = {_quote(wheel.url)},",
        f"        sha256 = {_quote(wheel.sha256)},",
        f"        out = {_quote(wheel.filename)},",
    ]
    if compatibility:
        lines.extend(_render_string_list("target_compatible_with", compatibility))
    lines.extend(_render_string_list("labels", labels))
    lines.extend(
        [
            "    )",
            "    native.prebuilt_python_library(",
            f"        name = {_quote(wheel_name)},",
            f"        binary_src = {_quote(':' + download_name)},",
        ]
    )
    if compatibility:
        lines.extend(_render_string_list("target_compatible_with", compatibility))
    lines.extend(_render_string_list("labels", labels))
    lines.append("    )")
    return lines


def _render_package_wrapper(
    package: LockedPackage,
    *,
    direct_dependencies: Sequence[str],
    universal_wheels: Sequence[str],
) -> list[str]:
    lines = [
        "    native.python_library(",
        f"        name = {_quote(package.name)},",
    ]
    base_dependencies = tuple(universal_wheels) + tuple(direct_dependencies)
    if universal_wheels:
        lines.extend(_render_string_list("deps", base_dependencies))
    else:
        if direct_dependencies:
            lines.extend(
                _render_string_list(
                    "deps",
                    direct_dependencies,
                    suffix=" + select({",
                )
            )
        else:
            lines.append("        deps = select({")
        lines.extend(_render_platform_select(package.name))
        lines.append("        }),")
    lines.extend(
        _render_string_list(
            "labels",
            (
                "locked-python",
                f"package={package.name}",
                f"version={package.version}",
            ),
        )
    )
    lines.append("    )")
    return lines


def _render_platform_select(package_name: str) -> list[str]:
    lines: list[str] = []
    platforms_by_os: dict[str, list[SupportedPlatform]] = {}
    for platform in SUPPORTED_PLATFORMS:
        platforms_by_os.setdefault(platform.os, []).append(platform)
    for os_name, platforms in sorted(platforms_by_os.items()):
        lines.append(f"            {_quote('prelude//os:' + os_name)}: select({{")
        for platform in sorted(platforms, key=lambda item: item.cpu):
            wheel_target = f":{package_name}__{platform.key}_wheel"
            lines.append(
                f"                {_quote(platform.cpu_setting)}: "
                f"[{_quote(wheel_target)}],"
            )
        lines.append("            }),")
    return lines


def _render_extra(extra_name: str, roots: Sequence[str]) -> list[str]:
    lines = [
        "",
        f"    # Project optional dependency group: {extra_name}",
        "    native.python_library(",
        f"        name = {_quote('extra-' + extra_name)},",
    ]
    lines.extend(_render_string_list("deps", [f":{root}" for root in roots]))
    lines.extend(
        [
            '        labels = ["locked-python", "optional-extra"],',
            '        visibility = ["PUBLIC"],',
            "    )",
        ]
    )
    return lines


def _render_string_list(
    name: str,
    values: Sequence[str],
    *,
    suffix: str = "",
) -> list[str]:
    lines = [f"        {name} = ["]
    lines.extend(f"            {_quote(value)}," for value in values)
    lines.append(f"        ]{suffix}" if suffix else "        ],")
    return lines


def _quote(value: str) -> str:
    return json.dumps(value)


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise DependencyBridgeError(f"{label} must be a table")
    return cast(dict[str, object], value)


def _require_sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise DependencyBridgeError(f"{label} must be an array")
    return cast(list[object], value)


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DependencyBridgeError(f"{label} must be a nonempty string")
    return value


if __name__ == "__main__":
    sys.exit(main())
