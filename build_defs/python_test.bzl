"""AgentRig's Python test policy."""

load("@prelude//:native.bzl", "native")

_TEST_SCOPES = (
    "unit",
    "contract",
    "integration",
    "eval",
)

_MANAGED_LABELS = _TEST_SCOPES + (
    "live",
    "offline",
)


def agentrig_python_test(
        name,
        scope,
        srcs,
        deps = [],
        base_module = None,
        env = {},
        labels = [],
        live = False,
        resources = [],
        visibility = []):
    """Declares a Python test with one scope and an explicit execution mode."""
    if scope not in _TEST_SCOPES:
        fail("scope must be one of: {}".format(", ".join(_TEST_SCOPES)))

    if scope == "unit" and live:
        fail("unit tests cannot be live")

    conflicting_labels = [label for label in labels if label in _MANAGED_LABELS]
    if conflicting_labels:
        fail(
            "labels {} are managed by agentrig_python_test".format(
                ", ".join(conflicting_labels),
            ),
        )

    main_module = None
    support_deps = []
    if scope == "unit":
        main_module = "tests.support.unit_test_main"
        support_deps = ["//tests/support:test_support"]
    elif live:
        main_module = "tests.support.live_test_main"
        support_deps = ["//tests/support:test_support"]

    execution_label = "live" if live else "offline"

    native.python_test(
        name = name,
        srcs = srcs,
        base_module = base_module,
        deps = deps + support_deps,
        env = env,
        labels = [scope, execution_label] + labels,
        main_module = main_module,
        resources = resources,
        visibility = visibility,
    )
