# Shared interpreter selection. Source before launching Python.
resolve_python() {
    local requested="${PYTHON:-python}"
    PYTHON="$(command -v "${requested}")" || {
        echo "ERROR: Python '${requested}' not found. Activate the Habitat environment or set PYTHON." >&2
        return 1
    }
    [ -x "${PYTHON}" ] || { echo "ERROR: PYTHON=${PYTHON} is not executable" >&2; return 1; }
    export PYTHON
}
