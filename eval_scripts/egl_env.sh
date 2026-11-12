# Shared EGL setup for habitat-sim in containers without the NVIDIA graphics driver. Source this from an eval script.
#
# The container only mounts the compute half of the NVIDIA driver
# (NVIDIA_DRIVER_CAPABILITIES=compute,utility), so there is no libEGL_nvidia.so
# and no /usr/share/glvnd/egl_vendor.d/10_nvidia.json. habitat-sim then finds
# only the Mesa software device and dies with
#   unable to find CUDA device 0 among 1 EGL devices in total
# Pointing the EGL loader at a version-matched extracted driver fixes it.
#
# The extracted version must match the running kernel module exactly, so it is
# resolved from nvidia-smi rather than hardcoded.

export EGL_PLATFORM=surfaceless
unset DISPLAY

NVIDIA_EGL_ROOT=${NVIDIA_EGL_ROOT:-${HOME}/nvidia-egl}
_egl_driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')

if [ -z "${_egl_driver_version}" ]; then
    echo "WARNING: nvidia-smi unavailable; skipping EGL setup. habitat-sim will likely fail to find a CUDA EGL device." >&2
else
    NVIDIA_EGL_DIR="${NVIDIA_EGL_ROOT}/NVIDIA-Linux-x86_64-${_egl_driver_version}"
    if [ -e "${NVIDIA_EGL_DIR}/libEGL_nvidia.so.0" ] && [ -e "${NVIDIA_EGL_DIR}/10_nvidia.json" ]; then
        export LD_LIBRARY_PATH="${NVIDIA_EGL_DIR}:${LD_LIBRARY_PATH:-}"
        export __EGL_VENDOR_LIBRARY_DIRS="${NVIDIA_EGL_DIR}"
    elif [ ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
        echo "WARNING: no extracted graphics libs for driver ${_egl_driver_version} at ${NVIDIA_EGL_DIR}." >&2
        echo "         habitat-sim will fall back to the Mesa software device and fail." >&2
        echo "         Extract the matching driver with:" >&2
        echo "           cd ${NVIDIA_EGL_ROOT} && ./NVIDIA-Linux-x86_64-${_egl_driver_version}.run -x" >&2
    fi
fi
unset _egl_driver_version
