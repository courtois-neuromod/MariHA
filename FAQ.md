# MariHA FAQ

## `--render_every` crashes with "Corruption of the global PassRegistry"

**Symptoms**: When running `mariha-run-single --render_every 2` (or any render
flag), training crashes with repeated errors:

```
Error: Required pass not found! Possible causes:
    - Pass misconfiguration (e.g.: missing macros)
    - Corruption of the global PassRegistry
```

Followed by a segmentation fault.

**Cause**: TensorFlow's `pywrap_dlopen_global_flags.py` sets `RTLD_GLOBAL` when
loading its shared libraries. This exports TF's bundled LLVM symbols into the
global symbol table. When the render window opens, Mesa's OpenGL driver
(radeonsi, llvmpipe, etc.) loads the system `libLLVM.so`, which conflicts with
TF's already-loaded LLVM symbols. The two LLVM versions fight over the global
`PassRegistry`, causing the crash.

This only affects systems where the GPU driver uses LLVM for shader compilation
(AMD radeonsi, or Mesa's llvmpipe software renderer). NVIDIA proprietary drivers
do not use LLVM and are unaffected.

**Fix**: Rename `pywrap_dlopen_global_flags.py` so TF's LLVM symbols stay
private and don't collide with Mesa's:

```bash
# Find TF's site-packages directory
TF_DIR=$(python -c "import tensorflow; print(tensorflow.__file__.rsplit('/',1)[0])")

mv "$TF_DIR/python/pywrap_dlopen_global_flags.py" \
   "$TF_DIR/python/pywrap_dlopen_global_flags.py.bak"

# Also remove the cached bytecode
rm -f "$TF_DIR/python/__pycache__/pywrap_dlopen_global_flags."*.pyc
```

This disables the global symbol export. TF continues to work normally (the flag
is only needed for `tf.load_op_library()` with statically-linked builds, which
MariHA does not use).

**Note**: This fix must be re-applied after upgrading or reinstalling
TensorFlow, since the file will be restored by `pip install`.

---

## `AttributeError: _ARRAY_API not found` on import

**Symptoms**: Importing MariHA (or TensorFlow) crashes immediately with:

```
AttributeError: _ARRAY_API not found
```

The full traceback goes through `matplotlib` -> `matplotlib.transforms` ->
`matplotlib._path`.

**Cause**: `matplotlib` was compiled against NumPy 1.x but NumPy 2.x is
installed. The C extensions are ABI-incompatible.

**Fix**: Upgrade matplotlib to a version compiled with NumPy 2.x support:

```bash
pip install --upgrade matplotlib
```

---

## TensorFlow / CUDA warnings on startup

**Symptoms**: Every run prints warnings like:

```
oneDNN custom operations are on ...
Could not find cuda drivers on your machine, GPU will not be used.
CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
```

**Cause**: TF is built with CUDA support but no NVIDIA GPU is available (e.g.,
on an AMD GPU system). These are harmless warnings.

**Fix**: Suppress them with environment variables:

```bash
export TF_CPP_MIN_LOG_LEVEL=2        # hide INFO/WARNING from TF C++ runtime
export TF_ENABLE_ONEDNN_OPTS=0       # silence oneDNN messages
```

Add these to your shell profile or to a wrapper script.
