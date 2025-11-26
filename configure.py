import platform
import re
import sys
from pathlib import Path


WINDOWS_ARGS = [
  "--toolchain=msvc",
  "--target-os=win64",
  "--incdir=INCLUDES",
  "--shlibdir=SHARED_LIBS",
  "--enable-mediafoundation",
]

WINDOWS_X86_64_ARGS = [
  "--arch=x86_64",
]

MACOS_ARGS = [
  "--target-os=darwin",
  "--incdir=INCLUDES",
  "--shlibdir=SHARED_LIBS",
  "--enable-videotoolbox",
]

MACOS_X86_64_ARGS = [
  "--arch=x86_64",
]

MACOS_ARM_ARGS = [
  "--arch=arm64",
  "--enable-cross-compile",
  "--extra-cflags=-arch",
  "--extra-cflags=arm64",
  "--extra-ldflags=-arch",
  "--extra-ldflags=arm64",
]

STANDARD_ARGS = [
  "--disable-debug",
  "--enable-stripping",
  "--enable-shared",
  "--enable-pic",
  "--disable-autodetect",
  "--disable-programs",
  "--enable-avcodec",
  "--enable-avdevice",
  "--enable-avfilter",
  "--enable-avformat",
  "--enable-avutil",
  "--enable-postproc",
  "--enable-swresample",
  "--enable-swscale",
]

LIBX264_ARGS = STANDARD_ARGS + [
  "--enable-libx264",
]

CONFIGURATIONS = {
  "standard": STANDARD_ARGS,
  "libx264": LIBX264_ARGS,
}

FFMPEG_CONFIGURE_PATH = Path("FFmpeg/configure")


def patch_libx264_license():
  """Remove libx264 from the GPL list and keep it in the generic library list."""
  configure_path = FFMPEG_CONFIGURE_PATH
  if not configure_path.exists():
    raise FileNotFoundError(f"Missing configure script at {configure_path}")

  content = configure_path.read_text()

  gpl_match = re.search(r'(EXTERNAL_LIBRARY_GPL_LIST="\n)(?P<entries>(?:    .*\n)+?)(")', content)
  if not gpl_match:
    raise RuntimeError("Could not locate EXTERNAL_LIBRARY_GPL_LIST in configure")

  gpl_entries = gpl_match.group("entries").splitlines()
  gpl_entries_no_x264 = [line for line in gpl_entries if line.strip() != "libx264"]

  patched = content
  if gpl_entries_no_x264 != gpl_entries:
    new_gpl_entries = "\n".join(gpl_entries_no_x264) + "\n"
    patched = (
      patched[:gpl_match.start("entries")] +
      new_gpl_entries +
      patched[gpl_match.end("entries"):]
    )

  # Ensure libx264 is still present in the full external library list.
  external_match = re.search(r'(EXTERNAL_LIBRARY_LIST="\n)(?P<entries>(?:    .*\n)+?)(")', patched)
  if not external_match:
    raise RuntimeError("Could not locate EXTERNAL_LIBRARY_LIST in configure")

  external_entries = external_match.group("entries").splitlines()
  if not any(line.strip() == "libx264" for line in external_entries):
    insert_at = next((i for i, line in enumerate(external_entries) if line.strip() == "libxevd"), len(external_entries))
    external_entries.insert(insert_at, "    libx264")
    new_external_entries = "\n".join(external_entries) + "\n"
    patched = (
      patched[:external_match.start("entries")] +
      new_external_entries +
      patched[external_match.end("entries"):]
    )

  if patched != content:
    configure_path.write_text(patched)
    print("Patched FFmpeg/configure to treat libx264 as non-GPL.")


def build():
  if len(sys.argv) != 3:
    raise ValueError("Invalid number of arguments, expected 2: configuration arch")

  arch = sys.argv[2]

  args = []

  # If windows, prepend windows args
  if platform.system() == "Windows":
    print("Windows detected")
    args += WINDOWS_ARGS
    if arch == "x86_64":
      args += WINDOWS_X86_64_ARGS
    else:
      raise ValueError(f"Invalid arch: {arch}")

  elif platform.system() == "Darwin":
    print("MacOS detected")
    args += MACOS_ARGS
    if arch == "x86_64":
      args += MACOS_X86_64_ARGS
    elif arch == "arm64":
      args += MACOS_ARM_ARGS
    else:
      raise ValueError(f"Invalid arch: {arch}")

  configuration = sys.argv[1]
  if configuration not in CONFIGURATIONS:
    raise ValueError(f"Invalid configuration: {configuration}")

  if configuration == "libx264":
    patch_libx264_license()

  args += CONFIGURATIONS[configuration]

  with open("run_configure.sh", "w") as f:
    f.write("#!/bin/bash\n")
    f.write("set -x\n")
    f.write(f"echo Running configuration {configuration}\n")
    f.write(f"./configure {' '.join(args)}\n")

build()
