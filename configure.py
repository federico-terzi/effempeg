import platform
import sys


WINDOWS_ARGS = [
  "--toolchain=msvc",
  "--target-os=win64",
  "--incdir=INCLUDES",
  "--shlibdir=SHARED_LIBS",
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

CONFIGURATIONS = {
  "standard": STANDARD_ARGS,
}

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
  
  args += CONFIGURATIONS[configuration]

  with open("run_configure.sh", "w") as f:
    f.write("#!/bin/bash\n")
    f.write("set -x\n")
    f.write(f"echo Running configuration {configuration}\n")
    f.write(f"./configure {' '.join(args)}\n")

build()