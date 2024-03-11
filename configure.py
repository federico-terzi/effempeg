import platform
import sys


WINDOWS_ARGS = [
  "--toolchain=msvc",
  "--target-os=win64",
  "--arch=x86_64",
]

STANDARD_ARGS = [
  "--disable-debug",
  "--enable-stripping",
  "--enable-shared",
  "--shlibdir=SHARED_LIBS",
  "--enable-pic",
  "--disable-autodetect",
  "--disable-programs",
  "--enable-avcodec",
  "--disable-avdevice",
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
  if len(sys.argv) != 2:
    raise ValueError("Invalid number of arguments, expected 1: configuration")

  args = []

  # If windows, prepend windows args
  if platform.system() == "Windows":
    args += WINDOWS_ARGS
  
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