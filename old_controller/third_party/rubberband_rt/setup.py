from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.1.0"

ext_modules = [
    Pybind11Extension(
        "rubberband_rt",
        ["main.cpp"],
        define_macros=[("VERSION_INFO", __version__)],
        include_dirs=["rubberband/rubberband"],
        library_dirs=["rubberband/build"],
        libraries=["rubberband"],
        extra_objects=["rubberband/build/librubberband.a"]
    ),
]

setup(
    name="rubberband_rt",
    version=__version__,
    author="Chengyuan Ma",
    author_email="macy404@mit.edu",
    description="A test project using pybind11",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
