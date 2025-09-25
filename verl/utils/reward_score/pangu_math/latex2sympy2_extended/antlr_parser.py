from importlib.metadata import version, PackageNotFoundError

try:
    antlr_version = version("antlr4-python3-runtime")
except PackageNotFoundError:
    antlr_version = ""

if antlr_version.startswith("4.13.2"):
    from .gen.antlr4_13_2.PSParser import PSParser
    from .gen.antlr4_13_2.PSLexer import PSLexer
elif antlr_version.startswith("4.11"):
    from .gen.antlr4_11_0.PSParser import PSParser
    from .gen.antlr4_11_0.PSLexer import PSLexer
elif antlr_version.startswith("4.9.3"):
    from .gen.antlr4_9_3.PSParser import PSParser
    from .gen.antlr4_9_3.PSLexer import PSLexer
else:
    raise ImportError(
        f"Unsupported ANTLR version {antlr_version}, "
        "only 4.9.3, 4.11.0, and 4.13.2 runtime versions are supported."
    )