from distutils.core import setup, Extension

def main():
    setup(name="netPy",
          version="1.0.8",
          description="Neuronal Net for Python",
          author="Konstantin Rossmann",
          author_email="konstantin.rossmann@web.de",
          ext_modules=[Extension("netPy", ["src/netPy_PyModule.c"])])

if __name__ == "__main__":
    main()