import importlib
import importlib.abc
import importlib.util
import sys

# Backward compatibility: before this package was installable, projects that vendor
# it as a git submodule imported it as ``rgbd_dataset.rgbd_dataset.<module>``.
# Resolve those names to the same module objects as ``rgbd_dataset.<module>``.
_LEGACY_PREFIX = __name__ + ".rgbd_dataset"


class _LegacyImportAlias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == _LEGACY_PREFIX or fullname.startswith(_LEGACY_PREFIX + "."):
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return importlib.import_module(__name__ + spec.name[len(_LEGACY_PREFIX) :])

    def exec_module(self, module):
        pass


if not any(isinstance(f, _LegacyImportAlias) for f in sys.meta_path):
    sys.meta_path.insert(0, _LegacyImportAlias())
