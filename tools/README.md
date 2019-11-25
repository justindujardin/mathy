Build Tools
---

These utilities are consistently named and present in a "tools" folder for each
app in the monorepo. This means you can CD into any app and run `sh tools/setup.sh`
to install its prerequisites and then `sh tools/build.sh` to build the app.

These files should **NEVER** be run from within the tools folder. They should always be called
from the root folder of the app (or repo).
