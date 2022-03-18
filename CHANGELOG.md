# Version - Development

## Features

- Adds trade tutorial to the documentation ([#16]https://github.com/SGIModel/MUSE_OS/pull/16)
- Updated branch names on pipelines ([#9](https://github.com/SGIModel/MUSE_OS/issues/9))
- Edited default package name from StarMUSE to MUSE ([#4](https://github.com/SGIModel/MUSE_OS/issues/4))
- Added new cases studies with multiple agents and spend limit ([#1](https://github.com/SGIModel/MUSE_OS/pull/1))
- Updates the model and the documentation to use the most recent version of MUSE
  ([#964](https://github.com/SGIModel/StarMuse/pull/964))
- Updates the model and the documentation to use the most recent version of MUSE
  ([#963](https://github.com/SGIModel/StarMuse/pull/963))
- Updates the documentation to use the most recent version of MUSE
  ([#922](https://github.com/SGIModel/StarMuse/pull/922))
- Updates the documentation to provide information on adhoc and scipy solvers as well as correctly defines ObjSort direction for minimisation/maximisation ([#949](https://github.com/SGIModel/StarMuse/pull/949))
- Introduces a check on the type of the data defining the objectives (string for Objective, float/int for ObjData, and Boolean for Objsort) ([#945](https://github.com/SGIModel/StarMuse/issues/945]))
- Updates the documentation to use the most recent version of MUSE ([#922](https://github.com/SGIModel/StarMuse/pull/922))
- - Improves the CI system, with a more thorough pre-commit hooks and QA
  ([#917](https://github.com/SGIModel/StarMuse/pull/917))
- Introduces the CHANGELOG file and PR template
  ([#916](https://github.com/SGIModel/StarMuse/pull/916))

## Optimizations

- None

## Bug fixes

- Fixes the dead links in the documentation now that the repository is open-sourced ([#3](https://github.com/SGIModel/MUSE_OS/issues/3))
- Ensures that the adhoc and scipy solvers require the same input in the agents file to minimise and maximise. Specifically, both solvers now require TRUE for minimisation and FALSE for maximisation ([#845](https://github.com/SGIModel/StarMuse/issues/845))
- Update the documentation to include a tutorial for implementing trade.
- Update the documentation on adding spend limit constraint description ([#941](https://github.com/SGIModel/StarMuse/issues/941))
- Fix typos in CHANGELOG file ([#939](https://github.com/SGIModel/StarMuse/pull/939))
- Specify error message for no commodity outputs ([#937](https://github.com/SGIModel/StarMuse/issues/937))
- Update the documentation on index redundancies ([#936](https://github.com/SGIModel/StarMuse/issues/936))
- Update the documentation on timeslice file removed ([#935](https://github.com/SGIModel/StarMuse/issues/935))
- Update the documentation on commodity types ([#934](https://github.com/SGIModel/StarMuse/issues/934))
- Update the documentation on regions file removed ([#933](https://github.com/SGIModel/StarMuse/issues/933))
- Update the documentation on commodity definition ([#932](https://github.com/SGIModel/StarMuse/issues/932))
- Update the documentation on agent share ([#931](https://github.com/SGIModel/StarMuse/issues/931))
- Update the documentation on equations displayed ([#930](https://github.com/SGIModel/StarMuse/issues/930))
- Update the documentation to correctly add an agent ([#927](https://github.com/SGIModel/StarMuse/issues/927))
- Comfort objective modified to keep asset dimensions ([#926](https://github.com/SGIModel/StarMuse/pull/926))
- Update the documentation on growth rate ([#923](https://github.com/SGIModel/StarMuse/issues/923))

## Breaking changes

- None
