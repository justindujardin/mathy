# [0.6.0](https://github.com/justindujardin/mathy/compare/v0.5.3...v0.6.0) (2020-01-05)


### Bug Fixes

* **colab:** force upgrade mathy in snippets to avoid outdated dependency issues ([a902a8c](https://github.com/justindujardin/mathy/commit/a902a8c0b36feabcb83500bad6e1f36b9fa27077))
* **envs:** use uniform random range for complex/binomial params ([8534239](https://github.com/justindujardin/mathy/commit/8534239c2f120569bcc9d77ee095f1b5a7dab849))
* **zero:** default to non-recurrent architecture ([38690c2](https://github.com/justindujardin/mathy/commit/38690c241289de4d31918c383bb86f2a48d318bc))


### Features

* **embedding:** add optional non-recurrent DenseNet+Attention architecture `--use-lstm=False` ([8b97e9d](https://github.com/justindujardin/mathy/commit/8b97e9d7591f6fcac4532f7d7111c67d342b55a8))
* **zero:** add `--profile` support for `num_workers=1` ([622cdfd](https://github.com/justindujardin/mathy/commit/622cdfd25a272614bde7607079c5b3a7e0e67370))
* **zero:** support `--show` argument for worker 0 ([82b3f66](https://github.com/justindujardin/mathy/commit/82b3f6604459eaf3e1acc4b92704fa1659d76649))


### Reverts

* **zero:** use old-style batch training ([4d5fd37](https://github.com/justindujardin/mathy/commit/4d5fd37143483e578a70e3fed9a77a9b71048be9))

## [0.5.3](https://github.com/justindujardin/mathy/compare/v0.5.2...v0.5.3) (2020-01-03)


### Bug Fixes

* **zero:** error on second set of training eps if model didn't train ([57513a6](https://github.com/justindujardin/mathy/commit/57513a6d6676f22d289226b9e9583f5241755a3d))

## [0.5.2](https://github.com/justindujardin/mathy/compare/v0.5.1...v0.5.2) (2020-01-03)


### Performance Improvements

* **mathy_alpha_sm:** slightly more trained multi-task model ([0f16d61](https://github.com/justindujardin/mathy/commit/0f16d61c65e8b39842d17a592ffd5736ea6541b7))

## [0.5.1](https://github.com/justindujardin/mathy/compare/v0.5.0...v0.5.1) (2020-01-03)


### Bug Fixes

* **models:** update default mathy version string to use range ([53292ac](https://github.com/justindujardin/mathy/commit/53292acc849c9fec7f863ba015c319dd103b902f))

## [0.5.0](https://github.com/justindujardin/mathy/compare/v0.4.0...v0.5.0) (2020-01-02)


### Bug Fixes

* **docs:** don't generate api docs for .DS_Store ([cb24977](https://github.com/justindujardin/mathy/commit/cb2497761d1db97a626bf1ea6b9ddfbfbcdaefb0))
* **mathy_alpha_sm:** set more liberal mathy range ([3a4e59c](https://github.com/justindujardin/mathy/commit/3a4e59c0006b80d80608db2684ab667b2629da35))


### Features

* **expressions:** better type hints ([ff8bd65](https://github.com/justindujardin/mathy/commit/ff8bd65b690e762443c014e235fd6f742e8122f0))

## [0.4.0](https://github.com/justindujardin/mathy/compare/v0.3.5...v0.4.0) (2020-01-02)


### Bug Fixes

* **mathy_alpha_sm:** revert defaults to last known good model ([fc27522](https://github.com/justindujardin/mathy/commit/fc275229425f984642dbe12d4a7f637c8267a526))
* **poly_simplify:** set default ops to + ([0ad2f5c](https://github.com/justindujardin/mathy/commit/0ad2f5c12b40d51620926047795f90cbf27a5935))


### Features

* **mathy_alpha_sm:** updated multi-task model ([a8f47a4](https://github.com/justindujardin/mathy/commit/a8f47a4654d677f72d4b83f199cd5045b9254933))

## [0.3.5](https://github.com/justindujardin/mathy/compare/v0.3.4...v0.3.5) (2020-01-02)


### Bug Fixes

* **mathy_alpha_sm:** add long_description to setup ([cd68ff0](https://github.com/justindujardin/mathy/commit/cd68ff0abda877e479d0674bc994197e3557321c))

## [0.3.4](https://github.com/justindujardin/mathy/compare/v0.3.3...v0.3.4) (2020-01-02)


### Bug Fixes

* **deploy:** sync mathy/mathy_alpha_sm versions ([cbee0fb](https://github.com/justindujardin/mathy/commit/cbee0fb0fd0601409c8e630fa347602eb9cafd7d))
* **mathy_alpha_sm:** include model data and readme ([13db587](https://github.com/justindujardin/mathy/commit/13db587a9ffa8b8959eb4b36081eefa5b84665cf))

## [0.3.3](https://github.com/justindujardin/mathy/compare/v0.3.2...v0.3.3) (2020-01-02)


### Bug Fixes

* **readme:** use absolute image path for logo ([0fb137f](https://github.com/justindujardin/mathy/commit/0fb137fd55dcd66bef5991388fdee5aa70cda087))

## [0.3.2](https://github.com/justindujardin/mathy/compare/v0.3.1...v0.3.2) (2020-01-02)


### Bug Fixes

* **mathy_alpha_sm:** typo in deploy script ([5911ed2](https://github.com/justindujardin/mathy/commit/5911ed2d9019abf782073b1ffe828b40772696d9))

## [0.3.1](https://github.com/justindujardin/mathy/compare/v0.3.0...v0.3.1) (2020-01-02)


### Bug Fixes

* **ci:** replace mathy/mathy_alpha_sm readmes during build ([c9d53ee](https://github.com/justindujardin/mathy/commit/c9d53ee0b9b9561f44da1b30f77e7406c7792b71))
* **ci:** setup/build mathy_alpha_sm model before deploy ([a990a24](https://github.com/justindujardin/mathy/commit/a990a245ccbd1330e4d0fea7b80190b8e3f2a816))

## [0.3.0](https://github.com/justindujardin/mathy/compare/v0.2.3...v0.3.0) (2020-01-02)


### Bug Fixes

* **mathy:** add long desc content type to setup.py ([ca8fa39](https://github.com/justindujardin/mathy/commit/ca8fa39703dd5fd046cb5059bb106da8e9595e0b))


### Features

* **mathy_alpha_sm:** add deploy for small model ([11e63d5](https://github.com/justindujardin/mathy/commit/11e63d521e7f2fe7d0bf8ceab5e31341a3c79f1c))

## [0.2.3](https://github.com/justindujardin/mathy/compare/v0.2.2...v0.2.3) (2020-01-02)


### Bug Fixes

* **ci:** use tag filter workaround ([bc3cac6](https://github.com/justindujardin/mathy/commit/bc3cac64da15dc21a8797b5d2cdc0920c6f4fe5e))

## [0.2.2](https://github.com/justindujardin/mathy/compare/v0.2.1...v0.2.2) (2020-01-02)


### Bug Fixes

* **ci:** add mathy/about.py to git changes from release ([ff411dd](https://github.com/justindujardin/mathy/commit/ff411dd01aa65521c1eb65c8de4a9ac3306b04a2))
* **ci:** don't add skip-ci to release commits ([c666d93](https://github.com/justindujardin/mathy/commit/c666d93b847764b54e3a480217299e8625677591))

## [0.2.1](https://github.com/justindujardin/mathy/compare/v0.2.0...v0.2.1) (2020-01-02)


### Bug Fixes

* **actions:** ci tag filter ([833bbbc](https://github.com/justindujardin/mathy/commit/833bbbc4ffa7cb3b9c6d0963df4d8945c608aaad))

## [0.2.0](https://github.com/justindujardin/mathy/compare/v0.1.0...v0.2.0) (2020-01-02)

The initial public packaging and release of Mathy!

### Features

- **Mathy:** Initial `mathy` python package
- **Models:** Initial `mathy_alpha_sm` python package
- **Website:** Initial website with runnable tests/examples
