## [0.7.14](https://github.com/justindujardin/mathy/compare/v0.7.13...v0.7.14) (2020-04-19)


### Features

* **fragile:** update for website examples ([d9c8128](https://github.com/justindujardin/mathy/commit/d9c81286b29ce61bf92b60f77d62e5208cdabe1b))

## [0.7.13](https://github.com/justindujardin/mathy/compare/v0.7.12...v0.7.13) (2020-04-12)


### Features

* **mkdocs:** update mkdocs-material to 5.x SPA ([0be38b4](https://github.com/justindujardin/mathy/commit/0be38b4284cdbbf321c9d67f9a14a0a706feb259))

## [0.7.12](https://github.com/justindujardin/mathy/compare/v0.7.11...v0.7.12) (2020-04-09)


### Features

* **fragile:** enable multiprocessing solver by default ([3016c48](https://github.com/justindujardin/mathy/commit/3016c4810e2e96fa7505277afd2ac197bd1fe930))

## [0.7.11](https://github.com/justindujardin/mathy/compare/v0.7.10...v0.7.11) (2020-04-05)


### Features

* **problems:** add `use_pretty_numbers` function ([#41](https://github.com/justindujardin/mathy/issues/41)) ([8c54e2e](https://github.com/justindujardin/mathy/commit/8c54e2ed21c8e6906faf2e93cb82f54ff464bb9b))

## [0.7.10](https://github.com/justindujardin/mathy/compare/v0.7.9...v0.7.10) (2020-04-05)


### Bug Fixes

* **python3.6:** add backports for new typing features ([11017f9](https://github.com/justindujardin/mathy/commit/11017f9b26959e75eb39e4124237a36639c75008))

## [0.7.9](https://github.com/justindujardin/mathy/compare/v0.7.8...v0.7.9) (2020-04-05)


### Features

* **fragile:** update to 0.0.45 ([a8bc2b4](https://github.com/justindujardin/mathy/commit/a8bc2b4c2dc3b6b59f669b9ec61276a56a19023a))

## [0.7.8](https://github.com/justindujardin/mathy/compare/v0.7.7...v0.7.8) (2020-04-04)


### Features

* **fragile:** update to 0.0.44 ([#35](https://github.com/justindujardin/mathy/issues/35)) ([af99730](https://github.com/justindujardin/mathy/commit/af99730ff01e4702dee1b84bba597c72e4faa630))

## [0.7.7](https://github.com/justindujardin/mathy/compare/v0.7.6...v0.7.7) (2020-04-04)


### Features

* **print_history:** add pretty print flag ([#38](https://github.com/justindujardin/mathy/issues/38)) ([4cf6255](https://github.com/justindujardin/mathy/commit/4cf625564695b10bff897102659b4da9ece1beef))

## [0.7.6](https://github.com/justindujardin/mathy/compare/v0.7.5...v0.7.6) (2020-04-02)


### Bug Fixes

* **pypi:** optional swarm install with mathy[fragile] ([b6203fd](https://github.com/justindujardin/mathy/commit/b6203fde9fece38dc5d6f387fdc7b74f914eddda))

## [0.7.5](https://github.com/justindujardin/mathy/compare/v0.7.4...v0.7.5) (2020-04-02)


### Features

* **fragile:** add swarm agent as untrained solver ([8515273](https://github.com/justindujardin/mathy/commit/85152731b5813f706a6c5dc239ab9272b20aaafd))

## [0.7.4](https://github.com/justindujardin/mathy/compare/v0.7.3...v0.7.4) (2020-03-30)


### Features

* **ci:** publish github releases with changelog ([be502a1](https://github.com/justindujardin/mathy/commit/be502a16aaccb40351a6829a57cc26819d901d4d))

## [0.7.3](https://github.com/justindujardin/mathy/compare/v0.7.2...v0.7.3) (2020-03-17)


### Features

* **env:** add print_history helper ([41b3a0f](https://github.com/justindujardin/mathy/commit/41b3a0f65407a24e0730999c0224bc84527e4d99))

## [0.7.2](https://github.com/justindujardin/mathy/compare/v0.7.1...v0.7.2) (2020-03-16)


### Bug Fixes

* **gym:** return node ids and action mask for np observations ([d04366e](https://github.com/justindujardin/mathy/commit/d04366ea8dcad5d0b1da5105a97c249163f6411f))

## [0.7.1](https://github.com/justindujardin/mathy/compare/v0.7.0...v0.7.1) (2020-03-15)


### Features

* Add support for integrating with Fragile library ([be3ab58](https://github.com/justindujardin/mathy/commit/be3ab5825626489a7fb7bde424decf3022aa9672))

## [0.7.0](https://github.com/justindujardin/mathy/compare/v0.6.7...v0.7.0) (2020-03-08)


### Bug Fixes

* **a3c:** remove root noise from action selector ([16f86ff](https://github.com/justindujardin/mathy/commit/16f86ff67a0f24b6fa701d5462acd2e547aaa258))
* **a3c:** use episode outcome for log coloring ([910bcd6](https://github.com/justindujardin/mathy/commit/910bcd6ad06ba1d89f901b661810161f3cfd2052))
* **cli:** use greedy selector during inference ([15cc58a](https://github.com/justindujardin/mathy/commit/15cc58a42912b24f2fb1087e7249404b4d61cfb4))
* **env:** clamp episode win signal to 2.0 max ([3d2d78b](https://github.com/justindujardin/mathy/commit/3d2d78b8dadc4f271cfaafb6dbfc89015e88884d))
* **env:** remove reentrant state reward scaling ([0849e3c](https://github.com/justindujardin/mathy/commit/0849e3cc0e53acbf40a545bbdab4fcd44e8c7b4f))
* **get_terms_ex:** support negated implicit coefficients ([f763e20](https://github.com/justindujardin/mathy/commit/f763e20672222c552dba5f0b5557d82b60f2569f))
* **parser:** memory leak in cache ([6b7a847](https://github.com/justindujardin/mathy/commit/6b7a8473cbfefe6a8aa449fd923fbd8ac0fadb36))
* **rewards:** restore reentrant state scaling ([1361d74](https://github.com/justindujardin/mathy/commit/1361d748bd5287f26b4bf52b81c8b144f2b3f03d))
* **rules:** make commutative swap choose the closest sibling ([f32600e](https://github.com/justindujardin/mathy/commit/f32600ea7b068dbc56f4c4e1146b618289ec853b))


### chore

* drop time feature from embedding ([f5740ad](https://github.com/justindujardin/mathy/commit/f5740ada991d7515a7f1f6cea716332ba24e3ebd))


### Code Refactoring

* **model:** remove episode long RNN state tracking ([11095ab](https://github.com/justindujardin/mathy/commit/11095ab494b010681b77cb2d12576be3b87ca974))


### Features

* **a3c:** add bahdanau attention layer ([daba776](https://github.com/justindujardin/mathy/commit/daba77605776cc505836485ee23d1317c02d49a0))
* **a3c:** add exponential decay to learning rate ([684191d](https://github.com/justindujardin/mathy/commit/684191d5ec8fbc001b4fe34c523e3e83c1d90f2f))
* **a3c:** add self-attention over sequences ([b750bfc](https://github.com/justindujardin/mathy/commit/b750bfceaaaded584a4dce4880d08c9ca1c526bf))
* **a3c:** use stepped learning rate decay ([e9cd8f5](https://github.com/justindujardin/mathy/commit/e9cd8f527ba245b7e00f66e8eb561b83ff3e460c))
* **embedding:** use bilstm for node sequences ([ad23139](https://github.com/justindujardin/mathy/commit/ad231392fc63d0db58f3b3a99a20078797d75e93))
* **embedding:** use LSTMs for batch and time axes ([a8f0d54](https://github.com/justindujardin/mathy/commit/a8f0d540743dd36a46150d7589ef14a5918a8946))
* **mathy_alpha_sm:** more stable recurrent model ([02e63e2](https://github.com/justindujardin/mathy/commit/02e63e25649a0acc973b698c789c8b9a7b9e9e52))
* **training:** add yellow output to weak wins ([fd9998a](https://github.com/justindujardin/mathy/commit/fd9998a40153a86153d1582078bcb2bee45071a9))


### BREAKING CHANGES

* this removes a model feature that makes previous pretrained models incompatible
* **model:** this removes long-term RNN state tracking across episodes. Tracking the state was a significant amount of code and it wasn't clear that it made the model substantially better at any given task.

The overhead associated with keeping lots of hidden states in memory and calculating state histories was not insignificant on CPU training setups as well.

## [0.6.7](https://github.com/justindujardin/mathy/compare/v0.6.6...v0.6.7) (2020-02-10)


### Features

* **mathy_pydoc:** fix formatting of str defaults ([b4f6fde](https://github.com/justindujardin/mathy/commit/b4f6fde5185b0d2ad5309144a7ad10fe0cfc26aa))

## [0.6.6](https://github.com/justindujardin/mathy/compare/v0.6.5...v0.6.6) (2020-02-10)


### Features

* **build:** deploy mathy_pydoc package to pypi ([e2d5775](https://github.com/justindujardin/mathy/commit/e2d57757e546d7321987898eb32c6b12c6c3987a))
* **mathy_pydoc:** cleanup return type annotations ([186be77](https://github.com/justindujardin/mathy/commit/186be770bb11280cdf27880d95f10a4319075e1d))
* **mathy_pydoc:** preserve Optional types in docs ([830c949](https://github.com/justindujardin/mathy/commit/830c949747e7408e352cb51260f98c8c0e1a1c65))
* **mathy_pydoc:** unwrap ForwardRef types ([4e172c4](https://github.com/justindujardin/mathy/commit/4e172c4791e0ec790a0f1c70b9d35e6c45f67a61))

## [0.6.5](https://github.com/justindujardin/mathy/compare/v0.6.4...v0.6.5) (2020-01-27)


### Bug Fixes

* **build:** really fix typing extensions ([7f15bca](https://github.com/justindujardin/mathy/commit/7f15bca3e5ae8f3fa56b45f1f908c1eed3c1fd7a))

## [0.6.4](https://github.com/justindujardin/mathy/compare/v0.6.3...v0.6.4) (2020-01-26)


### Bug Fixes

* **package:** require typing_extensions ([55b0bc9](https://github.com/justindujardin/mathy/commit/55b0bc97e831727459cd01c06ce3be21d9f6a011))

## [0.6.3](https://github.com/justindujardin/mathy/compare/v0.6.2...v0.6.3) (2020-01-26)


### Bug Fixes

* **commutative_swap:** don't transform commute chains in ways that cause inner-nesting ([ed662e3](https://github.com/justindujardin/mathy/commit/ed662e3a05ac3f69206789b3909d320f33948932))
* **model:** remove second LSTM from recurent model ([0241070](https://github.com/justindujardin/mathy/commit/02410702add5b8800751860f897440e9a333d654))
* **model:** when trasnferring weights from another model, copy the config file too ([401da56](https://github.com/justindujardin/mathy/commit/401da569883a0b1fe0a171f6a682944675073731))
* **policy_value_model:** value head was not learning from hidden state ([ee77ae5](https://github.com/justindujardin/mathy/commit/ee77ae505192fda9124bd02189106b0ccdae407c))
* **sleep:** use smaller worker_wait defaults ([460f80c](https://github.com/justindujardin/mathy/commit/460f80cdb0c0767622f80c70a34c013233f58abd))
* **training:** use n-step windows during a3c training ([02b11ee](https://github.com/justindujardin/mathy/commit/02b11eebfb3043ad5d46212a1d7b578a28e66de7))


### Features

* **a3c:** replace set_weights/get_weights with thinc from_bytes/to_bytes ([b04fbce](https://github.com/justindujardin/mathy/commit/b04fbce88a822610195335d8dc65a0983f43d384))
* **a3c:** set update frequency so multiple updates happen per episode ([1a28a9d](https://github.com/justindujardin/mathy/commit/1a28a9d9c54c39189eced3f8c10ac1751a0c932e))
* **build:** add tools/clean.sh ([e3c2308](https://github.com/justindujardin/mathy/commit/e3c230848c5642310e97deab86436f46d27cdcfb))
* **cli:** add `--lr` for setting adam learning rate ([427352f](https://github.com/justindujardin/mathy/commit/427352fce3c942650ed38374f0391749b4197426))
* **config:** add prediction_window_size ([d4095c5](https://github.com/justindujardin/mathy/commit/d4095c54fe962b40c7c2b981f83044ceeba0bbad))
* **envs:** rebalance poly/complex difficulties ([88a9b30](https://github.com/justindujardin/mathy/commit/88a9b30f0dcaf45c0d1dd2e5a3c36b087580aadf))
* **mathy_alpha_sm:** add pretrained model with simplified architecture ([5365a26](https://github.com/justindujardin/mathy/commit/5365a269c4dfa26059c84441f088683412ef09c7))
* add mathy.example helper for generating inputs ([92695d6](https://github.com/justindujardin/mathy/commit/92695d6724b1324aae39d9c589baaf512665a068))
* **MathyWindowObservation:** add option to return inputs using numpy instead of tf.Tensor ([0c76609](https://github.com/justindujardin/mathy/commit/0c766096bbe7eaaefb26b108f148c682c6e3447c))
* **tensorflow:** update to 2.1.0 ([95e764e](https://github.com/justindujardin/mathy/commit/95e764ef96ecd131c6012e1f822bd5d665479488))

## [0.6.2](https://github.com/justindujardin/mathy/compare/v0.6.1...v0.6.2) (2020-01-25)


### Bug Fixes

* **types:** don't shadow mathy with a py file ([a7c558c](https://github.com/justindujardin/mathy/commit/a7c558c3caf3b92db5e184d66aea0004d1ab49b0))


### Features

* add py.typed file to manifest ([3f955b1](https://github.com/justindujardin/mathy/commit/3f955b121da2f7fbe5e9b45458f5c13d0a591ee5))

## [0.6.1](https://github.com/justindujardin/mathy/compare/v0.6.0...v0.6.1) (2020-01-13)


### Features

* **config:** add print_model_call_times option ([47ad597](https://github.com/justindujardin/mathy/commit/47ad597b062d79381428cfaa4d719c57d3581cc5))

## [0.6.0](https://github.com/justindujardin/mathy/compare/v0.5.3...v0.6.0) (2020-01-05)


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
