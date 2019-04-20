Mathy Training Infrastructure
---

This folder contains various terraform/packer configurations for training Mathy. There is duplication and not a ton of documentation, but generally they are divided into small (1GPU 8CPUs) medium (24CPUs 1GPU) and monster (24CPUs and 1 Tesla T4 GPU)

None of these configurations have been used or tuned for usage in the latest mathy model. The original model based on AlphaZeroGeneral did well with these configurations, but the Mathy model hasn't performed noticeably better than the CPU training. It's still worth investigating making these work with Mathy going forward because it could greatly speed up training time.
