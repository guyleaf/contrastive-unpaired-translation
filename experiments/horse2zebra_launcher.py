from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def common_options(self):
        return [
            # Command 0
            Options(
                dataroot="./datasets/horse2zebra",
                name="horse2zebra_CUT",
                CUT_mode="CUT",
                display_env="horse2zebra_CUT",
            ),
            # Command 1
            Options(
                dataroot="./datasets/horse2zebra",
                name="horse2zebra_FastCUT_lambda_NCE_8",
                CUT_mode="FastCUT",
                display_env="horse2zebra_FastCUT_lambda_NCE_8",
                lambda_NCE=8.0,
            ),
        ]

    def commands(self):
        return ["python train.py " + str(opt) for opt in self.common_options()]

    def test_commands(self):
        return [
            "python test.py " + str(opt.set(num_test=500).remove("display_env"))
            for opt in self.common_options()
        ]
