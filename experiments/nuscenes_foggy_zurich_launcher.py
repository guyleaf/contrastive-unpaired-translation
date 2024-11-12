import os

from PIL import Image

from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def common_options(self):
        name = "nuscenes_foggy_zurich"
        dataroot = os.path.expanduser("~/data/Weather/nuScenes_Foggy_Zurich_unpaired")
        weathers = ["rainy", "foggy"]

        # Maximum training image size in RTX 3090 24GB: 768
        # issue: https://github.com/taesungp/contrastive-unpaired-translation/issues/139#issuecomment-1165577005
        max_crop_size = 768
        max_load_size = round(max_crop_size * 1.1)
        resampling = Image.Resampling.LANCZOS.name

        return [
            Options(
                dataroot=dataroot,
                dataset_mode="unaligned_weather",
                data_domainA="clear",
                data_domainB=weather,
                name=f"{name}_CUT_{weather}_load_{max_load_size}_crop_{max_crop_size}",
                CUT_mode="CUT",
                display_env=f"{name}_CUT_{weather}_load_{max_load_size}_crop_{max_crop_size}",
                max_load_size=max_load_size,
                max_crop_size=max_crop_size,
                resampling=resampling,
            )
            for weather in weathers
        ]

    def commands(self):
        lr = 0.0002
        lambda_GAN = 1.0
        n_epochs = 200
        n_epochs_decay = 200
        batch_size = 1

        return [
            "python train.py "
            + str(
                opt.set(
                    name=f"{opt.kvs['name']}_epochs_{n_epochs + n_epochs_decay}_bs_{batch_size}_no_seed",
                    display_env=f"{opt.kvs['name']}_epochs_{n_epochs + n_epochs_decay}_bs_{batch_size}_no_seed",
                    # data_domainA=os.path.join("train", opt.kvs["data_domainA"]),
                    # data_domainB=os.path.join("train", opt.kvs["data_domainB"]),
                    lr=lr,
                    lambda_GAN=lambda_GAN,
                    n_epochs=n_epochs,
                    n_epochs_decay=n_epochs_decay,
                    batch_size=batch_size,
                )
            )
            for opt in self.common_options()
        ]

    def test_commands(self):
        return [
            "python test.py "
            + str(
                opt.remove("display_env").set(
                    # name="nuscenes_foggy_zurich_CUT_foggy_load_563_crop_512_epochs_100_bs_2_no_seed",
                    name="nuscenes_foggy_zurich_CUT_foggy_load_563_crop_512_epochs_200_bs_2_no_seed",
                    # data_domainA=os.path.join("val", opt.kvs["data_domainA"]),
                    # data_domainB=os.path.join("val", opt.kvs["data_domainB"]),
                    num_test=1000,
                )
            )
            for opt in self.common_options()
        ]
