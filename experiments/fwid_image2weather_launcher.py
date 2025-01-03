import os

from PIL import Image

from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def common_options(self):
        name = "fwid_image2weather"
        dataroot = os.path.expanduser(
            "~/data/Weather/FWID_Image2Weather_unpaired/dataset"
        )
        weathers = ["foggy", "rainy", "snowy"]

        # Maximum training image size in RTX 3090 24GB
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
                # name=f"{name}_CUT_{weather}_load_{max_load_size}_crop_{max_crop_size}_epochs_{n_epochs*2}_no_seed",
                name=f"{name}_CUT_{weather}_load_{max_load_size}_crop_{max_crop_size}",
                CUT_mode="CUT",
                # display_env=f"{name}_CUT_{weather}_load_{max_load_size}_crop_{max_crop_size}_epochs_{n_epochs*2}_no_seed",
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

        return [
            "python train.py "
            + str(
                opt.set(
                    data_domainA=os.path.join("train", opt.kvs["data_domainA"]),
                    data_domainB=os.path.join("train", opt.kvs["data_domainB"]),
                    lr=lr,
                    lambda_GAN=lambda_GAN,
                    n_epochs=n_epochs,
                    n_epochs_decay=n_epochs,
                )
            )
            for opt in self.common_options()
        ]

    def test_commands(self):
        return [
            "python test.py "
            + str(
                opt.remove("display_env").set(
                    data_domainA=os.path.join("val", opt.kvs["data_domainA"]),
                    data_domainB=os.path.join("val", opt.kvs["data_domainB"]),
                    num_test=1000,
                    name="fwid_image2weather_CUT_snowy_load_845_crop_768_lambda_gan_12e-1_no_seed",
                    # preprocess="scale_shortside",
                    # eval=True,
                )
            )
            for opt in self.common_options()
        ]
