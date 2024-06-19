from setuptools import setup,find_packages

setup(
    name               = 'sd_embed'
    , version          = '1.240618.2'
    , license          = 'Apache License'
    , author           = "Andrew Zhu"
    , author_email     = 'xhinker@hotmail.com'
    , packages         = find_packages('src')
    , package_dir      = {'': 'src'}
    , url              = 'https://github.com/xhinker/sd_embed'
    , keywords         = 'diffusers stable-diffusion embedding'
    , install_requires = [
        'torch'
        , 'torchvision'
        , 'torchaudio'
        , 'sentencepiece'
        , 'accelerate'
        , 'peft'
        , 'transformers'
        , 'diffusers'
        , 'lark'
        , 'protobuf'
        , 'ipykernel'
        , 'ipywidgets'
    ]
    , include_package_data=True
)