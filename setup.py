from setuptools import setup,find_packages

setup(
    name               = 'sd_embed'
    , version          = '1.240829.1'
    , license          = 'Apache License'
    , author           = "Andrew Zhu"
    , author_email     = 'xhinker@hotmail.com'
    , packages         = find_packages('src')
    , package_dir      = {'': 'src'}
    , url              = 'https://github.com/xhinker/sd_embed'
    , keywords         = 'diffusers stable-diffusion embedding'
    , install_requires = [
        'optimum-quanto'
        , 'torch'
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
        , 'safetensors'
    ]
    , include_package_data=True
)