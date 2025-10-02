# rtsl
Real Time Sign Language - An app to translate sign language in real time

## Backend Setup
Please [install pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation), which we will use to make sure we are all on the same Python version. It lets you set local versions, so that you do not need to mess with your system version (and potentially break other projects you have).

Todo: create a "run.sh" that handles setting up venv.
```
# install pyenv if needed
pyenv install --skip-existing 3.13.3
pyenv virtualenv 3.13.3 rtsl-env
pyenv local rtsl-env
pip install -r requirements.txt
```