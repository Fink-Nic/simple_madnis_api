# Scuffed flake setup
### Set up the venv 
Set the flake
```shell
cp backup/use_to_create_venv_flake flake.nix
```
Use the flake
```shell
nix develop
```
### Run the test
```shell
python3 gaussian_test.py
```

### Swap to GammaBoard flake
```shell
cp backup/gammaboard_flake flake.nix
```

### GammaBoard test run

* `madnis_test_run.toml`
