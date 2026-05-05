{
  description = "madnis gammaboard API - Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        libs = with pkgs; [
          stdenv.cc.cc.lib
          zlib
          gmp
          mpfr
          libmpc
          python313
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            gcc
            maturin
            rustc
            cargo
            rust-analyzer
            clippy
            rustfmt
            # Python tooling
            python313Packages.pip
            python313Packages.virtualenv
          ];

          buildInputs = libs;

          shellHook = ''
            # 1. Handle LD_LIBRARY_PATH for compiled dependencies
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath libs}:/run/opengl-driver/lib:$LD_LIBRARY_PATH"

            # 2. Setup Python Virtual Environment
            # This keeps your Nix store clean and uses your pyproject.toml
            if [ ! -d ".venv" ]; then
              echo "Creating new virtual environment..."
              python -m venv .venv
            fi
            
            source .venv/bin/activate

            # 3. Install dependencies from pyproject.toml
            # This will handle numpy, torch, and the git-based madnis install
            pip install --upgrade pip
            pip install -e . 
          '';
        };
      }
    );
}