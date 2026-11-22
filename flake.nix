{
  description = "Los Podadores automation and RLlib workspace";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-26.05";
    nixpkgs-unstable.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      nixpkgs-unstable,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        unstable = import nixpkgs-unstable {
          inherit system;
          config.allowUnfree = true;
        };
        ldLibraries = with pkgs; [
          stdenv.cc.cc.lib
          zlib
          zstd
          openssl
          glib
          libxcb
          libglvnd
          libGL
          SDL2
          SDL2_image
          SDL2_mixer
          SDL2_ttf
          alsa-lib
          libX11
          wayland
          libxkbcommon
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          name = "los-podadores-dev";
          packages = with pkgs; [
            uv
            ruff

            pkg-config
            freetype
            sdl2-compat

            gcc
            cmake
            ninja
            bazel_8

            distrobox
          ];

          buildInputs = [ pkgs.bashInteractive ];

          shellHook = ''
            export PATH="$PWD/scripts:$PATH"
            unset PYTHONPATH
          '';

          env = {
            UV_PYTHON_PREFERENCE = "only-managed";
            NIX_LD = "${pkgs.stdenv.cc.bintools.dynamicLinker}";
            NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath ldLibraries;
          };
        };
      }
    );
}
