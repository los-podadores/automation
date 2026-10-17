{
  description = "Los Podadores automation and RLlib workspace";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
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
        libs = with pkgs; [
          stdenv.cc.cc
          zlib
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
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
          packages = with pkgs; [
            uv
            ruff
            bun

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

          env = { };
        };
      }
    );
}
