{
  description = "A flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
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
      in
      {
        devShells.default = pkgs.mkShell {
          LD_LIBRARY_PATH =
            with pkgs;
            lib.makeLibraryPath [
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
          packages = with pkgs;
            [
              uv
              nixd
              ruff
              bun

              pkg-config
              freetype
              sdl2-compat

              gcc
              cmake
            ];
          buildInputs = [ pkgs.bashInteractive ];
          env = {
          };
          shellHook = "";
        };
      }
    );
}
